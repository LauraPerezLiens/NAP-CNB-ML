#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Multimodal Logistic Regression Baselines — comparable LR/MLP version
===============================================================================

Author
------
Laura Pérez Liens

Description
-----------
Train seven streaming Logistic Regression baselines using the common
``dataset.py`` loader.

This version is designed to be directly comparable with ``mlp_baseline.py``:

- both models use the same ``master_split.csv`` through ``dataset.py``;
- both fit ``StandardScaler`` ONLY on TRAIN;
- both train on TRAIN;
- both select the best epoch using UNWEIGHTED validation log loss on VALIDATION;
- both evaluate final metrics ONLY on TEST;
- both use the same threshold for threshold-dependent metrics;
- both use the same balanced-class definition;
- both report exact ROC-AUC and Average Precision from disk-backed memmaps.

The TEST split is never used for model selection.
===============================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from dataset import (
    TRAIN,
    VALIDATION,
    TEST,
    Feature,
    build_input_matrix,
    discover_datasets,
    feature_combination_name,
    iterate_dataset,
    print_datasets,
)


# =============================================================================
# Experiment definitions
# =============================================================================

EXPERIMENTS: Tuple[Tuple[Feature, ...], ...] = (
    (Feature.PRIMARY,),
    (Feature.POSITION,),
    (Feature.SECONDARY,),
    (Feature.PRIMARY, Feature.POSITION),
    (Feature.PRIMARY, Feature.SECONDARY),
    (Feature.POSITION, Feature.SECONDARY),
    (Feature.PRIMARY, Feature.POSITION, Feature.SECONDARY),
)


# =============================================================================
# Configuration containers
# =============================================================================

@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    patience: int
    min_delta: float
    learning_rate: str
    eta0: float
    alpha: float
    penalty: str
    l1_ratio: float
    class_weight: str
    random_state: int
    threshold: float
    csv_chunk_size: int
    secondary_cache_size: int


@dataclass
class SplitCounts:
    total: int = 0
    negative: int = 0
    positive: int = 0

    def update(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.uint8).reshape(-1)
        positive = int(np.count_nonzero(y == 1))
        self.total += int(len(y))
        self.positive += positive
        self.negative += int(len(y) - positive)


# =============================================================================
# CLI / utilities
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming Logistic Regression baselines comparable with MLP."
    )

    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="outputs/logistic_baselines")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        help="Use 'all' or names such as primary, primary+position, secondary.",
    )
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--learning-rate",
        choices=["optimal", "constant", "invscaling", "adaptive"],
        default="optimal",
    )
    parser.add_argument("--eta0", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--penalty", choices=["l2", "l1", "elasticnet"], default="l2")
    parser.add_argument("--l1-ratio", type=float, default=0.15)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--csv-chunk-size", type=int, default=100_000)
    parser.add_argument("--secondary-cache-size", type=int, default=8)
    parser.add_argument(
        "--keep-evaluation-predictions",
        action="store_true",
        help="Keep validation/test memmap prediction files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative.")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.csv_chunk_size < 1:
        raise ValueError("--csv-chunk-size must be positive.")
    if args.secondary_cache_size < 1:
        raise ValueError("--secondary-cache-size must be positive.")


def select_experiments(names: Sequence[str]) -> List[Tuple[Feature, ...]]:
    aliases = {feature_combination_name(features): features for features in EXPERIMENTS}
    normalized_names = [name.strip().lower() for name in names]

    if "all" in normalized_names:
        return list(EXPERIMENTS)

    selected: List[Tuple[Feature, ...]] = []
    for name in normalized_names:
        canonical = name.replace(",", "+").replace(" ", "").replace("distance", "position")
        if canonical not in aliases:
            raise ValueError(
                f"Unknown experiment {name!r}. Valid experiments: {', '.join(aliases)}"
            )
        features = aliases[canonical]
        if features not in selected:
            selected.append(features)
    return selected


def save_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_history(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def iter_batches(
    datasets: Sequence[Mapping[str, object]],
    split: int,
    features: Sequence[Feature],
    config: TrainingConfig,
):
    for dataset in datasets:
        for batch in iterate_dataset(
            dataset,
            split=split,
            features=features,
            csv_chunk_size=config.csv_chunk_size,
            secondary_cache_size=config.secondary_cache_size,
        ):
            yield batch


def count_split_rows(
    datasets: Sequence[Mapping[str, object]],
    split: int,
    csv_chunk_size: int,
) -> SplitCounts:
    """Count rows directly from master_split.csv without loading embeddings."""
    counts = SplitCounts()
    for dataset in datasets:
        reader = pd.read_csv(
            Path(dataset["master_split"]),
            usecols=["y", "split_id"],
            chunksize=csv_chunk_size,
        )
        for block in reader:
            y = block.loc[block["split_id"] == split, "y"].to_numpy(
                dtype=np.uint8,
                copy=False,
            )
            counts.update(y)
    return counts


def class_sample_weights(
    y: np.ndarray,
    counts: SplitCounts,
    mode: str,
) -> Optional[np.ndarray]:
    """Balanced weights used identically by the LR and MLP implementations."""
    if mode == "none":
        return None

    if counts.negative == 0 or counts.positive == 0:
        raise RuntimeError("Balanced class weights require both classes in TRAIN.")

    negative_weight = counts.total / (2.0 * counts.negative)
    positive_weight = counts.total / (2.0 * counts.positive)

    return np.where(
        np.asarray(y) == 1,
        positive_weight,
        negative_weight,
    ).astype(np.float64, copy=False)


# =============================================================================
# Scaling
# =============================================================================

def fit_scaler(
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    config: TrainingConfig,
) -> Tuple[StandardScaler, SplitCounts]:
    """Fit StandardScaler ONLY on TRAIN and collect TRAIN class counts."""
    scaler = StandardScaler()
    counts = SplitCounts()
    seen_batch = False

    print("  Fitting StandardScaler on TRAIN...")

    for batch_number, batch in enumerate(
        iter_batches(datasets, TRAIN, features, config),
        start=1,
    ):
        X = build_input_matrix(batch, features, dtype=np.float32)
        y = batch["labels"]
        scaler.partial_fit(X)
        counts.update(y)
        seen_batch = True

        if batch_number % 100 == 0:
            print(
                f"\r    scaler batches={batch_number:,} samples={counts.total:,}",
                end="",
                flush=True,
            )

    print()

    if not seen_batch:
        raise RuntimeError("The TRAIN split contains no samples.")

    print(
        f"    train samples={counts.total:,}; negative={counts.negative:,}; "
        f"positive={counts.positive:,}; prevalence={counts.positive / counts.total:.6f}"
    )
    return scaler, counts


# =============================================================================
# Metrics
# =============================================================================

def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_confusion(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    total = tn + fp + fn + tp
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    balanced_accuracy = (recall + specificity) / 2.0

    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    f2 = safe_divide(5.0 * precision * recall, 4.0 * precision + recall)

    mcc_denominator = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = safe_divide(tp * tn - fp * fn, mcc_denominator)

    observed = accuracy
    expected = safe_divide(
        (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn),
        total * total,
    )
    kappa = safe_divide(observed - expected, 1.0 - expected)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1": f1,
        "f2": f2,
        "matthews_correlation_coefficient": mcc,
        "cohen_kappa": kappa,
        "prevalence": safe_divide(tp + fn, total),
        "predicted_positive_rate": safe_divide(tp + fp, total),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def exact_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = metrics_from_confusion(int(tn), int(fp), int(fn), int(tp))
    metrics.update(
        {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "average_precision": float(average_precision_score(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "n_samples": int(len(y_true)),
            "auc_is_approximate": False,
        }
    )

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    curves = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_curve,
        "recall": recall_curve,
    }
    return metrics, curves


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    classifier: SGDClassifier,
    scaler: StandardScaler,
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    config: TrainingConfig,
    work_dir: Path,
    split: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate exact metrics on VALIDATION or TEST using disk-backed arrays."""
    if split not in (VALIDATION, TEST):
        raise ValueError("evaluate_model is intended for VALIDATION or TEST.")

    split_name = "validation" if split == VALIDATION else "test"
    work_dir.mkdir(parents=True, exist_ok=True)

    split_counts = count_split_rows(datasets, split, config.csv_chunk_size)
    if split_counts.total == 0:
        raise RuntimeError(f"The {split_name.upper()} split contains no samples.")

    y_true_path = work_dir / f"{split_name}_y_true.uint8.mmap"
    y_prob_path = work_dir / f"{split_name}_y_probability.float32.mmap"

    y_true_memmap = np.memmap(
        y_true_path, dtype=np.uint8, mode="w+", shape=(split_counts.total,)
    )
    y_prob_memmap = np.memmap(
        y_prob_path, dtype=np.float32, mode="w+", shape=(split_counts.total,)
    )

    offset = 0
    for batch in iter_batches(datasets, split, features, config):
        X = build_input_matrix(batch, features, dtype=np.float32)
        X = scaler.transform(X)
        probabilities = classifier.predict_proba(X)[:, 1]
        labels = np.asarray(batch["labels"], dtype=np.uint8)
        end = offset + len(labels)
        y_true_memmap[offset:end] = labels
        y_prob_memmap[offset:end] = probabilities
        offset = end

    y_true_memmap.flush()
    y_prob_memmap.flush()

    if offset != split_counts.total:
        raise RuntimeError(
            f"{split_name} row mismatch: expected {split_counts.total}, processed {offset}."
        )

    metrics, curves = exact_binary_metrics(
        y_true_memmap,
        y_prob_memmap,
        config.threshold,
    )

    del y_true_memmap
    del y_prob_memmap
    gc.collect()
    return metrics, curves


# =============================================================================
# Training
# =============================================================================

def build_classifier(config: TrainingConfig) -> SGDClassifier:
    kwargs = {
        "loss": "log_loss",
        "penalty": config.penalty,
        "alpha": config.alpha,
        "learning_rate": config.learning_rate,
        "eta0": config.eta0,
        "random_state": config.random_state,
        "fit_intercept": True,
        "average": True,
    }
    if config.penalty == "elasticnet":
        kwargs["l1_ratio"] = config.l1_ratio
    return SGDClassifier(**kwargs)


def train_one_epoch(
    classifier: SGDClassifier,
    scaler: StandardScaler,
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    train_counts: SplitCounts,
    config: TrainingConfig,
) -> Dict[str, float]:
    """One streaming training epoch over TRAIN."""
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for batch_number, batch in enumerate(
        iter_batches(datasets, TRAIN, features, config),
        start=1,
    ):
        X = build_input_matrix(batch, features, dtype=np.float32)
        X = scaler.transform(X)
        y = np.asarray(batch["labels"], dtype=np.uint8)

        if hasattr(classifier, "classes_"):
            probability = classifier.predict_proba(X)[:, 1]
            prediction = probability >= config.threshold
            total_loss += float(
                log_loss(y, probability, labels=[0, 1], normalize=False)
            )
            correct += int(np.sum(prediction == y))

        weights = class_sample_weights(y, train_counts, config.class_weight)
        classifier.partial_fit(
            X,
            y,
            classes=np.array([0, 1], dtype=np.uint8),
            sample_weight=weights,
        )
        total_samples += len(y)

        if batch_number % 100 == 0:
            print(
                f"\r    train batches={batch_number:,} samples={total_samples:,}",
                end="",
                flush=True,
            )

    print()
    return {
        "train_monitor_loss": (
            total_loss / total_samples if total_samples and total_loss else float("nan")
        ),
        "train_monitor_accuracy": (
            correct / total_samples
            if total_samples and hasattr(classifier, "classes_")
            else float("nan")
        ),
        "train_samples": total_samples,
    }


# =============================================================================
# Plots / outputs
# =============================================================================

def save_curves(
    output_dir: Path,
    curves: Mapping[str, np.ndarray],
    experiment_name: str,
    prefix: str,
) -> None:
    np.savez_compressed(output_dir / f"{prefix}_curves.npz", **curves)

    plt.figure(figsize=(7, 6))
    plt.plot(curves["fpr"], curves["tpr"], label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC — {experiment_name} ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(curves["recall"], curves["precision"], label="Precision–Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall — {experiment_name} ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_precision_recall_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix_plot(
    output_dir: Path,
    metrics: Mapping[str, object],
    experiment_name: str,
) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        dtype=np.int64,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            plt.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    plt.title(f"Confusion matrix — {experiment_name} (TEST)")
    plt.tight_layout()
    plt.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close()


def train_experiment(
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    output_root: Path,
    config: TrainingConfig,
    keep_evaluation_predictions: bool,
) -> Dict[str, object]:
    experiment_name = feature_combination_name(features)
    experiment_dir = output_root / experiment_name
    work_dir = experiment_dir / "_evaluation_work"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 100)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 100)

    save_json(
        experiment_dir / "configuration.json",
        {
            "experiment": experiment_name,
            "features": [feature.value for feature in features],
            **asdict(config),
        },
    )

    scaler, train_counts = fit_scaler(datasets, features, config)
    validation_counts = count_split_rows(datasets, VALIDATION, config.csv_chunk_size)
    test_counts = count_split_rows(datasets, TEST, config.csv_chunk_size)

    print(
        f"  split counts: TRAIN={train_counts.total:,}, "
        f"VALIDATION={validation_counts.total:,}, TEST={test_counts.total:,}"
    )

    classifier = build_classifier(config)
    best_classifier: Optional[SGDClassifier] = None
    best_validation_metrics: Optional[Dict[str, float]] = None
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.time()
        print()
        print(f"  Epoch {epoch}/{config.max_epochs}")

        train_metrics = train_one_epoch(
            classifier,
            scaler,
            datasets,
            features,
            train_counts,
            config,
        )

        validation_metrics, validation_curves = evaluate_model(
            classifier,
            scaler,
            datasets,
            features,
            config,
            work_dir,
            split=VALIDATION,
        )

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(epoch_record)
        save_history(experiment_dir / "training_history.csv", history)

        print(
            f"    validation_log_loss={validation_metrics['log_loss']:.6f}; "
            f"ROC-AUC={validation_metrics['roc_auc']:.6f}; "
            f"AP={validation_metrics['average_precision']:.6f}; "
            f"F1={validation_metrics['f1']:.6f}; "
            f"MCC={validation_metrics['matthews_correlation_coefficient']:.6f}"
        )

        current_loss = float(validation_metrics["log_loss"])
        if current_loss < best_loss - config.min_delta:
            best_loss = current_loss
            best_epoch = epoch
            best_classifier = copy.deepcopy(classifier)
            best_validation_metrics = dict(validation_metrics)
            epochs_without_improvement = 0

            joblib.dump(
                {
                    "classifier": best_classifier,
                    "scaler": scaler,
                    "features": [feature.value for feature in features],
                    "threshold": config.threshold,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                },
                experiment_dir / "best_model.joblib",
                compress=3,
            )
            save_curves(experiment_dir, validation_curves, experiment_name, "validation_best")
            print("    New best model saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"    No improvement ({epochs_without_improvement}/{config.patience})."
            )
            if epochs_without_improvement >= config.patience:
                print("    Early stopping.")
                break

    if best_classifier is None or best_validation_metrics is None:
        raise RuntimeError(f"No valid model produced for {experiment_name}.")

    print()
    print(
        f"  Evaluating TEST with best checkpoint: epoch={best_epoch}, "
        f"validation_log_loss={best_loss:.6f}"
    )

    test_metrics, test_curves = evaluate_model(
        best_classifier,
        scaler,
        datasets,
        features,
        config,
        work_dir,
        split=TEST,
    )

    final_metrics: Dict[str, object] = {
        "experiment": experiment_name,
        "features": "+".join(feature.value for feature in features),
        "model": "logistic_regression",
        "class_weight": config.class_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        **test_metrics,
        "train_samples": train_counts.total,
        "train_negative": train_counts.negative,
        "train_positive": train_counts.positive,
        "validation_samples": validation_counts.total,
        "test_samples": test_counts.total,
        "threshold": config.threshold,
        "random_state": config.random_state,
    }

    save_json(experiment_dir / "test_metrics.json", final_metrics)
    pd.DataFrame([final_metrics]).to_csv(experiment_dir / "test_metrics.csv", index=False)
    save_curves(experiment_dir, test_curves, experiment_name, "test")
    save_confusion_matrix_plot(experiment_dir, test_metrics, experiment_name)

    if not keep_evaluation_predictions and work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"  Finished {experiment_name}; best epoch={best_epoch}.")
    return final_metrics


def save_comparison(
    output_root: Path,
    results: Sequence[Mapping[str, object]],
) -> None:
    frame = pd.DataFrame(results)
    preferred_columns = [
        "experiment",
        "features",
        "model",
        "class_weight",
        "best_epoch",
        "best_validation_loss",
        "n_samples",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "negative_predictive_value",
        "f1",
        "f2",
        "matthews_correlation_coefficient",
        "cohen_kappa",
        "roc_auc",
        "average_precision",
        "log_loss",
        "brier_score",
        "prevalence",
        "predicted_positive_rate",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "train_samples",
        "train_negative",
        "train_positive",
        "validation_samples",
        "test_samples",
        "threshold",
        "auc_is_approximate",
    ]
    ordered = [column for column in preferred_columns if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    frame = frame[ordered + remaining]
    frame.to_csv(output_root / "model_comparison.csv", index=False)

    ranking = frame.sort_values(
        ["average_precision", "roc_auc", "matthews_correlation_coefficient"],
        ascending=False,
    )
    ranking.to_csv(output_root / "model_ranking.csv", index=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    validate_args(args)

    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        learning_rate=args.learning_rate,
        eta0=args.eta0,
        alpha=args.alpha,
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        class_weight=args.class_weight,
        random_state=args.random_state,
        threshold=args.threshold,
        csv_chunk_size=args.csv_chunk_size,
        secondary_cache_size=args.secondary_cache_size,
    )

    datasets = discover_datasets(args.root)
    if not datasets:
        raise RuntimeError(
            f"No datasets with master_split.csv and ProtT5 chunks found below {args.root}."
        )

    print_datasets(datasets)
    experiments = select_experiments(args.experiments)

    print("Experiments:")
    for features in experiments:
        print(f"  - {feature_combination_name(features)}")

    results: List[Dict[str, object]] = []
    for features in experiments:
        result = train_experiment(
            datasets=datasets,
            features=features,
            output_root=output_root,
            config=config,
            keep_evaluation_predictions=args.keep_evaluation_predictions,
        )
        results.append(result)
        save_comparison(output_root, results)

    print()
    print("=" * 100)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 100)
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()