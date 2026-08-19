#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
MLP Focal Baseline — comparable LR/MLP version
===============================================================================

Author
------
Laura Pérez Liens

Description
-----------
GPU-ready multilayer perceptron baseline using Focal Loss for the multimodal
NAP-CNB datasets.

Architecture
------------
    input_dim -> 512 -> ReLU -> 128 -> ReLU -> 1

This script is designed to remain directly comparable with the final
``logistic_baselines.py`` and ``mlp_baseline.py``:

- uses the same ``dataset.py`` and ``master_split.csv``;
- fits ``StandardScaler`` ONLY on TRAIN;
- trains ONLY on TRAIN;
- uses Focal Loss only for TRAIN;
- selects the best epoch using UNWEIGHTED validation log loss on VALIDATION;
- evaluates final metrics ONLY on TEST;
- uses the same fixed threshold;
- reports exact ROC-AUC and Average Precision using disk-backed memmaps;
- keeps the same MLP architecture, optimizer, batch size and preprocessing
  conventions as the comparable MLP baseline.

The TEST split is never used for model selection.

Focal Loss
----------
For each sample:

    FL = alpha_t * (1 - p_t)^gamma * BCE(logit, target)

where:

- ``alpha`` is the weight assigned to the positive class;
- ``1 - alpha`` is assigned to the negative class;
- ``gamma`` controls the focusing effect on hard examples.

Both ``alpha`` and ``gamma`` are configurable from the command line.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from torch import nn

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
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Experiment definitions
# =============================================================================

EXPERIMENTS: Dict[str, List[Feature]] = {
    "primary": [Feature.PRIMARY],
    "position": [Feature.POSITION],
    "secondary": [Feature.SECONDARY],
    "primary_position": [Feature.PRIMARY, Feature.POSITION],
    "primary_secondary": [Feature.PRIMARY, Feature.SECONDARY],
    "position_secondary": [Feature.POSITION, Feature.SECONDARY],
    "all": [Feature.PRIMARY, Feature.POSITION, Feature.SECONDARY],
}


def parse_experiments(values: Sequence[str]) -> List[str]:
    """
    Validate experiment names.

    When ``all`` is requested, run every available experiment.
    """
    if not values or "all" in values:
        return list(EXPERIMENTS)

    unknown = [
        value
        for value in values
        if value not in EXPERIMENTS
    ]

    if unknown:
        raise ValueError(
            "Unknown experiment(s): "
            + ", ".join(unknown)
            + ". Valid experiments: "
            + ", ".join(EXPERIMENTS)
        )

    return list(dict.fromkeys(values))


# =============================================================================
# Model
# =============================================================================

class MLPBaseline(nn.Module):
    """Simple MLP: input -> hidden1 -> ReLU -> hidden2 -> ReLU -> 1 logit."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 512,
        hidden_dim_2: int = 128,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# =============================================================================
# Focal Loss
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss operating directly on logits.

    Parameters
    ----------
    alpha
        Weight assigned to positive examples. Negative examples receive
        ``1 - alpha``. Must be in [0, 1].

    gamma
        Focusing parameter. ``gamma=0`` removes the focusing term.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Focal alpha must be between 0 and 1.")

        if gamma < 0.0:
            raise ValueError("Focal gamma must be non-negative.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probabilities = torch.sigmoid(logits)

        p_t = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_factor = (1.0 - p_t).pow(self.gamma)

        return (alpha_t * focal_factor * bce).mean()


# =============================================================================
# Data helpers
# =============================================================================

def iter_minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield consecutive minibatches from one streamed block."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        yield X[start:end], y[start:end]


def get_input_dim(
    dataset: Mapping[str, object],
    features: Sequence[Feature],
    split: int = TRAIN,
    secondary_cache_size: int = 8,
) -> int:
    """Infer total input dimensionality from the first non-empty batch."""
    for batch in iterate_dataset(
        dataset,
        split=split,
        features=features,
        secondary_cache_size=secondary_cache_size,
    ):
        X = build_input_matrix(
            batch,
            features=features,
            dtype=np.float32,
        )

        if len(X) == 0:
            continue

        if X.ndim != 2:
            raise RuntimeError(
                f"Expected 2D input matrix, found shape {X.shape}."
            )

        return int(X.shape[1])

    raise RuntimeError(
        f"No samples found in split={split} "
        f"for dataset {dataset['name']!r}."
    )


@dataclass
class SplitCounts:
    """Class counts for a dataset split."""

    total: int = 0
    negative: int = 0
    positive: int = 0

    def update(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=np.uint8).reshape(-1)
        positive = int(np.count_nonzero(y == 1))

        self.total += int(len(y))
        self.positive += positive
        self.negative += int(len(y) - positive)


def count_split_rows(
    datasets: Sequence[Mapping[str, object]],
    split: int,
    csv_chunk_size: int,
) -> SplitCounts:
    """Count split rows directly from master_split.csv."""
    counts = SplitCounts()

    for dataset in datasets:
        reader = pd.read_csv(
            Path(dataset["master_split"]),
            usecols=["y", "split_id"],
            chunksize=csv_chunk_size,
        )

        for block in reader:
            y = block.loc[
                block["split_id"] == split,
                "y",
            ].to_numpy(
                dtype=np.uint8,
                copy=False,
            )
            counts.update(y)

    return counts


# =============================================================================
# Scaling — same preprocessing as LR and standard MLP
# =============================================================================

def fit_scaler(
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    csv_chunk_size: int,
    secondary_cache_size: int,
) -> Tuple[StandardScaler, SplitCounts]:
    """Fit StandardScaler ONLY on TRAIN and collect TRAIN class counts."""
    scaler = StandardScaler()
    counts = SplitCounts()
    seen_batch = False
    batch_number = 0

    print("Fitting StandardScaler on TRAIN...")

    for dataset in datasets:
        for block in iterate_dataset(
            dataset,
            split=TRAIN,
            features=features,
            csv_chunk_size=csv_chunk_size,
            secondary_cache_size=secondary_cache_size,
        ):
            batch_number += 1

            X = build_input_matrix(
                block,
                features=features,
                dtype=np.float32,
            )
            y = block["labels"]

            scaler.partial_fit(X)
            counts.update(y)
            seen_batch = True

            if batch_number % 100 == 0:
                print(
                    f"\r  scaler batches={batch_number:,} "
                    f"samples={counts.total:,}",
                    end="",
                    flush=True,
                )

    print()

    if not seen_batch:
        raise RuntimeError("The TRAIN split contains no samples.")

    print(
        f"train samples={counts.total:,} "
        f"negative={counts.negative:,} "
        f"positive={counts.positive:,} "
        f"prevalence={counts.positive / counts.total:.6f}"
    )

    return scaler, counts


# =============================================================================
# Training / validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    scaler: StandardScaler,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    csv_chunk_size: int,
    secondary_cache_size: int,
    log_every: int,
    epoch_seed: int,
) -> Tuple[float, int]:
    """Train one complete streamed epoch on TRAIN."""
    model.train()

    running_loss_sum = 0.0
    total_samples = 0
    minibatch_counter = 0

    rng = np.random.default_rng(epoch_seed)

    for dataset in datasets:
        print(f"  TRAIN dataset: {dataset['name']}")

        for block in iterate_dataset(
            dataset,
            split=TRAIN,
            features=features,
            csv_chunk_size=csv_chunk_size,
            secondary_cache_size=secondary_cache_size,
        ):
            X = build_input_matrix(
                block,
                features=features,
                dtype=np.float32,
            )
            X = scaler.transform(X).astype(
                np.float32,
                copy=False,
            )

            y = np.asarray(
                block["labels"],
                dtype=np.float32,
            )

            # Same block-level shuffling used by the comparable MLP baseline.
            permutation = rng.permutation(len(y))
            X = X[permutation]
            y = y[permutation]

            for xb_np, yb_np in iter_minibatches(
                X,
                y,
                batch_size,
            ):
                xb = torch.from_numpy(xb_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                yb = torch.from_numpy(yb_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                loss.backward()
                optimizer.step()

                n = len(yb_np)
                running_loss_sum += float(loss.detach().item()) * n
                total_samples += n
                minibatch_counter += 1

                if (
                    log_every > 0
                    and minibatch_counter % log_every == 0
                ):
                    print(
                        f"    minibatches={minibatch_counter:,} "
                        f"samples={total_samples:,} "
                        f"mean_loss="
                        f"{running_loss_sum / max(total_samples, 1):.6f}"
                    )

                del xb, yb, logits, loss

    if total_samples == 0:
        raise RuntimeError(
            "No TRAIN samples were seen during this epoch."
        )

    return running_loss_sum / total_samples, total_samples


@torch.inference_mode()
def evaluate_unweighted_log_loss(
    model: nn.Module,
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    scaler: StandardScaler,
    split: int,
    device: torch.device,
    batch_size: int,
    csv_chunk_size: int,
    secondary_cache_size: int,
) -> Tuple[float, int]:
    """
    Evaluate UNWEIGHTED BCE/log loss.

    Focal Loss is used only during TRAIN. Model selection therefore remains
    directly comparable with LR and the standard MLP.
    """
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    total_samples = 0

    for dataset in datasets:
        for block in iterate_dataset(
            dataset,
            split=split,
            features=features,
            csv_chunk_size=csv_chunk_size,
            secondary_cache_size=secondary_cache_size,
        ):
            X = build_input_matrix(
                block,
                features=features,
                dtype=np.float32,
            )
            X = scaler.transform(X).astype(
                np.float32,
                copy=False,
            )

            y = np.asarray(
                block["labels"],
                dtype=np.float32,
            )

            for xb_np, yb_np in iter_minibatches(
                X,
                y,
                batch_size,
            ):
                xb = torch.from_numpy(xb_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                yb = torch.from_numpy(yb_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                logits = model(xb)
                loss = criterion(logits, yb)

                n = len(yb_np)
                loss_sum += float(loss.item()) * n
                total_samples += n

                del xb, yb, logits, loss

    if total_samples == 0:
        raise RuntimeError(
            f"No samples found while evaluating split={split}."
        )

    return loss_sum / total_samples, total_samples


# =============================================================================
# Metrics
# =============================================================================

def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Safely divide two numbers."""
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> Dict[str, float]:
    """Compute threshold-dependent binary metrics."""
    total = tn + fp + fn + tp

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    balanced_accuracy = (recall + specificity) / 2.0

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    f2 = safe_divide(
        5.0 * precision * recall,
        4.0 * precision + recall,
    )

    mcc_denominator = math.sqrt(
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )

    mcc = safe_divide(
        tp * tn - fp * fn,
        mcc_denominator,
    )

    observed = accuracy

    expected = safe_divide(
        (tp + fp) * (tp + fn)
        + (fn + tn) * (fp + tn),
        total * total,
    )

    kappa = safe_divide(
        observed - expected,
        1.0 - expected,
    )

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
    """Compute exact TEST metrics and ROC/PR curves."""
    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    )
    y_prob = np.asarray(
        y_prob,
        dtype=np.float64,
    )

    y_pred = (
        y_prob >= threshold
    ).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = metrics_from_confusion(
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    )

    metrics.update(
        {
            "roc_auc": float(
                roc_auc_score(y_true, y_prob)
            ),
            "average_precision": float(
                average_precision_score(y_true, y_prob)
            ),
            "log_loss": float(
                log_loss(
                    y_true,
                    y_prob,
                    labels=[0, 1],
                )
            ),
            "brier_score": float(
                brier_score_loss(y_true, y_prob)
            ),
            "n_samples": int(len(y_true)),
            "auc_is_approximate": False,
        }
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true,
            y_prob,
        )
    )

    curves = {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision_curve,
        "recall": recall_curve,
    }

    return metrics, curves


@torch.inference_mode()
def evaluate_exact_metrics(
    model: nn.Module,
    datasets: Sequence[Mapping[str, object]],
    features: Sequence[Feature],
    scaler: StandardScaler,
    split: int,
    device: torch.device,
    batch_size: int,
    csv_chunk_size: int,
    secondary_cache_size: int,
    threshold: float,
    work_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate exact VALIDATION/TEST metrics using disk-backed arrays."""
    if split not in (VALIDATION, TEST):
        raise ValueError(
            "evaluate_exact_metrics is intended for VALIDATION or TEST."
        )

    split_name = (
        "validation"
        if split == VALIDATION
        else "test"
    )

    split_counts = count_split_rows(
        datasets,
        split,
        csv_chunk_size,
    )

    if split_counts.total == 0:
        raise RuntimeError(
            f"The {split_name.upper()} split contains no samples."
        )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true_path = (
        work_dir
        / f"{split_name}_y_true.uint8.mmap"
    )

    y_prob_path = (
        work_dir
        / f"{split_name}_y_probability.float32.mmap"
    )

    y_true_memmap = np.memmap(
        y_true_path,
        dtype=np.uint8,
        mode="w+",
        shape=(split_counts.total,),
    )

    y_prob_memmap = np.memmap(
        y_prob_path,
        dtype=np.float32,
        mode="w+",
        shape=(split_counts.total,),
    )

    model.eval()
    offset = 0

    for dataset in datasets:
        print(
            f"  {split_name.upper()} dataset: "
            f"{dataset['name']}"
        )

        for block in iterate_dataset(
            dataset,
            split=split,
            features=features,
            csv_chunk_size=csv_chunk_size,
            secondary_cache_size=secondary_cache_size,
        ):
            X = build_input_matrix(
                block,
                features=features,
                dtype=np.float32,
            )
            X = scaler.transform(X).astype(
                np.float32,
                copy=False,
            )

            y = np.asarray(
                block["labels"],
                dtype=np.uint8,
            )

            block_probabilities = np.empty(
                len(y),
                dtype=np.float32,
            )

            local_offset = 0

            for xb_np, _ in iter_minibatches(
                X,
                y,
                batch_size,
            ):
                xb = torch.from_numpy(xb_np).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                logits = model(xb)

                probabilities = (
                    torch.sigmoid(logits)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                end_local = (
                    local_offset
                    + len(probabilities)
                )

                block_probabilities[
                    local_offset:end_local
                ] = probabilities

                local_offset = end_local

                del xb, logits, probabilities

            end = offset + len(y)

            y_true_memmap[offset:end] = y
            y_prob_memmap[offset:end] = block_probabilities

            offset = end

    y_true_memmap.flush()
    y_prob_memmap.flush()

    if offset != split_counts.total:
        raise RuntimeError(
            f"{split_name} row mismatch: "
            f"expected {split_counts.total}, "
            f"processed {offset}."
        )

    metrics, curves = exact_binary_metrics(
        y_true_memmap,
        y_prob_memmap,
        threshold,
    )

    del y_true_memmap
    del y_prob_memmap
    gc.collect()

    return metrics, curves


# =============================================================================
# Output
# =============================================================================

def save_row_csv(
    path: Path,
    row: Mapping[str, object],
) -> None:
    """Append one result row to a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = not path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_curves(
    output_dir: Path,
    curves: Mapping[str, np.ndarray],
    experiment_name: str,
    prefix: str,
) -> None:
    """Save ROC and PR curves."""
    import matplotlib.pyplot as plt

    np.savez_compressed(
        output_dir / f"{prefix}_curves.npz",
        **curves,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        curves["fpr"],
        curves["tpr"],
        label="ROC curve",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(
        f"ROC — {experiment_name} ({prefix})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{prefix}_roc_curve.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.plot(
        curves["recall"],
        curves["precision"],
        label="Precision–Recall curve",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"Precision–Recall — {experiment_name} ({prefix})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"{prefix}_precision_recall_curve.png",
        dpi=180,
    )
    plt.close()


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    name: str,
    features: Sequence[Feature],
    datasets: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Train and evaluate one feature experiment."""
    print()
    print("=" * 100)
    print(f"Experiment:     {name}")
    print(f"Features:       {feature_combination_name(features)}")
    print("Loss:           focal")
    print(f"Focal alpha:    {args.focal_alpha}")
    print(f"Focal gamma:    {args.focal_gamma}")
    print(f"Device:         {device}")
    print("=" * 100)

    input_dim = get_input_dim(
        datasets[0],
        features=features,
        split=TRAIN,
        secondary_cache_size=args.secondary_cache_size,
    )

    print(f"Input dimension: {input_dim:,}")

    scaler, train_counts = fit_scaler(
        datasets=datasets,
        features=features,
        csv_chunk_size=args.csv_chunk_size,
        secondary_cache_size=args.secondary_cache_size,
    )

    validation_counts = count_split_rows(
        datasets,
        VALIDATION,
        args.csv_chunk_size,
    )

    test_counts = count_split_rows(
        datasets,
        TEST,
        args.csv_chunk_size,
    )

    print(
        f"Split counts: TRAIN={train_counts.total:,}, "
        f"VALIDATION={validation_counts.total:,}, "
        f"TEST={test_counts.total:,}"
    )

    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )
    print(model)

    training_criterion = BinaryFocalLoss(
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    experiment_dir = (
        args.output
        / name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work_dir = (
        experiment_dir
        / "_evaluation_work"
    )

    checkpoint_path = (
        experiment_dir
        / "best_model.pt"
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []

    for epoch in range(
        1,
        args.max_epochs + 1,
    ):
        print()
        print("-" * 100)
        print(
            f"Epoch {epoch}/{args.max_epochs}"
        )
        print("-" * 100)

        started = time.time()

        train_loss, train_samples_seen = train_one_epoch(
            model=model,
            datasets=datasets,
            features=features,
            scaler=scaler,
            optimizer=optimizer,
            criterion=training_criterion,
            device=device,
            batch_size=args.batch_size,
            csv_chunk_size=args.csv_chunk_size,
            secondary_cache_size=args.secondary_cache_size,
            log_every=args.log_every,
            epoch_seed=args.seed + epoch,
        )

        # IMPORTANT:
        # Always use unweighted BCE/log loss on VALIDATION, even though
        # TRAIN uses Focal Loss. This preserves comparability with LR/MLP.
        val_loss, val_samples_seen = evaluate_unweighted_log_loss(
            model=model,
            datasets=datasets,
            features=features,
            scaler=scaler,
            split=VALIDATION,
            device=device,
            batch_size=args.batch_size,
            csv_chunk_size=args.csv_chunk_size,
            secondary_cache_size=args.secondary_cache_size,
        )

        elapsed = time.time() - started

        epoch_record = {
            "epoch": epoch,
            "train_focal_loss": train_loss,
            "validation_log_loss": val_loss,
            "train_samples": train_samples_seen,
            "validation_samples": val_samples_seen,
            "epoch_seconds": elapsed,
        }

        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            experiment_dir
            / "training_history.csv",
            index=False,
        )

        print(
            f"epoch={epoch} "
            f"train_focal_loss={train_loss:.6f} "
            f"validation_log_loss={val_loss:.6f} "
            f"train_samples={train_samples_seen:,} "
            f"validation_samples={val_samples_seen:,} "
            f"time={elapsed / 60.0:.2f} min"
        )

        if (
            val_loss
            < best_val_loss - args.min_delta
        ):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim_1": args.hidden_dim_1,
                    "hidden_dim_2": args.hidden_dim_2,
                    "features": [
                        feature.value
                        for feature in features
                    ],
                    "loss": "focal",
                    "focal_alpha": args.focal_alpha,
                    "focal_gamma": args.focal_gamma,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                checkpoint_path,
            )

            print(
                f"  Saved best model -> "
                f"{checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"  No validation improvement "
                f"({epochs_without_improvement}/"
                f"{args.patience})"
            )

            if (
                epochs_without_improvement
                >= args.patience
            ):
                print("  Early stopping.")
                break

    if best_epoch == 0:
        raise RuntimeError(
            "No checkpoint was saved during training."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        f"Evaluating TEST with best checkpoint: "
        f"epoch={best_epoch}, "
        f"validation_log_loss={best_val_loss:.6f}"
    )

    test_metrics, test_curves = evaluate_exact_metrics(
        model=model,
        datasets=datasets,
        features=features,
        scaler=scaler,
        split=TEST,
        device=device,
        batch_size=args.batch_size,
        csv_chunk_size=args.csv_chunk_size,
        secondary_cache_size=args.secondary_cache_size,
        threshold=args.threshold,
        work_dir=work_dir,
    )

    result = {
        "experiment": name,
        "features": feature_combination_name(features),
        "model": "mlp",
        "loss": "focal",
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        **test_metrics,
        "train_samples": train_counts.total,
        "train_negative": train_counts.negative,
        "train_positive": train_counts.positive,
        "validation_samples": validation_counts.total,
        "test_samples": test_counts.total,
        "threshold": args.threshold,
        "input_dim": input_dim,
        "hidden_dim_1": args.hidden_dim_1,
        "hidden_dim_2": args.hidden_dim_2,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
    }

    save_row_csv(
        args.output
        / "mlp_focal_baseline_results.csv",
        result,
    )

    with (
        experiment_dir
        / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    save_curves(
        experiment_dir,
        test_curves,
        name,
        "test",
    )

    if (
        not args.keep_evaluation_predictions
        and work_dir.exists()
    ):
        shutil.rmtree(work_dir)

    print()
    print("Test metrics")
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "GPU MLP Focal Loss baseline directly comparable "
            "with logistic_baselines.py and mlp_baseline.py."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=list(EXPERIMENTS),
        help=(
            "Experiments to run. Use 'all' to run all seven "
            "feature combinations."
        ),
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--hidden-dim-1",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--hidden-dim-2",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help=(
            "Weight assigned to positive examples in Focal Loss. "
            "Negative examples receive 1-alpha."
        ),
    )

    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter used by Focal Loss.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--device",
        default="cuda:3",
    )

    parser.add_argument(
        "--csv-chunk-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--secondary-cache-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--keep-evaluation-predictions",
        action="store_true",
        help="Keep TEST memmap prediction files.",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.max_epochs <= 0:
        parser.error(
            "--max-epochs must be greater than zero."
        )

    if args.patience <= 0:
        parser.error(
            "--patience must be greater than zero."
        )

    if args.min_delta < 0:
        parser.error(
            "--min-delta cannot be negative."
        )

    if not 0.0 < args.threshold < 1.0:
        parser.error(
            "--threshold must be between 0 and 1."
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be greater than zero."
        )

    if args.csv_chunk_size <= 0:
        parser.error(
            "--csv-chunk-size must be greater than zero."
        )

    if args.secondary_cache_size <= 0:
        parser.error(
            "--secondary-cache-size must be greater than zero."
        )

    if not 0.0 <= args.focal_alpha <= 1.0:
        parser.error(
            "--focal-alpha must be between 0 and 1."
        )

    if args.focal_gamma < 0.0:
        parser.error(
            "--focal-gamma must be non-negative."
        )

    set_seed(args.seed)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"CUDA device {args.device!r} requested, "
            "but CUDA is not available."
        )

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

        print(
            "CUDA device: "
            f"{torch.cuda.get_device_name(device)}"
        )

    datasets = discover_datasets(args.root)

    if not datasets:
        raise RuntimeError(
            f"No datasets found below {args.root}"
        )

    print_datasets(datasets)

    selected_experiments = parse_experiments(
        args.experiments
    )

    args.output = (
        args.output
        .expanduser()
        .resolve()
    )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for experiment_name in selected_experiments:
        run_experiment(
            name=experiment_name,
            features=EXPERIMENTS[experiment_name],
            datasets=datasets,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()