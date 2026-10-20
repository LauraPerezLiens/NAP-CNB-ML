#!/usr/bin/env python3
"""
Evaluate all chunk-ladder MLP checkpoints on ONE COMMON TEST SET.
The common TEST set is defined as:
    1. Discover all physical Primary chunks.
    2. Reproduce the deterministic chunk selection used by mlp_baseline.py:
           seed = 42
           salt = 111111
    3. Select the first --test-chunks physical chunks from that permutation.
    4. From those physical chunks, evaluate ONLY rows with split_id == TEST.
    5. Use exactly the same TEST rows for every checkpoint.
This allows a fair comparison of models trained with different numbers of
physical chunks (10, 20, 30, ..., 500).
The script does NOT import mlp_baseline.py.
It only depends on dataset.py for dataset/embedding access and preprocessing.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from dataset import (
    TEST,
    ModalityStandardizer,
    discover_datasets,
    input_dimension,
    list_chunk_ids,
    load_chunk_metadata,
    load_rows,
    transform_input,
)
# =============================================================================
# Model
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
# =============================================================================
# Physical chunk jobs
# =============================================================================
@dataclass(frozen=True)
class ChunkJob:
    dataset_index: int
    chunk_id: int
def build_chunk_jobs(
    datasets: Sequence[Mapping[str, object]],
) -> list[ChunkJob]:
    jobs = []
    for dataset_index, dataset in enumerate(datasets):
        for chunk_id in list_chunk_ids(dataset):
            jobs.append(
                ChunkJob(
                    dataset_index=dataset_index,
                    chunk_id=int(chunk_id),
                )
            )
    return jobs
def select_fixed_jobs(
    all_jobs: Sequence[ChunkJob],
    seed: int,
    max_jobs: int,
    salt: int = 111111,
) -> list[ChunkJob]:
    """
    Exact chunk-selection logic used by mlp_baseline.py.
    The resulting physical chunks depend only on:
        all_jobs
        seed
        max_jobs
        salt
    """
    if max_jobs <= 0:
        raise ValueError("max_jobs must be > 0")
    if max_jobs > len(all_jobs):
        raise ValueError(
            f"Requested {max_jobs} physical chunks, "
            f"but only {len(all_jobs)} exist."
        )
    order = np.arange(len(all_jobs), dtype=np.int64)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, salt])
    )
    rng.shuffle(order)
    order = order[:max_jobs]
    return [all_jobs[int(i)] for i in order]
# =============================================================================
# Streaming metrics
# =============================================================================
class StreamingBinaryMetrics:
    """
    Streaming binary classification metrics.
    Threshold metrics use probability >= 0.5.
    ROC-AUC is estimated from a fixed probability histogram, avoiding the need
    to keep millions of predictions in RAM.
    """
    def __init__(self, auc_bins: int = 4096) -> None:
        if auc_bins < 2:
            raise ValueError("auc_bins must be >= 2")
        self.auc_bins = int(auc_bins)
        self.loss_sum = 0.0
        self.n = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.pos_hist = np.zeros(self.auc_bins, dtype=np.float64)
        self.neg_hist = np.zeros(self.auc_bins, dtype=np.float64)
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_sum: float,
    ) -> None:
        probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
        y = targets.detach().float().cpu().numpy()
        pred = probs >= 0.5
        positive = y >= 0.5
        negative = ~positive
        self.tp += int(np.count_nonzero(pred & positive))
        self.tn += int(np.count_nonzero((~pred) & negative))
        self.fp += int(np.count_nonzero(pred & negative))
        self.fn += int(np.count_nonzero((~pred) & positive))
        self.loss_sum += float(loss_sum)
        self.n += len(y)
        bins = np.floor(probs * self.auc_bins).astype(np.int64)
        bins = np.clip(bins, 0, self.auc_bins - 1)
        if np.any(positive):
            self.pos_hist += np.bincount(
                bins[positive],
                minlength=self.auc_bins,
            )
        if np.any(negative):
            self.neg_hist += np.bincount(
                bins[negative],
                minlength=self.auc_bins,
            )
    @staticmethod
    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")
    def roc_auc(self) -> float:
        n_pos = float(self.pos_hist.sum())
        n_neg = float(self.neg_hist.sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        neg_below = np.cumsum(self.neg_hist) - self.neg_hist
        concordant = np.sum(
            self.pos_hist *
            (neg_below + 0.5 * self.neg_hist)
        )
        return float(concordant / (n_pos * n_neg))
    def compute(self) -> dict[str, float]:
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn
        accuracy = self._safe(tp + tn, self.n)
        recall = self._safe(tp, tp + fn)
        specificity = self._safe(tn, tn + fp)
        balanced_accuracy = (
            (recall + specificity) / 2.0
            if not math.isnan(recall) and not math.isnan(specificity)
            else float("nan")
        )
        precision = self._safe(tp, tp + fp)
        f1 = self._safe(
            2 * precision * recall,
            precision + recall,
        )
        beta2 = 4.0
        f2 = self._safe(
            (1.0 + beta2) * precision * recall,
            beta2 * precision + recall,
        )
        mcc_den = math.sqrt(
            (tp + fp) *
            (tp + fn) *
            (tn + fp) *
            (tn + fn)
        )
        mcc = (
            (tp * tn - fp * fn) / mcc_den
            if mcc_den
            else float("nan")
        )
        return {
            "loss": self._safe(self.loss_sum, self.n),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "f2": f2,
            "mcc": float(mcc),
            "roc_auc": self.roc_auc(),
            "n": int(self.n),
            "negative": int(tn + fp),
            "positive": int(tp + fn),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }
# =============================================================================
# TEST iterator
# =============================================================================
def iter_test_batches(
    datasets,
    jobs,
    feature_mode: str,
    batch_size: int,
    secondary_cache_size: int,
):
    secondary_stores = {}
    for job_index, job in enumerate(jobs, start=1):
        dataset = datasets[job.dataset_index]
        meta = load_chunk_metadata(
            dataset,
            job.chunk_id,
            mmap=True,
        )
        rows = np.flatnonzero(
            meta["split_id"] == TEST
        ).astype(np.int64, copy=False)
        for start in range(0, len(rows), batch_size):
            selected_rows = rows[start:start + batch_size]
            gids = np.asarray(
                meta["group_id"][selected_rows],
                dtype=np.int64,
            )
            X = load_rows(
                dataset=dataset,
                chunk_id=job.chunk_id,
                row_indices=selected_rows,
                group_ids=gids,
                feature_mode=feature_mode,
                secondary_stores=secondary_stores,
                secondary_cache_size=secondary_cache_size,
            )
            y = np.asarray(
                meta["y"][selected_rows],
                dtype=np.float32,
            )
            yield {
                "X": X,
                "y": y,
                "job_index": job_index,
                "n_jobs": len(jobs),
                "dataset_name": str(dataset["name"]),
                "chunk_id": job.chunk_id,
            }
# =============================================================================
# Checkpoint / scaler loading
# =============================================================================
def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    features = checkpoint.get("features")
    if features not in ("primary", "primary_secondary"):
        raise RuntimeError(
            f"{checkpoint_path}: unsupported features={features!r}"
        )
    expected_dim = input_dimension(features)
    checkpoint_dim = int(checkpoint.get("input_dim", -1))
    if checkpoint_dim != expected_dim:
        raise RuntimeError(
            f"{checkpoint_path}: input_dim={checkpoint_dim}, "
            f"expected {expected_dim} for {features}."
        )
    dropout = float(checkpoint.get("dropout", 0.30))
    model = MLP(
        input_dim=checkpoint_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model, checkpoint
def load_standardizer(
    model_dir: Path,
    features: str,
) -> Optional[ModalityStandardizer]:
    if features == "primary":
        return None
    scaler_path = (
        model_dir /
        "standardizer_primary_secondary_2chunks.npz"
    )
    if not scaler_path.exists():
        raise FileNotFoundError(
            "\nStandardizer not found:\n"
            f"  {scaler_path}\n\n"
            "The model must be evaluated with the same preprocessing "
            "used during training."
        )
    standardizer = ModalityStandardizer.load(
        scaler_path
    )
    if len(standardizer.sampled_chunk_ids) != 2:
        raise RuntimeError(
            f"{scaler_path}: expected scaler fitted from exactly "
            f"2 chunks, found {len(standardizer.sampled_chunk_ids)}."
        )
    return standardizer
# =============================================================================
# Evaluation
# =============================================================================
@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    standardizer: Optional[ModalityStandardizer],
    datasets,
    test_jobs,
    features: str,
    device: torch.device,
    batch_size: int,
    secondary_cache_size: int,
    auc_bins: int,
    amp: bool,
    log_every_chunks: int,
):
    metrics = StreamingBinaryMetrics(
        auc_bins=auc_bins
    )
    use_amp = amp and device.type == "cuda"
    current_job = 0
    t0 = time.time()
    for batch in iter_test_batches(
        datasets=datasets,
        jobs=test_jobs,
        feature_mode=features,
        batch_size=batch_size,
        secondary_cache_size=secondary_cache_size,
    ):
        X = transform_input(
            batch["X"],
            feature_mode=features,
            standardizer=standardizer,
        )
        y = batch["y"]
        X_t = torch.from_numpy(
            np.asarray(X, dtype=np.float32)
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        y_t = torch.from_numpy(
            np.asarray(y, dtype=np.float32)
        ).to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(X_t)
            loss_sum = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y_t,
                    reduction="sum",
                )
            )
        metrics.update(
            logits=logits,
            targets=y_t,
            loss_sum=float(loss_sum.item()),
        )
        job_index = batch["job_index"]
        if job_index != current_job:
            current_job = job_index
            if (
                current_job == 1
                or current_job % log_every_chunks == 0
                or current_job == len(test_jobs)
            ):
                elapsed = time.time() - t0
                print(
                    f"  TEST chunk "
                    f"{current_job:>4}/{len(test_jobs)} | "
                    f"dataset={batch['dataset_name']} | "
                    f"id={batch['chunk_id']:06d} | "
                    f"n={metrics.n:,} | "
                    f"time={elapsed / 60:.1f} min",
                    flush=True,
                )
        del X, y, X_t, y_t, logits, loss_sum
    return metrics.compute()
# =============================================================================
# Reporting
# =============================================================================
def format_metrics(m: Mapping[str, float]) -> str:
    return (
        f"loss={m['loss']:.6f} | "
        f"acc={m['accuracy']:.4f} | "
        f"bal_acc={m['balanced_accuracy']:.4f} | "
        f"precision={m['precision']:.4f} | "
        f"recall={m['recall']:.4f} | "
        f"specificity={m['specificity']:.4f} | "
        f"F1={m['f1']:.4f} | "
        f"F2={m['f2']:.4f} | "
        f"MCC={m['mcc']:.4f} | "
        f"ROC-AUC~={m['roc_auc']:.4f} | "
        f"n={int(m['n']):,}"
    )
def save_results_csv(
    results: list[dict],
    path: Path,
) -> None:
    if not results:
        return
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fieldnames = list(results[0].keys())
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)
def save_individual_result(result: dict, model_dir: Path, test_chunks: int) -> tuple[Path, Path]:
    """Save one completed model evaluation immediately inside chunks_N/."""
    csv_path = model_dir / f"test_{test_chunks}chunks.csv"
    json_path = model_dir / f"test_{test_chunks}chunks.json"
    save_results_csv([result], csv_path)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    return csv_path, json_path

def load_individual_result(model_dir: Path, test_chunks: int) -> Optional[dict]:
    """Load a completed evaluation so interrupted runs can resume."""
    json_path = model_dir / f"test_{test_chunks}chunks.json"
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    expected_model_chunks = int(model_dir.name.replace("chunks_", ""))
    if int(result.get("model_chunks", -1)) != expected_model_chunks:
        raise RuntimeError(f"Unexpected model_chunks in {json_path}")
    if int(result.get("test_chunks", -1)) != int(test_chunks):
        raise RuntimeError(f"Unexpected test_chunks in {json_path}")
    return result

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chunk-ladder MLP models on one common "
            "fixed natural-distribution TEST set."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/nap/lperez_nn/data/data_input_nn"
        ),
    )
    parser.add_argument(
        "--ladder-dir",
        type=Path,
        default=Path(
            "/data/nap/lperez_nn/data/NN/"
            "Multilayer_Percepton/chunk_ladder"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=int,
        default=[
            10,
            20,
            30,
            50,
            75,
            100,
            150,
            200,
            300,
            500,
        ],
        help=(
            "Chunk-ladder models to evaluate. "
            "Each model is expected under chunks_N/."
        ),
    )
    parser.add_argument(
        "--test-chunks",
        type=int,
        default=500,
        help=(
            "Number of physical chunks defining the COMMON TEST set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--salt",
        type=int,
        default=111111,
        help=(
            "Must match the fixed physical-chunk selection "
            "used during training."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--secondary-cache-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--auc-bins",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()
# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    if args.test_chunks <= 0:
        raise ValueError("--test-chunks must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print("=" * 100)
    print("COMMON TEST EVALUATION - CHUNK LADDER")
    print("=" * 100)
    print(f"Root          : {args.root}")
    print(f"Ladder dir    : {args.ladder_dir}")
    print(f"Models        : {args.models}")
    print(f"Common TEST   : {args.test_chunks} physical chunks")
    print(f"Seed          : {args.seed}")
    print(f"Salt          : {args.salt}")
    print(f"Batch size    : {args.batch_size:,}")
    print(f"Device        : {device}")
    print(f"AMP           : {args.amp}")
    print()
    # -------------------------------------------------------------------------
    # Discover datasets
    # -------------------------------------------------------------------------
    datasets = discover_datasets(args.root)
    all_jobs = build_chunk_jobs(datasets)
    print(
        f"Discovered {len(datasets)} datasets "
        f"with {len(all_jobs):,} total physical chunks."
    )
    # -------------------------------------------------------------------------
    # COMMON TEST physical chunks
    # -------------------------------------------------------------------------
    test_jobs = select_fixed_jobs(
        all_jobs=all_jobs,
        seed=args.seed,
        max_jobs=args.test_chunks,
        salt=args.salt,
    )
    print()
    print(
        f"Selected {len(test_jobs):,} COMMON physical chunks."
    )
    # Count TEST rows once.
    n_test = 0
    n_positive = 0
    n_negative = 0
    print("Counting TEST rows...")
    for job in test_jobs:
        dataset = datasets[job.dataset_index]
        meta = load_chunk_metadata(
            dataset,
            job.chunk_id,
            mmap=True,
        )
        rows = np.flatnonzero(
            meta["split_id"] == TEST
        )
        y = np.asarray(
            meta["y"][rows],
            dtype=np.uint8,
        )
        positives = int(
            np.count_nonzero(y == 1)
        )
        n_test += len(y)
        n_positive += positives
        n_negative += len(y) - positives
    print(
        f"COMMON TEST rows : {n_test:,}"
    )
    print(
        f"  negative       : {n_negative:,} "
        f"({n_negative / n_test:.2%})"
    )
    print(
        f"  positive       : {n_positive:,} "
        f"({n_positive / n_test:.2%})"
    )
    print()
    # -------------------------------------------------------------------------
    # Evaluate models
    # -------------------------------------------------------------------------
    results = []
    for model_chunks in args.models:
        model_dir = (
            args.ladder_dir /
            f"chunks_{model_chunks}"
        )
        previous_result = load_individual_result(
            model_dir=model_dir,
            test_chunks=args.test_chunks,
        )
        if previous_result is not None:
            print()
            print("#" * 100)
            print(f"MODEL TRAINED WITH {model_chunks} PHYSICAL CHUNKS")
            print("#" * 100)
            print(
                f"SKIP: TEST result already exists: "
                f"{model_dir / f'test_{args.test_chunks}chunks.json'}"
            )
            print("COMMON TEST: " + format_metrics(previous_result))
            results.append(previous_result)
            continue
        checkpoint_path = (
            model_dir /
            "best_mlp_primary_secondary.pt"
        )
        print()
        print("#" * 100)
        print(
            f"MODEL TRAINED WITH {model_chunks} PHYSICAL CHUNKS"
        )
        print("#" * 100)
        if not checkpoint_path.exists():
            print(
                f"SKIP: checkpoint not found:\n"
                f"  {checkpoint_path}"
            )
            continue
        # ---------------------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------------------
        model, checkpoint = load_checkpoint_model(
            checkpoint_path,
            device,
        )
        features = checkpoint["features"]
        checkpoint_args = checkpoint.get(
            "args",
            {},
        )
        trained_max_chunks = checkpoint_args.get(
            "max_chunks"
        )
        print(f"Checkpoint     : {checkpoint_path}")
        print(f"Features       : {features}")
        print(
            f"Best epoch     : "
            f"{checkpoint.get('epoch', 'UNKNOWN')}"
        )
        print(
            f"Best val loss  : "
            f"{checkpoint.get('best_val_loss', float('nan'))}"
        )
        print(
            f"Checkpoint "
            f"max_chunks     : {trained_max_chunks}"
        )
        if (
            trained_max_chunks is not None
            and int(trained_max_chunks) != model_chunks
        ):
            raise RuntimeError(
                f"Directory says chunks_{model_chunks}, "
                f"but checkpoint args['max_chunks']="
                f"{trained_max_chunks}."
            )
        # ---------------------------------------------------------------------
        # Load scaler belonging to this training run
        # ---------------------------------------------------------------------
        standardizer = load_standardizer(
            model_dir=model_dir,
            features=features,
        )
        if standardizer is not None:
            print(
                "Scaler chunks   : "
            + ", ".join(
                    f"{name}:{int(cid):06d}"
                    for name, cid in zip(
                        standardizer.sampled_dataset_names,
                        standardizer.sampled_chunk_ids,
                    )
                )
            )
        # ---------------------------------------------------------------------
        # Evaluate
        # ---------------------------------------------------------------------
        t0 = time.time()
        metrics = evaluate_model(
            model=model,
            standardizer=standardizer,
            datasets=datasets,
            test_jobs=test_jobs,
            features=features,
            device=device,
            batch_size=args.batch_size,
            secondary_cache_size=args.secondary_cache_size,
            auc_bins=args.auc_bins,
            amp=args.amp,
            log_every_chunks=args.log_every_chunks,
        )
        elapsed = time.time() - t0
        print()
        print(
            "COMMON TEST: "
            + format_metrics(metrics)
        )
        print(
            f"Evaluation time: "
            f"{elapsed / 60:.2f} min"
        )
        result = {
            "model_chunks": model_chunks,
            "test_chunks": args.test_chunks,
            "best_epoch": checkpoint.get(
                "epoch"
            ),
            "best_validation_loss": checkpoint.get(
                "best_val_loss"
            ),
            **metrics,
            "evaluation_seconds": elapsed,
            "checkpoint": str(checkpoint_path),
        }
        results.append(result)
        individual_csv, individual_json = save_individual_result(
            result=result,
            model_dir=model_dir,
            test_chunks=args.test_chunks,
        )
        print(f"Saved result   : {individual_csv}")
        print(f"                 {individual_json}")
        del model, checkpoint, standardizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    if args.output is None:
        output_path = (
            args.ladder_dir /
            f"common_test_{args.test_chunks}chunks.csv"
        )
    else:
        output_path = args.output
    save_results_csv(
        results,
        output_path,
    )
    json_path = output_path.with_suffix(
        ".json"
    )
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            allow_nan=True,
        )
    # -------------------------------------------------------------------------
    # Final table
    # -------------------------------------------------------------------------
    print()
    print("=" * 160)
    print("FINAL COMPARISON - EXACT SAME TEST SET")
    print("=" * 160)
    header = (
        f"{'chunks':>7} "
        f"{'epoch':>6} "
        f"{'loss':>9} "
        f"{'acc':>8} "
        f"{'bal_acc':>8} "
        f"{'prec':>8} "
        f"{'recall':>8} "
        f"{'spec':>8} "
        f"{'F1':>8} "
        f"{'F2':>8} "
        f"{'MCC':>8} "
        f"{'AUC':>8} "
        f"{'N':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        best_epoch = r.get("best_epoch")
        epoch_text = str(best_epoch) if best_epoch is not None else "N/A"
        print(
            f"{r['model_chunks']:>7} "
            f"{epoch_text:>6} "
            f"{r['loss']:>9.5f} "
            f"{r['accuracy']:>8.4f} "
            f"{r['balanced_accuracy']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['specificity']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['f2']:>8.4f} "
            f"{r['mcc']:>8.4f} "
            f"{r['roc_auc']:>8.4f} "
            f"{int(r['n']):>12,}"
        )
    print("=" * 160)
    print()
    print(f"CSV  : {output_path}")
    print(f"JSON : {json_path}")
if __name__ == "__main__":
    main()