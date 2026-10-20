#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed MLP training for NAP-CNB Primary / Primary+Secondary embeddings.
Changes relative to the previous baseline:
- TRAIN and TEST only; TEST loss selects BEST. All epochs run.
- No CPU/RAM/GPU resource-monitoring output.
- TRAIN is still exact 50/50 within each physical chunk.
- TEST uses global 50/50 sampling across selected physical chunks.
- All TEST positives are retained.
- TRAIN EVAL and TEST EVAL use fixed global 50/50 samples without replacement.
- All three modes report exact ROC-AUC from logits.
- TRAIN, VALIDATION and TEST use the same selected physical chunks; only split_id differs.
- --seed + --max-chunks define a fixed physical-chunk subset reused across epochs.
- Training is processed in deterministic blocks of physical chunks.
- TRAIN ONLINE accumulates during training; TRAIN EVAL and TEST EVAL run at epoch end.
- A recovery checkpoint is saved after every block. If the machine/session dies,
--resume can continue from the next unprocessed block of the same epoch.
- A completed-epoch checkpoint is also saved.
- Primary+Secondary uses the same fixed 2-TRAIN-chunk standardizer logic.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import (
    TRAIN,
    VALIDATION,
    TEST,
    CyclingClassSampler,
    ModalityStandardizer,
    discover_datasets,
    fit_primary_secondary_standardizer,
    input_dimension,
    list_chunk_ids,
    load_chunk_metadata,
    load_rows,
    transform_input,
    validate_feature_mode,
)
# -----------------------------------------------------------------------------
# Reproducibility / distributed setup
# -----------------------------------------------------------------------------
def seed_everything(seed: int, rank: int = 0) -> None:
    s = int(seed) + int(rank)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
def distributed_setup() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU DDP requires CUDA.")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Explicit device_id avoids NCCL having to guess the rank -> GPU mapping.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda", 0)
        else:
            device = torch.device("cpu")
    return distributed, rank, local_rank, world_size, device
def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()
def is_main(rank: int) -> bool:
    return rank == 0
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.30) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)
# -----------------------------------------------------------------------------
# Chunk jobs
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkJob:
    dataset_index: int
    chunk_id: int
def build_chunk_jobs(datasets: Sequence[Mapping[str, object]]) -> list[ChunkJob]:
    jobs: list[ChunkJob] = []
    for dataset_index, dataset in enumerate(datasets):
        for chunk_id in list_chunk_ids(dataset):
            jobs.append(ChunkJob(dataset_index, int(chunk_id)))
    return jobs
def select_fixed_jobs(
    all_jobs: Sequence[ChunkJob],
    seed: int,
    max_jobs: Optional[int],
    salt: int,
) -> list[ChunkJob]:
    """
    Select a FIXED subset of physical chunks.
    The selected chunks depend only on:
      \\- all_jobs
      \\- seed
      \\- max_jobs
      \\- salt
    Therefore, the same seed always selects the same physical chunks,
    independently of the epoch.
    """
    order = np.arange(len(all_jobs), dtype=np.int64)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, salt])
    )
    rng.shuffle(order)
    if max_jobs is not None:
        order = order[:max_jobs]
    return [all_jobs[int(i)] for i in order]
def ordered_epoch_jobs(
    fixed_jobs,
    seed,
    epoch,
):
    return list(fixed_jobs)
def shard_block(
    block_jobs: Sequence[ChunkJob],
    rank: int,
    world_size: int,
) -> list[ChunkJob]:
    """Round-robin one GLOBAL training block across DDP ranks."""
    return list(block_jobs[rank::world_size])
def fixed_test_jobs(
    all_jobs: Sequence[ChunkJob],
    max_jobs: Optional[int],
) -> list[ChunkJob]:
    """Fixed, unshuffled physical-chunk set used for TEST evaluation."""
    jobs = list(all_jobs)
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    return jobs
# -----------------------------------------------------------------------------
# Standardizer
# -----------------------------------------------------------------------------
def prepare_standardizer(
    args: argparse.Namespace,
    datasets: Sequence[Mapping[str, object]],
    distributed: bool,
    rank: int,
) -> Optional[ModalityStandardizer]:
    if args.features == "primary":
        return None
    scaler_path = (
    Path(args.output_dir).resolve().parent
    / "standardizer_primary_secondary_2chunks.npz"
    )
    if is_main(rank):
        if scaler_path.exists() and not args.refit_scaler:
            existing = ModalityStandardizer.load(scaler_path)
            if len(existing.sampled_chunk_ids) != 2:
                raise RuntimeError(
                    f"Existing scaler was not fitted from exactly 2 chunks: {scaler_path}"
                )
            print(f"[SCALER] Reusing {scaler_path}", flush=True)
        else:
            print(
                "[SCALER] Fitting once from exactly 2 physical TRAIN chunks "
                f"(seed={args.seed}).",
                flush=True,
            )
            standardizer = fit_primary_secondary_standardizer(
                datasets=datasets,
                n_train_chunks=2,
                seed=args.seed,
                secondary_cache_size=args.secondary_cache_size,
                stats_batch_size=args.stats_batch_size,
            )
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            standardizer.save(scaler_path)
            print(f"[SCALER] Saved: {scaler_path}", flush=True)
    barrier(distributed)
    standardizer = ModalityStandardizer.load(scaler_path)
    if len(standardizer.sampled_chunk_ids) != 2:
        raise RuntimeError("Scaler must contain statistics from exactly 2 chunks.")
    return standardizer
# -----------------------------------------------------------------------------
# Streaming metrics
# -----------------------------------------------------------------------------
class StreamingBinaryMetrics:
    """Streaming threshold metrics with exact ROC-AUC from stored logits."""
    def __init__(self, auc_bins: int = 4096) -> None:
        self.auc_bins = int(auc_bins)
        self.loss_sum = 0.0
        self.n = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.y_true: list[np.ndarray] = []
        self.logits: list[np.ndarray] = []
    def merge(self, other: "StreamingBinaryMetrics") -> None:
        for key in ("loss_sum", "n", "tp", "tn", "fp", "fn"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        self.y_true.extend(other.y_true)
        self.logits.extend(other.logits)

    def state_dict(self) -> dict:
        return {key: getattr(self, key) for key in (
            "auc_bins", "loss_sum", "n", "tp", "tn", "fp", "fn",
            "y_true", "logits")}

    def load_state_dict(self, state: dict) -> None:
        if "y_true" not in state or "logits" not in state:
            raise RuntimeError(
                "Recovery checkpoint lacks labels/logits for exact ROC-AUC; "
                "resume from a completed-epoch checkpoint or start a new run."
            )
        for key in ("loss_sum", "n", "tp", "tn", "fp", "fn", "y_true", "logits"):
            setattr(self, key, state[key])

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_sum: float,
    ) -> None:
        scores = logits.detach().float().cpu().numpy().reshape(-1)
        y = targets.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1)
        pred = scores >= 0.0
        pos = y == 1
        neg = ~pos
        self.tp += int(np.count_nonzero(pred & pos))
        self.tn += int(np.count_nonzero((~pred) & neg))
        self.fp += int(np.count_nonzero(pred & neg))
        self.fn += int(np.count_nonzero((~pred) & pos))
        self.loss_sum += float(loss_sum)
        self.n += int(len(y))
        self.y_true.append(y.copy())
        self.logits.append(scores.copy())
    def synchronize(self, device: torch.device, distributed: bool) -> None:
        if not distributed:
            return
        scalars = torch.tensor(
            [
                self.loss_sum,
                float(self.n),
                float(self.tp),
                float(self.tn),
                float(self.fp),
                float(self.fn),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
        values = scalars.cpu().numpy()
        self.loss_sum = float(values[0])
        self.n = int(values[1])
        self.tp = int(values[2])
        self.tn = int(values[3])
        self.fp = int(values[4])
        self.fn = int(values[5])
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (self.y_true, self.logits))
        self.y_true = [batch for labels, _ in gathered for batch in labels]
        self.logits = [batch for _, scores in gathered for batch in scores]
    @staticmethod
    def _safe(num: float, den: float) -> float:
        return float(num / den) if den else float("nan")
    def roc_auc(self) -> float:
        if not self.y_true:
            return float("nan")
        y_true = np.concatenate(self.y_true)
        if np.unique(y_true).size < 2:
            return float("nan")
        logits = np.concatenate(self.logits)
        return float(roc_auc_score(y_true, logits))
    def compute(self) -> dict[str, float]:
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        precision = self._safe(tp, tp + fp)
        recall = self._safe(tp, tp + fn)
        specificity = self._safe(tn, tn + fp)
        accuracy = self._safe(tp + tn, tp + tn + fp + fn)
        balanced_accuracy = (
            (recall + specificity) / 2
            if not math.isnan(recall) and not math.isnan(specificity)
            else float("nan")
        )
        f1 = self._safe(2 * tp, 2 * tp + fp + fn)
        f2 = self._safe(5 * tp, 5 * tp + 4 * fn + fp)
        mcc_den = math.sqrt(
            max(0.0, float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        )
        mcc = self._safe(tp * tn - fp * fn, mcc_den)
        return {
            "loss": self._safe(self.loss_sum, self.n),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "f2": f2,
            "mcc": mcc,
            "roc_auc": self.roc_auc(),
            "n": float(self.n),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
# -----------------------------------------------------------------------------
# Batch loading
# -----------------------------------------------------------------------------
def _train_rows_by_class(
    dataset: Mapping[str, object],
    chunk_id: int,
) -> tuple[Mapping[str, np.ndarray], np.ndarray, np.ndarray]:
    meta = load_chunk_metadata(dataset, chunk_id, mmap=True)
    train_rows = np.flatnonzero(meta["split_id"] == TRAIN).astype(np.int64, copy=False)
    if len(train_rows) == 0:
        empty = np.empty(0, dtype=np.int64)
        return meta, empty, empty
    train_y = np.asarray(meta["y"][train_rows], dtype=np.uint8)
    neg_rows = train_rows[train_y == 0]
    pos_rows = train_rows[train_y == 1]
    return meta, neg_rows, pos_rows
def _find_train_donor(
    datasets: Sequence[Mapping[str, object]],
    all_jobs: Sequence[ChunkJob],
    missing_class: int,
    seed: int,
    epoch: int,
    dataset_index: int,
    chunk_id: int,
) -> tuple[ChunkJob, Mapping[str, np.ndarray], np.ndarray]:
    """Find a deterministic TRAIN-only donor for a monoclasse source chunk."""
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [seed, epoch, dataset_index, chunk_id, missing_class, 918273]
        )
    )
    start = int(rng.integers(0, len(all_jobs)))
    for offset in range(len(all_jobs)):
        candidate = all_jobs[(start + offset) % len(all_jobs)]
        if (
            candidate.dataset_index == dataset_index
            and candidate.chunk_id == chunk_id
        ):
            continue
        candidate_dataset = datasets[candidate.dataset_index]
        meta, neg_rows, pos_rows = _train_rows_by_class(
            candidate_dataset, candidate.chunk_id
        )
        rows = neg_rows if missing_class == 0 else pos_rows
        if len(rows) > 0:
            return candidate, meta, rows
    raise RuntimeError(
        f"No TRAIN donor containing class {missing_class} was found."
    )
def _load_chunk_rows(
    dataset: Mapping[str, object],
    chunk_id: int,
    meta: Mapping[str, np.ndarray],
    rows: np.ndarray,
    feature_mode: str,
    secondary_cache_size: int,
    secondary_stores: dict,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(rows)
    sorted_rows = rows[order]
    restore = np.argsort(order)
    gids = np.asarray(meta["group_id"][sorted_rows], dtype=np.int64)
    X_sorted = load_rows(
        dataset=dataset,
        chunk_id=chunk_id,
        row_indices=sorted_rows,
        group_ids=gids,
        feature_mode=feature_mode,
        secondary_stores=secondary_stores,
        secondary_cache_size=secondary_cache_size,
    )
    X = X_sorted[restore]
    y = np.asarray(meta["y"][rows], dtype=np.float32)
    return X, y
def iter_balanced_chunk_batches(
    dataset: Mapping[str, object],
    chunk_id: int,
    split: int,
    feature_mode: str,
    batch_size: int,
    seed: int,
    epoch: int,
    dataset_index: int,
    secondary_cache_size: int,
    secondary_stores: dict,
    datasets: Sequence[Mapping[str, object]],
    all_jobs: Sequence[ChunkJob],
):
    """Yield exact 50/50 TRAIN batches while covering every source TRAIN row."""
    if batch_size % 2 != 0:
        raise ValueError("batch_size must be even for exact 50/50 TRAIN batches.")
    meta, neg_rows, pos_rows = _train_rows_by_class(dataset, chunk_id)
    if len(neg_rows) == 0 and len(pos_rows) == 0:
        return
    half = batch_size // 2
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, epoch, dataset_index, chunk_id])
    )
    if len(neg_rows) > 0 and len(pos_rows) > 0:
        neg_sampler = CyclingClassSampler(neg_rows, rng)
        pos_sampler = CyclingClassSampler(pos_rows, rng)
        n_batches = int(np.ceil(max(len(neg_rows), len(pos_rows)) / half))
        for _ in range(n_batches):
            selected = np.concatenate(
                (neg_sampler.take(half), pos_sampler.take(half))
            )
            rng.shuffle(selected)
            X, y = _load_chunk_rows(
                dataset, chunk_id, meta, selected, feature_mode,
                secondary_cache_size, secondary_stores,
            )
            yield X, y
        return
    source_class = 0 if len(neg_rows) > 0 else 1
    missing_class = 1 - source_class
    source_rows = neg_rows if source_class == 0 else pos_rows
    donor_job, donor_meta, donor_rows = _find_train_donor(
        datasets=datasets,
        all_jobs=all_jobs,
        missing_class=missing_class,
        seed=seed,
        epoch=epoch,
        dataset_index=dataset_index,
        chunk_id=chunk_id,
    )
    donor_dataset = datasets[donor_job.dataset_index]
    source_sampler = CyclingClassSampler(source_rows, rng)
    donor_rng = np.random.default_rng(
        np.random.SeedSequence([
            seed, epoch, dataset_index, chunk_id,
            donor_job.dataset_index, donor_job.chunk_id, 445566,
        ])
    )
    donor_sampler = CyclingClassSampler(donor_rows, donor_rng)
    n_batches = int(np.ceil(len(source_rows) / half))
    for _ in range(n_batches):
        source_selected = source_sampler.take(half)
        donor_selected = donor_sampler.take(half)
        X_source, y_source = _load_chunk_rows(
            dataset, chunk_id, meta, source_selected, feature_mode,
            secondary_cache_size, secondary_stores,
        )
        X_donor, y_donor = _load_chunk_rows(
            donor_dataset, donor_job.chunk_id, donor_meta, donor_selected,
            feature_mode, secondary_cache_size, secondary_stores,
        )
        X = np.concatenate((X_source, X_donor), axis=0)
        y = np.concatenate((y_source, y_donor), axis=0)
        permutation = rng.permutation(len(y))
        yield X[permutation], y[permutation]
def iter_natural_split_batches(
    dataset: Mapping[str, object],
    chunk_id: int,
    split: int,
   feature_mode: str,
    batch_size: int,
    secondary_cache_size: int,
    secondary_stores: dict,
):
    meta = load_chunk_metadata(dataset, chunk_id, mmap=True)
    rows = np.flatnonzero(meta["split_id"] == split).astype(np.int64, copy=False)
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        gids = np.asarray(meta["group_id"][selected], dtype=np.int64)
        X = load_rows(
            dataset=dataset,
            chunk_id=chunk_id,
            row_indices=selected,
            group_ids=gids,
            feature_mode=feature_mode,
            secondary_stores=secondary_stores,
            secondary_cache_size=secondary_cache_size,
        )
        y = np.asarray(meta["y"][selected], dtype=np.float32)
        yield X, y

def build_global_balanced_eval_plan(
    datasets: Sequence[Mapping[str, object]],
    eval_jobs: Sequence[ChunkJob],
    seed: int,
    split: int,
) -> dict[ChunkJob, np.ndarray]:
    """
    Build a globally balanced evaluation sampling plan across selected chunks.

    - Keep ALL split positives.
    - Sample the same number of split negatives globally.
    - Sample without replacement.
    - Keep selected rows fixed across epochs for a given seed and split.
    - Return row indices grouped by physical chunk.

    All DDP ranks independently build the same deterministic plan.
    Only metadata are loaded; embeddings are not loaded here.
    """
    positive_rows = {}
    negative_rows = {}
    negative_counts = []

    total_positive = 0
    total_negative = 0

    for job in eval_jobs:
        dataset = datasets[job.dataset_index]

        meta = load_chunk_metadata(
            dataset,
            job.chunk_id,
            mmap=True,
        )

        test_rows = np.flatnonzero(
            meta["split_id"] == split
        ).astype(np.int64, copy=False)

        test_y = np.asarray(
            meta["y"][test_rows],
            dtype=np.uint8,
        )

        pos_rows = test_rows[test_y == 1]
        neg_rows = test_rows[test_y == 0]

        positive_rows[job] = pos_rows
        negative_rows[job] = neg_rows
        negative_counts.append(len(neg_rows))

        total_positive += len(pos_rows)
        total_negative += len(neg_rows)

    if total_positive == 0:
        raise RuntimeError(
            "Selected split chunks contain no positive examples."
        )

    if total_negative < total_positive:
        raise RuntimeError(
            "Cannot keep all split positives with exact 50/50 "
            "sampling without replacement: "
            f"positives={total_positive:,}, "
            f"negatives={total_negative:,}."
        )

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, split, 987654])
    )

    # Sample global negative positions without replacement.
    selected_negative_positions = np.sort(
        rng.choice(
            total_negative,
            size=total_positive,
            replace=False,
        )
    )

    negative_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(negative_counts, dtype=np.int64),
        )
    )

    plan = {}

    for index, job in enumerate(eval_jobs):
        start = negative_offsets[index]
        end = negative_offsets[index + 1]

        left = np.searchsorted(
            selected_negative_positions,
            start,
            side="left",
        )

        right = np.searchsorted(
            selected_negative_positions,
            end,
            side="left",
        )

        local_negative_positions = (
            selected_negative_positions[left:right] - start
        )

        selected_negatives = negative_rows[job][
            local_negative_positions
        ]

        selected_rows = np.concatenate(
            (
                positive_rows[job],
                selected_negatives,
            )
        )

        # Shuffle rows deterministically within each chunk.
        chunk_rng = np.random.default_rng(
            np.random.SeedSequence([
                seed,
                split,
                job.dataset_index,
                job.chunk_id,
                123456,
            ])
        )

        chunk_rng.shuffle(selected_rows)

        plan[job] = selected_rows

    return plan


def iter_selected_test_chunk_batches(
    dataset: Mapping[str, object],
    chunk_id: int,
    selected_rows: np.ndarray,
    feature_mode: str,
    batch_size: int,
    secondary_cache_size: int,
    secondary_stores: dict,
):
    """
    Load and yield previously selected TEST rows in batches.
    """
    if len(selected_rows) == 0:
        return

    meta = load_chunk_metadata(
        dataset,
        chunk_id,
        mmap=True,
    )

    for start in range(0, len(selected_rows), batch_size):
        rows = selected_rows[start:start + batch_size]

        X, y = _load_chunk_rows(
            dataset=dataset,
            chunk_id=chunk_id,
            meta=meta,
            rows=rows,
            feature_mode=feature_mode,
            secondary_cache_size=secondary_cache_size,
            secondary_stores=secondary_stores,
        )

        yield X, y

def to_device_batch(
    X: np.ndarray,
    y: np.ndarray,
    feature_mode: str,
    standardizer: Optional[ModalityStandardizer],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    X = transform_input(X, feature_mode, standardizer)
    X_t = torch.from_numpy(np.ascontiguousarray(X)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    return X_t, y_t
# -----------------------------------------------------------------------------
# Training / test
# -----------------------------------------------------------------------------
def train_block(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    standardizer: Optional[ModalityStandardizer],
    datasets: Sequence[Mapping[str, object]],
    all_jobs: Sequence[ChunkJob],
    block_jobs: Sequence[ChunkJob],
    args: argparse.Namespace,
    epoch: int,
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> StreamingBinaryMetrics:
    """Train one GLOBAL block and return mergeable metrics."""
    model.train()
    jobs = shard_block(block_jobs, rank, world_size)
    metrics = StreamingBinaryMetrics(args.auc_bins)
    secondary_stores: dict = {}
    # Some chunks generate more minibatches than others. DDP join prevents a
    # rank that finishes first from hanging while the other rank completes.
    join_context = (
        model.join(divide_by_initial_world_size=True)
        if distributed
        else nullcontext()
    )
    with join_context:
        for job in jobs:
            dataset = datasets[job.dataset_index]
            for X, y in iter_balanced_chunk_batches(
                dataset=dataset,
                chunk_id=job.chunk_id,
                split=TRAIN,
                feature_mode=args.features,
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
                dataset_index=job.dataset_index,
                secondary_cache_size=args.secondary_cache_size,
                secondary_stores=secondary_stores,
                datasets=datasets,
                all_jobs=all_jobs,
            ):
                X_t, y_t = to_device_batch(
                    X, y, args.features, standardizer, device
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(X_t)
                loss = criterion(logits, y_t)

                loss.backward()
                optimizer.step()
                metrics.update(
                    logits=logits,
                    targets=y_t,
                    loss_sum=float(loss.detach().item()) * len(y),
                )
                del X, y, X_t, y_t, logits, loss
    metrics.synchronize(device, distributed)
    return metrics
@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    criterion: nn.Module,
    standardizer: Optional[ModalityStandardizer],
    datasets: Sequence[Mapping[str, object]],
    eval_jobs: Sequence[ChunkJob],
    split: int,
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    sampling_plan: Optional[Mapping[ChunkJob, np.ndarray]] = None,
) -> dict[str, float]:

    """Evaluate fixed 50/50 samples with dropout off and per-sample LayerNorm."""
    was_training = model.training
    eval_model = model.module if isinstance(model, DDP) else model
    # Use identical running statistics on every rank during evaluation.
    if distributed:
        for buffer in eval_model.buffers():
            dist.broadcast(buffer, src=0)
    eval_model.eval()
    if sampling_plan is None:
        sampling_plan = build_global_balanced_eval_plan(
            datasets=datasets, eval_jobs=eval_jobs, seed=args.seed, split=split,
        )

    # Preserve the existing DDP distribution.
    jobs = list(eval_jobs[rank::world_size])

    metrics = StreamingBinaryMetrics(args.auc_bins)

    secondary_stores: dict = {}

    for job in jobs:

        dataset = datasets[job.dataset_index]

        batch_iterator = iter_selected_test_chunk_batches(
            dataset=dataset,
            chunk_id=job.chunk_id,
            selected_rows=sampling_plan[job],
            feature_mode=args.features,
            batch_size=args.eval_batch_size,
            secondary_cache_size=args.secondary_cache_size,
            secondary_stores=secondary_stores,
        )

        for X, y in batch_iterator:

            X_t, y_t = to_device_batch(
                X,
                y,
                args.features,
                standardizer,
                device,
            )

            logits = eval_model(X_t)

            loss_sum = nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_t,
                reduction="sum",
            )

            metrics.update(
                logits,
                y_t,
                float(loss_sum.item()),
            )

            del X, y, X_t, y_t, logits, loss_sum

    metrics.synchronize(device, distributed)

    model.train(was_training)

    return metrics.compute()
# -----------------------------------------------------------------------------
# Reporting / checkpointing
# -----------------------------------------------------------------------------
def format_metrics(prefix: str, m: Mapping[str, float]) -> str:
    return (
        f"{prefix}: loss={m['loss']:.6f} | acc={m['accuracy']:.4f} | "
        f"bal_acc={m['balanced_accuracy']:.4f} | precision={m['precision']:.4f} | "
        f"recall={m['recall']:.4f} | specificity={m['specificity']:.4f} | "
        f"F1={m['f1']:.4f} | F2={m['f2']:.4f} | MCC={m['mcc']:.4f} | "
        f"ROC-AUC={m['roc_auc']:.4f} | n={int(m['n']):,}"
    )
def append_history(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=True) + "\n")
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    args: argparse.Namespace,
    epoch_index: int,
    next_chunk_offset: int,
    epoch_total_chunks: int,
    completed_epochs: int,
    best_val_loss: float = float("inf"),
    best_epoch: int = 0,
    epochs_without_improvement: int = 0,
    train_epoch_metrics_state: Optional[dict] = None,
) -> None:
    """
    Save enough state to resume either:
      \\- inside the current epoch (next_chunk_offset < epoch_total_chunks), or
      \\- at the next epoch (completed_epochs == epoch_index + 1).
    """
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "checkpoint_version": 4,
            "normalization": "layernorm",
            "train_epoch_metrics_state": train_epoch_metrics_state,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "features": args.features,
            "input_dim": input_dimension(args.features),
            "dropout": args.dropout,
            "seed": args.seed,
            "epoch_index": int(epoch_index),
            "next_chunk_offset": int(next_chunk_offset),
            "epoch_total_chunks": int(epoch_total_chunks),
            "completed_epochs": int(completed_epochs),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_without_improvement": int(epochs_without_improvement),
            "args": vars(args),
        },
        path,
    )
# -----------------------------------------------------------------------------
# Arguments / main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/nap/lperez_nn/data/data_input_nn",
    )
    parser.add_argument(
        "--features",
        choices=("primary", "primary_secondary"),
        default="primary_secondary",
    )
    parser.add_argument("--epochs", type=int, default=100)
    # No early stopping: always complete --epochs.

    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Cosine Annealing with Warm Restarts
    parser.add_argument("--scheduler-t0", type=int, default=20)
    parser.add_argument("--scheduler-tmult", type=int, default=1)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--secondary-cache-size", type=int, default=2)
    parser.add_argument("--stats-batch-size", type=int, default=8192)
    parser.add_argument(
        "--auc-bins", type=int, default=4096,
        help="Deprecated compatibility option; exact ROC-AUC does not use bins.",
    )
    parser.add_argument("--refit-scaler", action="store_true")
    parser.add_argument(
        "--report-every-chunks",
        type=int,
        default=500,
        help=(
            "GLOBAL physical TRAIN chunks per training/report/checkpoint block. "
            "With 2 GPUs, 500 means about 250 chunks per rank."
        ),
    )
    parser.add_argument(
        "--monitor-test-chunks",
        type=int,
        default=0,
        help=(
            "Number of the FIXED selected physical chunks used for periodic TEST monitoring. "
            "Set 0 to disable periodic TEST monitoring."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "Optional GLOBAL physical-chunk limit. "
            "With a fixed --seed, increasing this value produces nested chunk subsets."
        ),
    )
    parser.add_argument("--output-dir", default="./mlp_results")
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume from recovery/last checkpoint. --epochs remains the TOTAL "
            "target number of epochs."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from a previous BEST checkpoint, but start "
            "a new training stage: new optimizer, epoch=1 and reset early stopping."
        ),
    )
    parser.add_argument(
        "--full-test-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional diagnostic only: run TEST every epoch. Keep disabled for model selection.",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    validate_feature_mode(args.features)
    if args.batch_size <= 0 or args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be a positive even number.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be > 0.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if args.report_every_chunks <= 0:
        raise ValueError("--report-every-chunks must be > 0.")
    if args.monitor_test_chunks < 0:
        raise ValueError("--monitor-test-chunks must be >= 0.")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("--max-chunks must be > 0.")
    distributed, rank, local_rank, world_size, device = distributed_setup()
    seed_everything(args.seed, rank)
    output_dir = Path(args.output_dir).resolve()
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)
    datasets = discover_datasets(args.root)
    all_jobs = build_chunk_jobs(datasets)
    fixed_train_jobs = select_fixed_jobs(
        all_jobs=all_jobs,
        seed=args.seed,
        max_jobs=args.max_chunks,
        salt=111111,
    )
    # TEST uses the same physical chunks but only TEST-labelled rows.
    fixed_test_jobs_selected = list(fixed_train_jobs)
    if is_main(rank):
        print("=" * 100)
        print(f"MLP LAYERNORM | TRAIN ONLINE + TRAIN EVAL + TEST EVAL | {args.epochs} epochs | BEST by TEST loss")
        print("=" * 100)
        print(f"Datasets                 : {len(datasets)}")
        print(f"Physical chunks          : {len(all_jobs):,}")
        print(f"Features                 : {args.features}")
        print(f"Input dimension          : {input_dimension(args.features)}")
        print(f"GPUs / DDP ranks         : {world_size}")
        print(f"Batch size / GPU         : {args.batch_size:,} (exact 50/50 TRAIN)")
        print(f"Report/checkpoint every  : {args.report_every_chunks:,} GLOBAL TRAIN chunks")
        print("Early stopping           : DISABLED")
        if args.monitor_test_chunks:
            print(f"Periodic TEST monitor    : up to {args.monitor_test_chunks:,} of the SAME selected chunks")
        else:
            print("Periodic TEST monitor    : disabled")
        print(f"Output                   : {output_dir}")
        print(f"Resume                   : {args.resume if args.resume else 'NONE'}")
        print(
            f"Init checkpoint          : "
            f"{args.init_checkpoint if args.init_checkpoint else 'NONE'}"
        )
        print()
        print(f"FIXED PHYSICAL CHUNKS (seed={args.seed})")
        print("TRAIN -> split_id == TRAIN | TEST -> split_id == TEST")
        print("All splits are read from these SAME physical chunks:")
        print(f"Selected {len(fixed_train_jobs):,} physical chunks (details omitted).")
        print()
    standardizer = prepare_standardizer(
        args=args,
        datasets=datasets,
        distributed=distributed,
        rank=rank,
    )
    barrier(distributed)
    model = MLP(
        input_dim=input_dimension(args.features),
        dropout=args.dropout,
    ).to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
        )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args.scheduler_t0,
    T_mult=args.scheduler_tmult,
    eta_min=args.scheduler_min_lr,
    )
    recovery_path = output_dir / f"recovery_mlp_{args.features}.pt"
    last_path = output_dir / f"last_mlp_{args.features}.pt"
    best_path = output_dir / f"best_mlp_{args.features}.pt"
    history_path = output_dir / f"history_mlp_{args.features}.jsonl"
    raw_model = model.module if isinstance(model, DDP) else model
    # Resume position.
    start_epoch = 0
    resume_offset = 0
    best_test_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    resumed_train_metrics_state = None
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError(
            "--resume and --init-checkpoint are mutually exclusive."
        )
    if args.init_checkpoint is not None:
        init_path = Path(args.init_checkpoint).resolve()
        if not init_path.exists():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {init_path}"
            )
        checkpoint = torch.load(
            init_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("features") != args.features:
            raise RuntimeError(
                f"Checkpoint features={checkpoint.get('features')!r}, "
                f"but --features={args.features!r}."
            )
        if int(checkpoint.get("input_dim", -1)) != input_dimension(args.features):
            raise RuntimeError(
                "Checkpoint input dimension does not match."
            )
        if not math.isclose(
            float(checkpoint.get("dropout", args.dropout)),
            args.dropout,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Checkpoint dropout does not match --dropout."
            )

        raw_model.load_state_dict(checkpoint["model_state_dict"])
        if is_main(rank):
            print(
                f"[INIT] Model weights loaded from: {init_path}",
                flush=True,
            )
            print(
                "[INIT] New training stage: optimizer, epoch and early stopping reset.",
                flush=True,
            )
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("features") != args.features:
            raise RuntimeError(
                f"Checkpoint features={checkpoint.get('features')!r}, "
                f"but --features={args.features!r}."
            )
        if int(checkpoint.get("input_dim", -1)) != input_dimension(args.features):
            raise RuntimeError("Checkpoint input dimension does not match.")
        if not math.isclose(
            float(checkpoint.get("dropout", args.dropout)),
            args.dropout,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Checkpoint dropout does not match --dropout.")
        if int(checkpoint.get("seed", args.seed)) != args.seed:
            raise RuntimeError(
                "Checkpoint seed differs from --seed. Exact intra-epoch resume "
                "requires the same seed."
            )
        raw_model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain scheduler state."
            )

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_test_loss = float(checkpoint.get("best_val_loss", float("inf")))
        resumed_train_metrics_state = checkpoint.get("train_epoch_metrics_state")
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        completed_epochs = int(checkpoint.get("completed_epochs", 0))
        epoch_index = int(checkpoint.get("epoch_index", completed_epochs))
        next_offset = int(checkpoint.get("next_chunk_offset", 0))
        epoch_total = int(checkpoint.get("epoch_total_chunks", 0))
        if next_offset > 0 and next_offset < epoch_total:
            if resumed_train_metrics_state is None:
                raise RuntimeError("Recovery checkpoint lacks accumulated TRAIN metrics; use a new output directory.")
            start_epoch = epoch_index
            resume_offset = next_offset
            if is_main(rank):
                print(
                    f"[RESUME] Epoch {start_epoch + 1}: continuing from "
                    f"GLOBAL TRAIN chunk {resume_offset:,}/{epoch_total:,}.",
                    flush=True,
                )
        else:
            start_epoch = completed_epochs
            resume_offset = 0
            if is_main(rank):
                print(
                    f"[RESUME] {completed_epochs} complete epoch(s); "
                    f"continuing with epoch {start_epoch + 1}.",
                    flush=True,
                )
        if args.epochs <= start_epoch:
            raise ValueError(
                f"--epochs={args.epochs} must be greater than completed "
                f"epochs={start_epoch}."
            )
    else:
        if is_main(rank) and history_path.exists():
            history_path.unlink()
    barrier(distributed)
    # Build both evaluation populations once and reuse the same rows every epoch.
    full_test_jobs = fixed_test_jobs_selected
    train_eval_plan = build_global_balanced_eval_plan(
        datasets, fixed_train_jobs, args.seed, TRAIN,
    )
    test_eval_plan = build_global_balanced_eval_plan(
        datasets, full_test_jobs, args.seed, TEST,
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        epoch_jobs = ordered_epoch_jobs(
            fixed_jobs=fixed_train_jobs,
            seed=args.seed,
            epoch=epoch,
        )
        total_chunks = len(epoch_jobs)
        offset = resume_offset if epoch == start_epoch else 0
        train_epoch_metrics = StreamingBinaryMetrics(args.auc_bins)
        if offset:
            train_epoch_metrics.load_state_dict(resumed_train_metrics_state)
        if offset > total_chunks:
            raise RuntimeError(
                f"Resume offset {offset} exceeds epoch size {total_chunks}."
            )
        if is_main(rank):
            print(
                f"\nEPOCH {epoch + 1}/{args.epochs} | "
                f"TRAIN chunks={total_chunks:,} | starting at {offset:,}",
                flush=True,
            )
        while offset < total_chunks:
            block_start = offset
            block_end = min(
                offset + args.report_every_chunks,
                total_chunks,
            )
            block_jobs = epoch_jobs[block_start:block_end]
            block_t0 = time.time()
            train_metrics = train_block(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                standardizer=standardizer,
                datasets=datasets,
                all_jobs=fixed_train_jobs,
                block_jobs=block_jobs,
                args=args,
                epoch=epoch,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
                device=device,
            )
            barrier(distributed)
            train_epoch_metrics.merge(train_metrics)
            offset = block_end
            if is_main(rank) and offset < total_chunks:
                # Keep recovery checkpoints without per-block terminal output.
                save_checkpoint(
                    path=recovery_path, model=model, optimizer=optimizer, scheduler=scheduler,
                    args=args, epoch_index=epoch,
                    next_chunk_offset=offset, epoch_total_chunks=total_chunks,
                    completed_epochs=epoch, best_val_loss=best_test_loss,
                    best_epoch=best_epoch,
                    train_epoch_metrics_state=train_epoch_metrics.state_dict(),
                )
            barrier(distributed)
        # Freeze weights and disable dropout for both end-of-epoch evaluations.
        train_eval_metrics = evaluate_split(
            model=model,
            criterion=criterion,
            standardizer=standardizer,
            datasets=datasets,
            eval_jobs=fixed_train_jobs,
            split=TRAIN,
            args=args,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            device=device,
            sampling_plan=train_eval_plan,
        )
        full_test_metrics = evaluate_split(
            model=model,
            criterion=criterion,
            standardizer=standardizer,
            datasets=datasets,
            eval_jobs=full_test_jobs,
            split=TEST,
            args=args,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            device=device,
            sampling_plan=test_eval_plan,
        )
        barrier(distributed)
        current_test_loss = float(full_test_metrics["loss"])
        if is_main(rank):

            n_positive = (
                full_test_metrics["tp"] + full_test_metrics["fn"]
            )

            n_negative = (
                full_test_metrics["tn"] + full_test_metrics["fp"]
            )

            print(
                f"[TEST BALANCE] "
                f"positive={int(n_positive):,} | "
                f"negative={int(n_negative):,} | "
                f"total={int(full_test_metrics['n']):,}",
                flush=True,
            )

            if n_positive != n_negative:
                raise RuntimeError(
                    "TEST is not balanced 50/50."
                )
        improved = (
            math.isfinite(current_test_loss)
            and current_test_loss < best_test_loss
        )

        if improved:
            best_test_loss = current_test_loss
            best_epoch = epoch + 1

        if best_epoch == 0:
            raise RuntimeError(
                "TEST loss is not finite; cannot select BEST checkpoint."
            )

        epoch_train_metrics = train_epoch_metrics.compute()

        # --------------------------------------------------
        # COSINE ANNEALING WARM RESTARTS
        # --------------------------------------------------

        # Learning rate used during the completed epoch
        current_lr = optimizer.param_groups[0]["lr"]

        # Update the scheduler for the next epoch
        scheduler.step(epoch + 1)

        # Learning rate that will be used in the next epoch
        next_lr = optimizer.param_groups[0]["lr"]

        if is_main(rank):
            print(
                f"[LR] Epoch {epoch + 1}/{args.epochs} | "
                f"current_lr={current_lr:.8f} | "
                f"next_lr={next_lr:.8f}",
                flush=True,
            )
        if is_main(rank):
            elapsed = time.time() - epoch_start
            print("\n" + "=" * 100)
            print(
                f"EPOCH {epoch + 1}/{args.epochs} COMPLETE | "
                f"{elapsed / 3600:.2f} h"
            )
            print(format_metrics("TRAIN ONLINE (50/50; exact AUC)", epoch_train_metrics))
            print(format_metrics("TRAIN EVAL (fixed 50/50; exact AUC)", train_eval_metrics))
            print(format_metrics("TEST EVAL (fixed 50/50; exact AUC)", full_test_metrics))
            print(f"BEST | epoch={best_epoch} | test_loss={best_test_loss:.6f}")
            print("=" * 100, flush=True)
            if improved:
                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                    epoch_index=epoch,
                    next_chunk_offset=total_chunks,
                    epoch_total_chunks=total_chunks,
                    completed_epochs=epoch + 1,
                    best_val_loss=best_test_loss,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                )
                print(
                    f"[BEST] Test loss improved -> {best_test_loss:.6f} "
                    f"(epoch {best_epoch}) | saved: {best_path}",
                    flush=True,
                )
            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch_index=epoch,
                next_chunk_offset=total_chunks,
                epoch_total_chunks=total_chunks,
                completed_epochs=epoch + 1,
                best_val_loss=best_test_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            # Keep recovery synchronized with the latest safe point too.
            save_checkpoint(
                path=recovery_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch_index=epoch,
                next_chunk_offset=total_chunks,
                epoch_total_chunks=total_chunks,
                completed_epochs=epoch + 1,
                best_val_loss=best_test_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            append_history(
                history_path,
                {
                    "type": "epoch_end",
                    "epoch": epoch + 1,
                    "elapsed_seconds": elapsed,
                    "train": epoch_train_metrics,
                    "train_online": epoch_train_metrics,
                    "train_eval": train_eval_metrics,
                    "test_eval": full_test_metrics,
                    "test": full_test_metrics,
                    "best_test_loss": best_test_loss,
                    "best_epoch": best_epoch,
                    "learning_rate": current_lr,
                    "next_learning_rate": next_lr,
                },
            )
            print(f"[CHECKPOINT] Completed epoch saved: {last_path}", flush=True)
        barrier(distributed)
        resume_offset = 0
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()