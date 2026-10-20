#!/usr/bin/env python3
"""
Chunk-based dataset utilities for the NAP-CNB MLP pipeline.

Training no longer reads master_split.csv. The split preparation step must
first generate, for every physical Primary chunk:

    chunk_metadata/
        chunk_000000_y.npy
        chunk_000000_group_id.npy
        chunk_000000_split_id.npy
        ...

The position in each metadata array is the row index in the corresponding
Primary embedding chunk.

Supported feature modes
-----------------------
primary
    ProtT5 embeddings only (1024 features). No input standardization.

primary_secondary
    ProtT5 (1024) + SecondaryBERT (768) = 1792 features.
    Primary and Secondary are standardized separately using statistics
    estimated from exactly N TRAIN physical chunks (default: 2), then
    concatenated.

Training sampling
-----------------
For every physical TRAIN chunk:
    1. Obtain TRAIN rows from split_id == TRAIN.
    2. Separate positive and negative rows.
    3. Shuffle each class without replacement.
    4. Produce 50/50 minibatches.
    5. If one class is exhausted, reshuffle/recycle that class.
    6. Do not advance to the next physical chunk until every TRAIN row
       of both classes has been seen at least once.

Validation and test keep their natural class distribution.
"""

from __future__ import annotations

from collections import OrderedDict
import argparse
import gc
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd


TRAIN = 0
VALIDATION = 1
TEST = 2

PRIMARY_DIM = 1024
SECONDARY_DIM = 768

FEATURE_MODES = {
    "primary": ("primary",),
    "primary_secondary": ("primary", "secondary"),
}

DEFAULT_SECONDARY_CACHE_SIZE = 2


def validate_feature_mode(feature_mode: str) -> None:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"Unknown feature mode {feature_mode!r}. "
            f"Choose from {sorted(FEATURE_MODES)}."
        )


def input_dimension(feature_mode: str) -> int:
    validate_feature_mode(feature_mode)
    return PRIMARY_DIM if feature_mode == "primary" else PRIMARY_DIM + SECONDARY_DIM


def _find_primary_chunks_dir(dataset_dir: Path) -> Optional[Path]:
    candidates = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir()
        and "ProtT5" in p.name
        and (p / "chunk_metadata").is_dir()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple Primary chunk directories with chunk_metadata under {dataset_dir}:\n"
            + "\n".join(str(p) for p in candidates)
        )
    return None


def _find_secondary_dir(dataset_dir: Path) -> Optional[Path]:
    candidates = sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir()
        and "SecondaryBERT_by_group" in p.name
        and (p / "chunks_manifest.csv").exists()
        and (p / "group_id.npy").exists()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple SecondaryBERT directories under {dataset_dir}:\n"
            + "\n".join(str(p) for p in candidates)
        )
    return None


def discover_datasets(root: Union[str, Path]) -> list[dict]:
    """
    Discover datasets from Primary directories containing chunk_metadata/.

    A dataset record contains:
        name
        dataset_dir
        primary_path
        metadata_path
        secondary_path
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    datasets = []

    for metadata_dir in sorted(root.rglob("chunk_metadata")):
        primary_path = metadata_dir.parent

        if "ProtT5" not in primary_path.name:
            continue

        # Avoid accidentally discovering malformed/empty metadata directories.
        if not any(metadata_dir.glob("chunk_*_split_id.npy")):
            continue

        dataset_dir = primary_path.parent
        secondary_path = _find_secondary_dir(dataset_dir)

        rel = dataset_dir.relative_to(root)
        name = str(rel)

        datasets.append(
            {
                "name": name,
                "dataset_dir": dataset_dir,
                "primary_path": primary_path,
                "metadata_path": metadata_dir,
                "secondary_path": secondary_path,
            }
        )

    if not datasets:
        raise RuntimeError(
            f"No datasets with prepared chunk_metadata were found under:\n{root}\n"
            "Run prepare_chunk_metadata.py first."
        )

    return datasets


def print_datasets(datasets: Sequence[Mapping[str, object]]) -> None:
    print(f"Discovered {len(datasets)} datasets:")
    for i, ds in enumerate(datasets, 1):
        sec = ds.get("secondary_path")
        print(
            f"  [{i:02d}] {ds['name']}\n"
            f"       Primary   : {ds['primary_path']}\n"
            f"       Metadata  : {ds['metadata_path']}\n"
            f"       Secondary : {sec if sec is not None else 'NOT FOUND'}"
        )


def _parse_chunk_id(path: Path, suffix: str) -> int:
    name = path.name
    prefix = "chunk_"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"Unexpected metadata filename: {path}")
    return int(name[len(prefix):-len(suffix)])


def list_chunk_ids(dataset: Mapping[str, object]) -> list[int]:
    metadata_path = Path(dataset["metadata_path"])
    split_files = sorted(metadata_path.glob("chunk_*_split_id.npy"))

    if not split_files:
        raise RuntimeError(f"No split metadata found in {metadata_path}")

    chunk_ids = [_parse_chunk_id(p, "_split_id.npy") for p in split_files]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(f"Duplicate chunk IDs in {metadata_path}")

    return chunk_ids


def _resolve_primary_embedding_file(primary_path: Path, chunk_id: int) -> Path:
    """
    Resolve the Primary embedding file while tolerating the two common layouts:
        <primary>/chunks/chunk_XXXXXX_X.npy
        <primary>/chunk_XXXXXX_X.npy
    """
    names = (
        f"chunk_{chunk_id:06d}_X.npy",
        f"chunk_{chunk_id:06d}_X_primary.npy",
    )

    candidates = []
    for parent in (primary_path / "chunks", primary_path):
        for name in names:
            p = parent / name
            if p.exists():
                candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        # Last-resort exact chunk-prefix lookup, but remain strict about ambiguity.
        matches = []
        for parent in (primary_path / "chunks", primary_path):
            if parent.exists():
                matches.extend(parent.glob(f"chunk_{chunk_id:06d}_X*.npy"))
        matches = sorted(set(matches))
        if len(matches) == 1:
            return matches[0]
        candidates = matches

    raise RuntimeError(
        f"Could not uniquely resolve Primary embedding for chunk {chunk_id} "
        f"under {primary_path}. Candidates: {candidates}"
    )


def load_chunk_metadata(
    dataset: Mapping[str, object],
    chunk_id: int,
    mmap: bool = True,
) -> dict:
    """
    Load y, group_id and split_id for one complete physical chunk.

    Array position == row_idx_in_chunk by construction of prepare_chunk_metadata.py.
    """
    metadata_path = Path(dataset["metadata_path"])
    mode = "r" if mmap else None

    y_file = metadata_path / f"chunk_{chunk_id:06d}_y.npy"
    gid_file = metadata_path / f"chunk_{chunk_id:06d}_group_id.npy"
    split_file = metadata_path / f"chunk_{chunk_id:06d}_split_id.npy"

    for p in (y_file, gid_file, split_file):
        if not p.exists():
            raise FileNotFoundError(p)

    y = np.load(y_file, mmap_mode=mode)
    group_id = np.load(gid_file, mmap_mode=mode)
    split_id = np.load(split_file, mmap_mode=mode)

    n = len(y)
    if len(group_id) != n or len(split_id) != n:
        raise RuntimeError(
            f"{dataset['name']} chunk {chunk_id}: inconsistent metadata lengths "
            f"(y={len(y)}, group_id={len(group_id)}, split_id={len(split_id)})."
        )

    if y.ndim != 1 or group_id.ndim != 1 or split_id.ndim != 1:
        raise RuntimeError(
            f"{dataset['name']} chunk {chunk_id}: metadata arrays must be 1-D."
        )

    return {
        "chunk_id": int(chunk_id),
        "y": y,
        "group_id": group_id,
        "split_id": split_id,
        "n_rows": n,
    }


class SecondaryBERTStore:
    """
    Lookup SecondaryBERT embeddings by global group_id.

    The global group_id array is memory-mapped and searched with np.searchsorted.
    Secondary physical embedding chunks are opened lazily and retained in a
    small LRU mmap cache.
    """

    def __init__(
        self,
        root: Union[str, Path],
        cache_size: int = DEFAULT_SECONDARY_CACHE_SIZE,
    ) -> None:
        self.root = Path(root)
        self.cache_size = int(cache_size)

        if self.cache_size < 1:
            raise ValueError("cache_size must be >= 1")

        gid_file = self.root / "group_id.npy"
        manifest_file = self.root / "chunks_manifest.csv"

        if not gid_file.exists():
            raise FileNotFoundError(gid_file)
        if not manifest_file.exists():
            raise FileNotFoundError(manifest_file)

        self.global_group_ids = np.load(gid_file, mmap_mode="r")
        self.manifest = pd.read_csv(manifest_file)

        required = {"chunk_id", "start", "end", "x_file"}
        missing = required.difference(self.manifest.columns)
        if missing:
            raise RuntimeError(
                f"{manifest_file} is missing columns: {sorted(missing)}"
            )

        self.starts = self.manifest["start"].to_numpy(dtype=np.int64)
        self.ends = self.manifest["end"].to_numpy(dtype=np.int64)
        self.chunk_ids = self.manifest["chunk_id"].to_numpy(dtype=np.int64)
        self.x_files = self.manifest["x_file"].astype(str).to_numpy()

        if len(self.global_group_ids) == 0:
            raise RuntimeError(f"Empty Secondary group_id array: {gid_file}")

        if np.any(self.global_group_ids[1:] <= self.global_group_ids[:-1]):
            raise RuntimeError(
                f"Secondary global group_id must be strictly increasing: {gid_file}"
            )

        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _open_embedding_chunk(self, manifest_row: int) -> np.ndarray:
        manifest_row = int(manifest_row)

        if manifest_row in self._cache:
            arr = self._cache.pop(manifest_row)
            self._cache[manifest_row] = arr
            return arr

        x_file = Path(self.x_files[manifest_row])
        if not x_file.is_absolute():
            x_file = self.root / x_file

        if not x_file.exists():
            raise FileNotFoundError(x_file)

        arr = np.load(x_file, mmap_mode="r")

        if arr.ndim != 2 or arr.shape[1] != SECONDARY_DIM:
            raise RuntimeError(
                f"Unexpected Secondary embedding shape {arr.shape} in {x_file}"
            )

        self._cache[manifest_row] = arr

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return arr

    def get(self, group_ids: np.ndarray) -> np.ndarray:
        """
        Return Secondary embeddings in exactly the same order as group_ids.

        Duplicate group IDs are looked up only once within the request.
        """
        group_ids = np.asarray(group_ids, dtype=np.int64)

        if group_ids.ndim != 1:
            raise ValueError("group_ids must be 1-D")
        if len(group_ids) == 0:
            return np.empty((0, SECONDARY_DIM), dtype=np.float32)

        unique_gids, inverse = np.unique(group_ids, return_inverse=True)

        positions = np.searchsorted(self.global_group_ids, unique_gids)

        invalid = positions >= len(self.global_group_ids)
        valid_pos = positions[~invalid]
        invalid[~invalid] |= self.global_group_ids[valid_pos] != unique_gids[~invalid]

        if np.any(invalid):
            missing = unique_gids[invalid][:10]
            raise KeyError(
                f"{np.count_nonzero(invalid)} group_id values are absent from "
                f"SecondaryBERT. First missing: {missing.tolist()}"
            )

        manifest_rows = np.searchsorted(self.ends, positions, side="right")

        if np.any(manifest_rows >= len(self.ends)):
            raise RuntimeError("Secondary position falls outside manifest ranges.")

        if np.any(positions < self.starts[manifest_rows]):
            raise RuntimeError("Secondary manifest contains inconsistent ranges.")

        unique_embeddings = np.empty(
            (len(unique_gids), SECONDARY_DIM),
            dtype=np.float32,
        )

        for manifest_row in np.unique(manifest_rows):
            mask = manifest_rows == manifest_row
            local_rows = positions[mask] - self.starts[manifest_row]

            emb = self._open_embedding_chunk(int(manifest_row))
            unique_embeddings[mask] = emb[local_rows]

        return unique_embeddings[inverse]


def _get_secondary_store(
    dataset: Mapping[str, object],
    stores: Dict[str, SecondaryBERTStore],
    cache_size: int,
) -> SecondaryBERTStore:
    key = str(dataset["name"])

    if key not in stores:
        secondary_path = dataset.get("secondary_path")
        if secondary_path is None:
            raise FileNotFoundError(
                f"No SecondaryBERT directory found for dataset {dataset['name']!r}."
            )
        stores[key] = SecondaryBERTStore(
            secondary_path,
            cache_size=cache_size,
        )

    return stores[key]


def load_rows(
    dataset: Mapping[str, object],
    chunk_id: int,
    row_indices: np.ndarray,
    group_ids: np.ndarray,
    feature_mode: str,
    secondary_stores: Optional[Dict[str, SecondaryBERTStore]] = None,
    secondary_cache_size: int = DEFAULT_SECONDARY_CACHE_SIZE,
) -> np.ndarray:
    """Load selected rows from one physical chunk."""
    validate_feature_mode(feature_mode)

    row_indices = np.asarray(row_indices, dtype=np.int64)
    group_ids = np.asarray(group_ids, dtype=np.int64)

    if len(row_indices) != len(group_ids):
        raise ValueError("row_indices and group_ids must have equal length.")

    primary_path = Path(dataset["primary_path"])
    x_file = _resolve_primary_embedding_file(primary_path, chunk_id)
    X_primary_mmap = np.load(x_file, mmap_mode="r")

    if X_primary_mmap.ndim != 2 or X_primary_mmap.shape[1] != PRIMARY_DIM:
        raise RuntimeError(
            f"Unexpected Primary shape {X_primary_mmap.shape} in {x_file}"
        )

    if len(row_indices):
        if row_indices.min() < 0 or row_indices.max() >= len(X_primary_mmap):
            raise IndexError(
                f"{dataset['name']} chunk {chunk_id}: row index outside "
                f"Primary array with {len(X_primary_mmap)} rows."
            )

    X_primary = np.asarray(X_primary_mmap[row_indices], dtype=np.float32)

    if feature_mode == "primary":
        return X_primary

    if secondary_stores is None:
        raise ValueError(
            "secondary_stores is required for primary_secondary."
        )

    store = _get_secondary_store(
        dataset,
        secondary_stores,
        secondary_cache_size,
    )
    X_secondary = store.get(group_ids)

    X = np.empty(
        (len(row_indices), PRIMARY_DIM + SECONDARY_DIM),
        dtype=np.float32,
    )
    X[:, :PRIMARY_DIM] = X_primary
    X[:, PRIMARY_DIM:] = X_secondary

    return X


class CyclingClassSampler:
    """
    Shuffle a class without replacement and recycle only after exhaustion.

    Consequently, requesting enough samples to cover the larger class guarantees
    that every row from both classes is observed at least once.
    """

    def __init__(
        self,
        indices: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = rng

        if len(self.indices) == 0:
            raise ValueError("Cannot create a sampler for an empty class.")

        self.order = self.rng.permutation(self.indices)
        self.cursor = 0

    def take(self, n: int) -> np.ndarray:
        pieces = []
        remaining = int(n)

        while remaining > 0:
            available = len(self.order) - self.cursor
            take_now = min(remaining, available)

            pieces.append(
                self.order[self.cursor:self.cursor + take_now]
            )

            self.cursor += take_now
            remaining -= take_now

            if self.cursor == len(self.order) and remaining > 0:
                self.order = self.rng.permutation(self.indices)
                self.cursor = 0

        return np.concatenate(pieces)


def iter_balanced_train_batches(
    datasets: Sequence[Mapping[str, object]],
    feature_mode: str,
    batch_size: int,
    seed: int = 42,
    epoch: int = 0,
    secondary_cache_size: int = DEFAULT_SECONDARY_CACHE_SIZE,
    shuffle_datasets: bool = True,
    shuffle_chunks: bool = False,
) -> Iterator[dict]:
    """
    Yield 50/50 TRAIN minibatches while completing one physical chunk at a time.

    Every TRAIN row in a chunk is guaranteed to be sampled at least once before
    advancing to the next physical chunk.

    batch_size must be even because every minibatch contains exactly half
    positives and half negatives.
    """
    validate_feature_mode(feature_mode)

    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError("batch_size must be a positive even number.")

    half = batch_size // 2
    secondary_stores: Dict[str, SecondaryBERTStore] = {}

    dataset_order = np.arange(len(datasets))
    epoch_rng = np.random.default_rng(
        np.random.SeedSequence([seed, epoch, 918273])
    )

    if shuffle_datasets:
        epoch_rng.shuffle(dataset_order)

    for dataset_position in dataset_order:
        dataset = datasets[int(dataset_position)]
        chunk_ids = list_chunk_ids(dataset)

        if shuffle_chunks:
            chunk_ids = list(
                epoch_rng.permutation(np.asarray(chunk_ids, dtype=np.int64))
            )

        for chunk_id in chunk_ids:
            chunk_id = int(chunk_id)
            meta = load_chunk_metadata(dataset, chunk_id, mmap=True)

            split_id = meta["split_id"]
            y_all = meta["y"]
            gid_all = meta["group_id"]

            train_rows = np.flatnonzero(split_id == TRAIN).astype(
                np.int64, copy=False
            )

            if len(train_rows) == 0:
                continue

            train_y = np.asarray(y_all[train_rows], dtype=np.uint8)

            neg_rows = train_rows[train_y == 0]
            pos_rows = train_rows[train_y == 1]

            if len(neg_rows) == 0 or len(pos_rows) == 0:
                raise RuntimeError(
                    f"{dataset['name']} chunk {chunk_id}: balanced training "
                    f"requires both classes, found neg={len(neg_rows):,}, "
                    f"pos={len(pos_rows):,}."
                )

            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [seed, epoch, int(dataset_position), chunk_id]
                )
            )

            neg_sampler = CyclingClassSampler(neg_rows, rng)
            pos_sampler = CyclingClassSampler(pos_rows, rng)

            n_batches = int(
                np.ceil(max(len(neg_rows), len(pos_rows)) / half)
            )

            for batch_index in range(n_batches):
                neg = neg_sampler.take(half)
                pos = pos_sampler.take(half)

                selected_rows = np.concatenate((neg, pos))
                rng.shuffle(selected_rows)

                # Sorting improves mmap locality. Restore the randomized batch
                # order after loading.
                sort_order = np.argsort(selected_rows)
                sorted_rows = selected_rows[sort_order]
                restore_order = np.argsort(sort_order)

                sorted_gids = np.asarray(
                    gid_all[sorted_rows],
                    dtype=np.int64,
                )

                X_sorted = load_rows(
                    dataset=dataset,
                    chunk_id=chunk_id,
                    row_indices=sorted_rows,
                    group_ids=sorted_gids,
                    feature_mode=feature_mode,
                    secondary_stores=secondary_stores,
                    secondary_cache_size=secondary_cache_size,
                )

                X = X_sorted[restore_order]
                y = np.asarray(
                    y_all[selected_rows],
                    dtype=np.float32,
                )

                yield {
                    "X": X,
                    "y": y,
                    "dataset_name": str(dataset["name"]),
                    "chunk_id": chunk_id,
                    "batch_index": batch_index,
                    "n_batches": n_batches,
                    "n_train_rows": len(train_rows),
                    "n_neg": len(neg_rows),
                    "n_pos": len(pos_rows),
                    # Useful for correctness/debug checks; negligible compared
                    # with the embedding matrix.
                    "row_indices": selected_rows,
                }


def iter_natural_batches(
    datasets: Sequence[Mapping[str, object]],
    feature_mode: str,
    split: int,
    batch_size: int,
    secondary_cache_size: int = DEFAULT_SECONDARY_CACHE_SIZE,
) -> Iterator[dict]:
    """Yield VALIDATION or TEST rows with their natural class distribution."""
    validate_feature_mode(feature_mode)

    if split not in (VALIDATION, TEST):
        raise ValueError(
            "iter_natural_batches is intended for VALIDATION or TEST."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    secondary_stores: Dict[str, SecondaryBERTStore] = {}

    for dataset in datasets:
        for chunk_id in list_chunk_ids(dataset):
            meta = load_chunk_metadata(dataset, chunk_id, mmap=True)

            rows = np.flatnonzero(meta["split_id"] == split).astype(
                np.int64, copy=False
            )

            for start in range(0, len(rows), batch_size):
                selected_rows = rows[start:start + batch_size]

                gids = np.asarray(
                    meta["group_id"][selected_rows],
                    dtype=np.int64,
                )

                X = load_rows(
                    dataset=dataset,
                    chunk_id=chunk_id,
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
                    "dataset_name": str(dataset["name"]),
                    "chunk_id": int(chunk_id),
                    "row_indices": selected_rows,
                }


@dataclass
class ModalityStandardizer:
    """
    Fixed per-feature normalization for Primary + Secondary.

    Primary-only experiments bypass this class entirely.
    """

    primary_mean: np.ndarray
    primary_scale: np.ndarray
    secondary_mean: np.ndarray
    secondary_scale: np.ndarray
    sampled_dataset_names: np.ndarray
    sampled_chunk_ids: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)

        expected_dim = PRIMARY_DIM + SECONDARY_DIM
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(
                f"Expected (N, {expected_dim}), found {X.shape}."
            )

        if not X.flags.writeable:
            X = X.copy()

        X[:, :PRIMARY_DIM] -= self.primary_mean
        X[:, :PRIMARY_DIM] /= self.primary_scale

        X[:, PRIMARY_DIM:] -= self.secondary_mean
        X[:, PRIMARY_DIM:] /= self.secondary_scale

        return X

    def save(self, path: Union[str, Path]) -> None:
        np.savez(
            path,
            primary_mean=self.primary_mean,
            primary_scale=self.primary_scale,
            secondary_mean=self.secondary_mean,
            secondary_scale=self.secondary_scale,
            sampled_dataset_names=self.sampled_dataset_names,
            sampled_chunk_ids=self.sampled_chunk_ids,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModalityStandardizer":
        data = np.load(path)
        return cls(
            primary_mean=data["primary_mean"].astype(np.float32),
            primary_scale=data["primary_scale"].astype(np.float32),
            secondary_mean=data["secondary_mean"].astype(np.float32),
            secondary_scale=data["secondary_scale"].astype(np.float32),
            sampled_dataset_names=data["sampled_dataset_names"],
            sampled_chunk_ids=data["sampled_chunk_ids"].astype(np.int64),
        )


class _RunningMoments:
    """Streaming mean and variance for one modality."""

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)

    def update(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(
                f"Expected (N, {self.dimension}), found {X.shape}."
            )
        if len(X) == 0:
            return

        n = len(X)
        block_mean = X.mean(axis=0)
        centered = X - block_mean
        block_m2 = np.einsum("ij,ij->j", centered, centered)

        if self.count == 0:
            self.mean[:] = block_mean
            self.m2[:] = block_m2
            self.count = n
            return

        delta = block_mean - self.mean
        new_count = self.count + n

        self.mean += delta * (n / new_count)
        self.m2 += (
            block_m2
            + delta * delta * self.count * n / new_count
        )
        self.count = new_count

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError("Not enough rows to estimate normalization.")

        variance = self.m2 / self.count
        scale = np.sqrt(variance)
        scale[scale < 1e-12] = 1.0

        return (
            self.mean.astype(np.float32),
            scale.astype(np.float32),
        )


def fit_primary_secondary_standardizer(
    datasets: Sequence[Mapping[str, object]],
    n_train_chunks: int = 2,
    seed: int = 42,
    secondary_cache_size: int = DEFAULT_SECONDARY_CACHE_SIZE,
    stats_batch_size: int = 8192,
) -> ModalityStandardizer:
    """
    Estimate separate Primary and Secondary statistics from N TRAIN chunks.

    The chunks are selected reproducibly from the prepared chunk metadata.
    Only TRAIN rows from those chunks contribute. No master_split.csv is read.
    """
    if n_train_chunks <= 0:
        raise ValueError("n_train_chunks must be > 0")
    if stats_batch_size <= 0:
        raise ValueError("stats_batch_size must be > 0")

    candidates = []
    for dataset_index, dataset in enumerate(datasets):
        for chunk_id in list_chunk_ids(dataset):
            meta = load_chunk_metadata(dataset, chunk_id, mmap=True)
            n_train = int(np.count_nonzero(meta["split_id"] == TRAIN))
            if n_train > 0:
                candidates.append((dataset_index, int(chunk_id)))

    if len(candidates) < n_train_chunks:
        raise RuntimeError(
            f"Requested {n_train_chunks} TRAIN chunks, "
            f"but only {len(candidates)} are available."
        )

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        len(candidates),
        size=n_train_chunks,
        replace=False,
    )

    primary_stats = _RunningMoments(PRIMARY_DIM)
    secondary_stats = _RunningMoments(SECONDARY_DIM)
    secondary_stores: Dict[str, SecondaryBERTStore] = {}

    sampled_dataset_names = []
    sampled_chunk_ids = []

    print(
        f"Estimating Primary/Secondary normalization from "
        f"{n_train_chunks} physical TRAIN chunks..."
    )

    for candidate_position in selected:
        dataset_index, chunk_id = candidates[int(candidate_position)]
        dataset = datasets[dataset_index]
        meta = load_chunk_metadata(dataset, chunk_id, mmap=True)

        train_rows = np.flatnonzero(
            meta["split_id"] == TRAIN
        ).astype(np.int64, copy=False)

        sampled_dataset_names.append(str(dataset["name"]))
        sampled_chunk_ids.append(chunk_id)

        print(
            f"  scaler sample: dataset={dataset['name']} "
            f"chunk={chunk_id:06d} train_rows={len(train_rows):,}"
        )

        for start in range(0, len(train_rows), stats_batch_size):
            rows = train_rows[start:start + stats_batch_size]
            gids = np.asarray(meta["group_id"][rows], dtype=np.int64)

            # Load the combined matrix once, then accumulate the two modalities
            # separately. This preserves the intended modality-specific scaling.
            X = load_rows(
                dataset=dataset,
                chunk_id=chunk_id,
                row_indices=rows,
                group_ids=gids,
                feature_mode="primary_secondary",
                secondary_stores=secondary_stores,
                secondary_cache_size=secondary_cache_size,
            )

            primary_stats.update(X[:, :PRIMARY_DIM])
            secondary_stats.update(X[:, PRIMARY_DIM:])

    primary_mean, primary_scale = primary_stats.finalize()
    secondary_mean, secondary_scale = secondary_stats.finalize()

    print(
        f"Normalization estimated from {primary_stats.count:,} TRAIN rows."
    )

    return ModalityStandardizer(
        primary_mean=primary_mean,
        primary_scale=primary_scale,
        secondary_mean=secondary_mean,
        secondary_scale=secondary_scale,
        sampled_dataset_names=np.asarray(sampled_dataset_names),
        sampled_chunk_ids=np.asarray(sampled_chunk_ids, dtype=np.int64),
    )


def transform_input(
    X: np.ndarray,
    feature_mode: str,
    standardizer: Optional[ModalityStandardizer],
) -> np.ndarray:
    """
    Apply feature preprocessing.

    primary:
        Return ProtT5 embeddings unchanged.

    primary_secondary:
        Apply the fixed modality-specific standardizer.
    """
    validate_feature_mode(feature_mode)

    X = np.asarray(X, dtype=np.float32)

    if feature_mode == "primary":
        return X

    if standardizer is None:
        raise RuntimeError(
            "primary_secondary requires a ModalityStandardizer."
        )

    return standardizer.transform(X)


# =============================================================================
# Standalone diagnostic
# =============================================================================

def _bytes_to_gib(value: int) -> float:
    return float(value) / (1024 ** 3)


def _read_proc_status_kib(field: str) -> Optional[int]:
    """Read a memory field from /proc/self/status without extra dependencies."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _system_memory() -> dict:
    """Read Linux system RAM information from /proc/meminfo."""
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.split()[0]) * 1024
    except OSError:
        return {}

    return {
        "total": values.get("MemTotal", 0),
        "available": values.get("MemAvailable", 0),
        "swap_total": values.get("SwapTotal", 0),
        "swap_free": values.get("SwapFree", 0),
    }


def _gpu_snapshot() -> list[dict]:
    """
    Query GPU state with nvidia-smi.

    dataset.py itself does not allocate CUDA tensors. GPU memory should therefore
    remain essentially unchanged during this loader diagnostic.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "used_mib": int(parts[2]),
                "total_mib": int(parts[3]),
                "util": int(parts[4]),
            }
        )
    return gpus


def print_resource_snapshot(label: str) -> None:
    """Print process RAM, system RAM/swap and GPU state."""
    rss_kib = _read_proc_status_kib("VmRSS")
    peak_kib = _read_proc_status_kib("VmHWM")
    sysmem = _system_memory()

    print(f"\n[RESOURCES] {label}")

    if rss_kib is not None:
        print(
            f"  Process RAM : RSS={rss_kib / 1024:.1f} MiB"
            + (
                f" | peak={peak_kib / 1024:.1f} MiB"
                if peak_kib is not None else ""
            )
        )

    if sysmem:
        used = sysmem["total"] - sysmem["available"]
        print(
            f"  System RAM  : used~{_bytes_to_gib(used):.2f} GiB"
            f" | available={_bytes_to_gib(sysmem['available']):.2f} GiB"
            f" | total={_bytes_to_gib(sysmem['total']):.2f} GiB"
        )

        swap_used = sysmem["swap_total"] - sysmem["swap_free"]
        print(
            f"  Swap        : used={_bytes_to_gib(swap_used):.2f} GiB"
            f" | total={_bytes_to_gib(sysmem['swap_total']):.2f} GiB"
        )

    gpus = _gpu_snapshot()
    if not gpus:
        print("  GPU         : nvidia-smi unavailable")
    else:
        for gpu in gpus:
            print(
                f"  GPU {gpu['index']}       : {gpu['name']} | "
                f"{gpu['used_mib']} / {gpu['total_mib']} MiB | "
                f"util={gpu['util']}%"
            )


def secondary_manifest_rows_for_group_ids(
    store: SecondaryBERTStore,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Return the Secondary physical manifest rows touched by group_ids."""
    gids = np.unique(np.asarray(group_ids, dtype=np.int64))
    positions = np.searchsorted(store.global_group_ids, gids)

    invalid = positions >= len(store.global_group_ids)
    valid = ~invalid
    if np.any(valid):
        invalid[valid] |= store.global_group_ids[positions[valid]] != gids[valid]

    if np.any(invalid):
        raise KeyError(
            f"Missing Secondary group IDs: {gids[invalid][:10].tolist()}"
        )

    manifest_rows = np.searchsorted(store.ends, positions, side="right")

    if np.any(manifest_rows >= len(store.ends)):
        raise RuntimeError("Secondary lookup outside manifest.")

    return np.unique(manifest_rows)


def _select_datasets_for_diagnostic(
    datasets: Sequence[Mapping[str, object]],
    pattern: Optional[str],
) -> list[Mapping[str, object]]:
    if not pattern:
        return [datasets[0]]

    pattern_lower = pattern.lower()
    selected = [
        ds for ds in datasets
        if pattern_lower in str(ds["name"]).lower()
    ]

    if not selected:
        names = "\n".join(f"  - {ds['name']}" for ds in datasets)
        raise ValueError(
            f"No dataset matches {pattern!r}. Available datasets:\n{names}"
        )

    if len(selected) > 1:
        names = "\n".join(f"  - {ds['name']}" for ds in selected)
        raise ValueError(
            f"Dataset pattern {pattern!r} matches more than one dataset:\n{names}\n"
            "Use a more specific value."
        )

    return selected


def diagnostic_chunk(
    dataset: Mapping[str, object],
    chunk_id: int,
    feature_mode: str,
    batch_size: int,
    seed: int,
    secondary_cache_size: int,
    resource_every: int,
) -> dict:
    """
    Exhaustively test the balanced TRAIN sampler for one physical chunk.

    This does NOT train a model. It verifies:
      - metadata/Primary row-count agreement
      - TRAIN/VAL/TEST counts
      - 50/50 composition of every TRAIN minibatch
      - exact feature dimensions
      - finite loaded embeddings
      - Secondary group_id lookup
      - complete TRAIN-row coverage before leaving the chunk
      - expected class recycling
      - RAM/GPU behaviour during loading
    """
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("--batch-size must be a positive even number.")

    half = batch_size // 2

    print("\n" + "=" * 88)
    print(f"DATASET : {dataset['name']}")
    print(f"CHUNK   : {chunk_id:06d}")
    print(f"MODE    : {feature_mode}")
    print("=" * 88)

    meta = load_chunk_metadata(dataset, chunk_id, mmap=True)

    primary_file = _resolve_primary_embedding_file(
        Path(dataset["primary_path"]),
        chunk_id,
    )
    primary_mmap = np.load(primary_file, mmap_mode="r")

    if len(primary_mmap) != meta["n_rows"]:
        raise RuntimeError(
            f"Primary/metadata row mismatch: Primary={len(primary_mmap):,}, "
            f"metadata={meta['n_rows']:,}"
        )

    if primary_mmap.shape[1] != PRIMARY_DIM:
        raise RuntimeError(
            f"Primary dimension mismatch: {primary_mmap.shape}"
        )

    y_all = meta["y"]
    split_all = meta["split_id"]
    gid_all = meta["group_id"]

    train_rows = np.flatnonzero(split_all == TRAIN).astype(np.int64, copy=False)
    val_rows = np.flatnonzero(split_all == VALIDATION)
    test_rows = np.flatnonzero(split_all == TEST)

    train_y = np.asarray(y_all[train_rows], dtype=np.uint8)
    neg_rows = train_rows[train_y == 0]
    pos_rows = train_rows[train_y == 1]

    print(
        f"Rows     : total={meta['n_rows']:,} | "
        f"train={len(train_rows):,} | val={len(val_rows):,} | test={len(test_rows):,}"
    )
    print(
        f"TRAIN    : neg={len(neg_rows):,} | pos={len(pos_rows):,} | "
        f"positive_rate={len(pos_rows) / len(train_rows):.4%}"
    )
    print(f"Primary  : {primary_file}")
    print(
        f"Primary shape={primary_mmap.shape} dtype={primary_mmap.dtype} "
        f"(mmap={isinstance(primary_mmap, np.memmap)})"
    )

    if len(neg_rows) == 0 or len(pos_rows) == 0:
        raise RuntimeError(
            f"Balanced sampling impossible: neg={len(neg_rows)}, pos={len(pos_rows)}"
        )

    secondary_stores: Dict[str, SecondaryBERTStore] = {}
    secondary_manifest_rows = np.empty(0, dtype=np.int64)

    if feature_mode == "primary_secondary":
        store = _get_secondary_store(
            dataset,
            secondary_stores,
            secondary_cache_size,
        )
        secondary_manifest_rows = secondary_manifest_rows_for_group_ids(
            store,
            np.asarray(gid_all[train_rows], dtype=np.int64),
        )
        secondary_chunk_ids = store.chunk_ids[secondary_manifest_rows]
        print(
            f"Secondary: TRAIN group_ids touch "
            f"{len(secondary_manifest_rows)} physical Secondary chunk(s): "
            f"{secondary_chunk_ids.tolist()}"
        )
        print(
            f"Secondary mmap cache limit: {secondary_cache_size} chunk(s)"
        )

    rng = np.random.default_rng(
        np.random.SeedSequence([seed, chunk_id, 314159])
    )
    neg_sampler = CyclingClassSampler(neg_rows, rng)
    pos_sampler = CyclingClassSampler(pos_rows, rng)

    n_batches = int(np.ceil(max(len(neg_rows), len(pos_rows)) / half))
    expected_samples_per_class = n_batches * half

    print(
        f"Sampler  : batch_size={batch_size:,} = {half:,} neg + {half:,} pos"
    )
    print(
        f"           batches={n_batches:,} | "
        f"draws/class={expected_samples_per_class:,}"
    )
    print(
        f"           expected recycling: "
        f"neg x{expected_samples_per_class / len(neg_rows):.2f}, "
        f"pos x{expected_samples_per_class / len(pos_rows):.2f}"
    )

    # Boolean coverage arrays are tiny (~0.1 MB for a 100k-row chunk) and make
    # the coverage assertion exact rather than approximate.
    seen_neg = np.zeros(meta["n_rows"], dtype=bool)
    seen_pos = np.zeros(meta["n_rows"], dtype=bool)

    total_neg_draws = 0
    total_pos_draws = 0
    expected_dim = input_dimension(feature_mode)

    print_resource_snapshot("before first batch")
    t0 = time.perf_counter()

    for batch_index in range(n_batches):
        neg = neg_sampler.take(half)
        pos = pos_sampler.take(half)

        seen_neg[neg] = True
        seen_pos[pos] = True
        total_neg_draws += len(neg)
        total_pos_draws += len(pos)

        selected_rows = np.concatenate((neg, pos))
        rng.shuffle(selected_rows)

        # Same locality strategy used by the real training iterator.
        order = np.argsort(selected_rows)
        sorted_rows = selected_rows[order]
        restore = np.argsort(order)

        gids = np.asarray(gid_all[sorted_rows], dtype=np.int64)

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
        y = np.asarray(y_all[selected_rows], dtype=np.float32)

        if X.shape != (batch_size, expected_dim):
            raise RuntimeError(
                f"Batch {batch_index}: unexpected X shape {X.shape}; "
                f"expected {(batch_size, expected_dim)}."
            )

        if y.shape != (batch_size,):
            raise RuntimeError(
                f"Batch {batch_index}: unexpected y shape {y.shape}."
            )

        n_pos_batch = int(np.count_nonzero(y == 1))
        n_neg_batch = int(np.count_nonzero(y == 0))

        if n_pos_batch != half or n_neg_batch != half:
            raise RuntimeError(
                f"Batch {batch_index}: not 50/50: "
                f"neg={n_neg_batch}, pos={n_pos_batch}."
            )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"Batch {batch_index}: NaN/Inf found in embeddings."
            )

        if batch_index == 0 or batch_index == n_batches - 1:
            print(
                f"Batch {batch_index + 1:>4}/{n_batches}: "
                f"X={X.shape} {X.dtype} | "
                f"neg={n_neg_batch:,} pos={n_pos_batch:,} | "
                f"X mean={float(X.mean()):.5f} std={float(X.std()):.5f}"
            )

        if (
            resource_every > 0
            and (
                (batch_index + 1) % resource_every == 0
                or batch_index == n_batches - 1
            )
        ):
            print_resource_snapshot(
                f"after batch {batch_index + 1}/{n_batches}"
            )

        # Explicitly release the large temporary batch before the next one.
        del X_sorted, X, y, selected_rows, sorted_rows, gids

    elapsed = time.perf_counter() - t0

    covered_neg = int(np.count_nonzero(seen_neg[neg_rows]))
    covered_pos = int(np.count_nonzero(seen_pos[pos_rows]))

    missing_neg = len(neg_rows) - covered_neg
    missing_pos = len(pos_rows) - covered_pos

    if missing_neg or missing_pos:
        raise RuntimeError(
            f"INCOMPLETE COVERAGE: missing neg={missing_neg:,}, "
            f"missing pos={missing_pos:,}."
        )

    print("\n[COVERAGE]")
    print(
        f"  Negatives: {covered_neg:,}/{len(neg_rows):,} unique TRAIN rows seen "
        f"({covered_neg / len(neg_rows):.2%}) | total draws={total_neg_draws:,}"
    )
    print(
        f"  Positives: {covered_pos:,}/{len(pos_rows):,} unique TRAIN rows seen "
        f"({covered_pos / len(pos_rows):.2%}) | total draws={total_pos_draws:,}"
    )
    print(
        f"  Repeated draws caused by balancing: "
        f"neg={total_neg_draws - covered_neg:,}, "
        f"pos={total_pos_draws - covered_pos:,}"
    )
    print(f"  Time: {elapsed:.2f} s ({elapsed / n_batches:.3f} s/batch)")

    # Close references and make the post-chunk memory state visible.
    del primary_mmap, meta, seen_neg, seen_pos
    secondary_stores.clear()
    gc.collect()

    print_resource_snapshot("after chunk + gc")
    print("\nRESULT   : PASS")

    return {
        "dataset": str(dataset["name"]),
        "chunk_id": chunk_id,
        "n_batches": n_batches,
        "n_train": len(train_rows),
        "n_neg": len(neg_rows),
        "n_pos": len(pos_rows),
        "secondary_chunks": len(secondary_manifest_rows),
        "elapsed_seconds": elapsed,
    }


def run_standalone_diagnostic(args: argparse.Namespace) -> None:
    validate_feature_mode(args.features)

    datasets = discover_datasets(args.root)
    print_datasets(datasets)

    selected_datasets = _select_datasets_for_diagnostic(
        datasets,
        args.dataset,
    )
    dataset = selected_datasets[0]

    chunk_ids = list_chunk_ids(dataset)

    if args.chunk_id is not None:
        if args.chunk_id not in chunk_ids:
            raise ValueError(
                f"Chunk {args.chunk_id} does not exist in {dataset['name']}."
            )
        selected_chunks = [args.chunk_id]
    else:
        selected_chunks = chunk_ids[:args.chunks]

    if not selected_chunks:
        raise RuntimeError("No chunks selected.")

    print("\n" + "#" * 88)
    print("STANDALONE DATASET DIAGNOSTIC")
    print("#" * 88)
    print(f"Selected dataset : {dataset['name']}")
    print(f"Selected chunks  : {selected_chunks}")
    print(f"Feature mode     : {args.features}")
    print(f"Batch size       : {args.batch_size:,}")
    print(f"Secondary cache  : {args.secondary_cache_size}")
    print(
        "NOTE: GPU tensors are intentionally NOT created. GPU memory should stay "
        "flat; this test isolates dataset/I/O/RAM behaviour."
    )

    print_resource_snapshot("diagnostic start")

    results = []

    for chunk_id in selected_chunks:
        result = diagnostic_chunk(
            dataset=dataset,
            chunk_id=int(chunk_id),
            feature_mode=args.features,
            batch_size=args.batch_size,
            seed=args.seed,
            secondary_cache_size=args.secondary_cache_size,
            resource_every=args.resource_every,
        )
        results.append(result)

    print("\n" + "#" * 88)
    print("FINAL SUMMARY")
    print("#" * 88)

    for result in results:
        print(
            f"PASS | chunk={result['chunk_id']:06d} | "
            f"train={result['n_train']:,} | "
            f"neg={result['n_neg']:,} | pos={result['n_pos']:,} | "
            f"batches={result['n_batches']:,} | "
            f"Secondary chunks={result['secondary_chunks']} | "
            f"time={result['elapsed_seconds']:.2f}s"
        )

    print_resource_snapshot("diagnostic end")
    print("\nALL SELECTED CHUNKS PASSED.")


def _build_diagnostic_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone integrity/resource diagnostic for the chunk-based "
            "NAP-CNB dataset loader. No neural network is trained."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/nap/lperez_nn/data/data_input_nn"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HLA-A",
        help="Unique substring identifying one dataset (default: HLA-A).",
    )
    parser.add_argument(
        "--features",
        choices=sorted(FEATURE_MODES),
        default="primary_secondary",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=2,
        help="Number of first physical chunks to test (default: 2).",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="Test one exact chunk instead of --chunks.",
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
        help="Maximum number of Secondary embedding mmaps kept open.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--resource-every",
        type=int,
        default=2,
        help="Print RAM/GPU state every N batches; 0 disables intermediate snapshots.",
    )

    return parser


if __name__ == "__main__":
    run_standalone_diagnostic(
        _build_diagnostic_parser().parse_args()
    )