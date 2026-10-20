#!/usr/bin/env python3

"""
Generate compact per-chunk metadata from master_split.csv.

For each physical chunk, creates:

    chunk_metadata/
        chunk_000000_y.npy
        chunk_000000_group_id.npy
        chunk_000000_split_id.npy
        chunk_000001_y.npy
        ...

The original master_split.csv is only read during this preparation step.
Training can subsequently operate directly on physical chunks without
parsing master_split.csv.

Expected master_split.csv columns:
    chunk_id
    row_idx_in_chunk
    y
    group_id
    split_id

protein_group_id is intentionally not loaded because it is not required
during model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/data/nap/lperez_nn/data/data_input_nn"
)

CSV_READ_ROWS = 500_000
OUTPUT_DIRNAME = "chunk_metadata"


def discover_master_splits(root: Path) -> list[Path]:
    """Find all master_split.csv files below root."""
    files = sorted(root.rglob("master_split.csv"))

    if not files:
        raise FileNotFoundError(
            f"No master_split.csv files found under:\n{root}"
        )

    return files


def validate_chunk(
    chunk_id: int,
    row_idx: np.ndarray,
    y: np.ndarray,
    group_id: np.ndarray,
    split_id: np.ndarray,
) -> None:
    """Run strict integrity checks before saving a physical chunk."""

    n = len(row_idx)

    if n == 0:
        raise RuntimeError(f"Chunk {chunk_id} is empty.")

    if not (
        len(y) == n
        and len(group_id) == n
        and len(split_id) == n
    ):
        raise RuntimeError(
            f"Chunk {chunk_id}: inconsistent metadata lengths."
        )

    # row_idx_in_chunk must map directly to array position.
    expected = np.arange(n, dtype=row_idx.dtype)

    if not np.array_equal(row_idx, expected):
        bad = np.flatnonzero(row_idx != expected)
        first = int(bad[0])

        raise RuntimeError(
            f"Chunk {chunk_id}: row_idx_in_chunk is not sequential.\n"
            f"First mismatch at position {first}: "
            f"found={row_idx[first]}, expected={expected[first]}"
        )

    if not np.all((y == 0) | (y == 1)):
        values = np.unique(y)
        raise RuntimeError(
            f"Chunk {chunk_id}: invalid y values: {values}"
        )

    if not np.all(
        (split_id == 0)
        | (split_id == 1)
        | (split_id == 2)
    ):
        values = np.unique(split_id)
        raise RuntimeError(
            f"Chunk {chunk_id}: invalid split_id values: {values}"
        )

    if np.any(group_id < 0):
        raise RuntimeError(
            f"Chunk {chunk_id}: negative group_id detected."
        )


def save_chunk(
    output_dir: Path,
    chunk_id: int,
    row_idx: np.ndarray,
    y: np.ndarray,
    group_id: np.ndarray,
    split_id: np.ndarray,
    force: bool,
) -> None:
    """Validate and save metadata for one complete physical chunk."""

    validate_chunk(
        chunk_id=chunk_id,
        row_idx=row_idx,
        y=y,
        group_id=group_id,
        split_id=split_id,
    )

    y_file = output_dir / f"chunk_{chunk_id:06d}_y.npy"
    gid_file = output_dir / f"chunk_{chunk_id:06d}_group_id.npy"
    split_file = output_dir / f"chunk_{chunk_id:06d}_split_id.npy"

    outputs = [y_file, gid_file, split_file]

    existing = [p for p in outputs if p.exists()]

    if existing and not force:
        raise FileExistsError(
            f"Metadata already exists for chunk {chunk_id}.\n"
            f"Use --force to overwrite."
        )

    # Compact dtypes.
    y = np.asarray(y, dtype=np.uint8)
    split_id = np.asarray(split_id, dtype=np.uint8)
    group_id = np.asarray(group_id, dtype=np.int64)

    np.save(y_file, y)
    np.save(gid_file, group_id)
    np.save(split_file, split_id)


def process_master_split(
    master_split: Path,
    csv_read_rows: int,
    force: bool,
) -> None:
    """Convert one master_split.csv into per-chunk NPY metadata."""

    dataset_dir = master_split.parent
    output_dir = dataset_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"Dataset : {dataset_dir.name}")
    print(f"Input   : {master_split}")
    print(f"Output  : {output_dir}")
    print("=" * 80)

    usecols = [
        "chunk_id",
        "row_idx_in_chunk",
        "y",
        "group_id",
        "split_id",
    ]

    dtypes = {
        "chunk_id": np.int32,
        "row_idx_in_chunk": np.int32,
        "y": np.uint8,
        "group_id": np.int64,
        "split_id": np.uint8,
    }

    pending_chunk_id = None
    pending_parts = []

    processed_chunks = 0
    processed_rows = 0
    seen_chunks = set()

    reader = pd.read_csv(
        master_split,
        usecols=usecols,
        dtype=dtypes,
        chunksize=csv_read_rows,
    )

    def flush_pending() -> None:
        nonlocal pending_chunk_id
        nonlocal pending_parts
        nonlocal processed_chunks
        nonlocal processed_rows

        if pending_chunk_id is None:
            return

        if len(pending_parts) == 1:
            df = pending_parts[0]
        else:
            df = pd.concat(
                pending_parts,
                ignore_index=True,
                copy=False,
            )

        cid = int(pending_chunk_id)

        if cid in seen_chunks:
            raise RuntimeError(
                f"Chunk {cid} appears discontinuously in "
                f"{master_split}."
            )

        row_idx = df["row_idx_in_chunk"].to_numpy(copy=False)
        y = df["y"].to_numpy(copy=False)
        group_id = df["group_id"].to_numpy(copy=False)
        split_id = df["split_id"].to_numpy(copy=False)

        save_chunk(
            output_dir=output_dir,
            chunk_id=cid,
            row_idx=row_idx,
            y=y,
            group_id=group_id,
            split_id=split_id,
            force=force,
        )

        n = len(df)

        n_pos = int(np.count_nonzero(y == 1))
        n_neg = n - n_pos

        n_train = int(np.count_nonzero(split_id == 0))
        n_val = int(np.count_nonzero(split_id == 1))
        n_test = int(np.count_nonzero(split_id == 2))

        processed_chunks += 1
        processed_rows += n
        seen_chunks.add(cid)

        if (
            processed_chunks <= 5
            or processed_chunks % 100 == 0
        ):
            print(
                f"[{processed_chunks:>5}] "
                f"chunk={cid:06d} | "
                f"rows={n:,} | "
                f"pos={n_pos:,} | "
                f"neg={n_neg:,} | "
                f"train={n_train:,} | "
                f"val={n_val:,} | "
                f"test={n_test:,} | "
                f"total={processed_rows:,}"
            )

        pending_chunk_id = None
        pending_parts = []

    # master_split is expected to be ordered by chunk_id.
    #
    # We cannot assume that a pandas CSV block ends exactly at a
    # physical chunk boundary, so the current chunk is retained until
    # the next chunk_id is observed.
    for block in reader:

        chunk_ids = block["chunk_id"].to_numpy()

        if len(chunk_ids) == 0:
            continue

        # Locations at which chunk_id changes.
        boundaries = np.flatnonzero(
            chunk_ids[1:] != chunk_ids[:-1]
        ) + 1

        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(block)]))

        for start, end in zip(starts, ends):

            part = block.iloc[start:end]
            cid = int(part["chunk_id"].iloc[0])

            if pending_chunk_id is None:
                pending_chunk_id = cid
                pending_parts = [part]

            elif cid == pending_chunk_id:
                # Same physical chunk continued across CSV blocks.
                pending_parts.append(part)

            else:
                # We have observed the next physical chunk, therefore
                # the pending one is complete.
                flush_pending()

                pending_chunk_id = cid
                pending_parts = [part]

    # Final physical chunk.
    flush_pending()

    print("-" * 80)
    print(
        f"COMPLETED: {dataset_dir.name}\n"
        f"  chunks : {processed_chunks:,}\n"
        f"  rows   : {processed_rows:,}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate per-chunk NPY metadata from master_split.csv."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing the datasets.",
    )

    parser.add_argument(
        "--csv-read-rows",
        type=int,
        default=CSV_READ_ROWS,
        help=(
            "Number of CSV rows read by pandas at once. "
            "This affects only preparation memory usage."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk metadata.",
    )

    args = parser.parse_args()

    root = args.root.resolve()

    print(f"Searching datasets under: {root}")

    master_splits = discover_master_splits(root)

    print(
        f"Found {len(master_splits)} master_split.csv files."
    )

    for i, master_split in enumerate(master_splits, start=1):

        print()
        print(
            f"DATASET {i}/{len(master_splits)}"
        )

        process_master_split(
            master_split=master_split,
            csv_read_rows=args.csv_read_rows,
            force=args.force,
        )

    print()
    print("=" * 80)
    print("ALL DATASETS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()