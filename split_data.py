#!/usr/bin/env python3
"""Create reproducible random row splits, stratified by label within each chunk.

Neither group_id nor protein_group_id constrains the split. Their values and all
source rows are preserved. The same group may occur in multiple splits.
Ratios default to TRAIN/VALIDATION/TEST = 80/10/10 by rows of each class.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

TRAIN, VALIDATION, TEST = 0, 1, 2


def split_counts(n: int, train: float, validation: float) -> tuple[int, int, int]:
    n_train = min(round(n * train), n)
    n_validation = min(round(n * validation), n - n_train)
    return n_train, n_validation, n - n_train - n_validation


def validate_ratios(train: float, validation: float, test: float) -> None:
    ratios = np.array([train, validation, test])
    if not np.isfinite(ratios).all() or (ratios < 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError('Split ratios must be finite, nonnegative, and sum to 1')


def iter_physical_chunks(path: Path, read_rows: int):
    """Yield complete consecutive physical chunks regardless of CSV block size."""
    current = None
    parts = []
    seen = set()
    for block in pd.read_csv(path, chunksize=read_rows):
        if block.empty:
            continue
        if 'chunk_id' not in block:
            raise ValueError(f'{path}: missing chunk_id')
        values = block['chunk_id'].to_numpy()
        if block['chunk_id'].isna().any() or not np.all(values == values.astype(np.int64)) or (values < 0).any():
            raise ValueError('chunk_id must contain nonnegative integers')
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            cid = int(values[start])
            if cid != current:
                if parts:
                    yield current, pd.concat(parts, ignore_index=True)
                if cid in seen:
                    raise ValueError(f'Chunk {cid} appears discontinuously')
                seen.add(cid)
                current, parts = cid, []
            parts.append(block.iloc[start:end])
    if parts:
        yield current, pd.concat(parts, ignore_index=True)


def assign_row_splits(labels, seed: int, chunk_id: int,
                      train: float = .8, validation: float = .1,
                      test: float = .1) -> np.ndarray:
    """Shuffle row positions per class; group identifiers are not inputs."""
    validate_ratios(train, validation, test)
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Labels must be 0 or 1')
    rng = np.random.default_rng(np.random.SeedSequence([seed, chunk_id]))
    result = np.empty(len(labels), dtype=np.uint8)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_train, n_validation, _ = split_counts(len(indices), train, validation)
        result[indices[:n_train]] = TRAIN
        result[indices[n_train:n_train + n_validation]] = VALIDATION
        result[indices[n_train + n_validation:]] = TEST
    return result


def process_dataset(master_index: Path, train_ratio: float = .8,
                    validation_ratio: float = .1, test_ratio: float = .1,
                    seed: int = 42, read_rows: int = 500_000) -> None:
    validate_ratios(train_ratio, validation_ratio, test_ratio)
    master_index = Path(master_index)
    header = pd.read_csv(master_index, nrows=0)
    required = {'chunk_id', 'row_idx_in_chunk', 'y', 'group_id', 'protein_group_id'}
    if not required.issubset(header.columns):
        raise ValueError(f'{master_index}: missing {sorted(required - set(header.columns))}')
    if 'split_id' in header:
        raise ValueError('master_index must not already contain split_id')
    output = master_index.with_name('master_split.csv')
    counts = np.zeros((3, 2), dtype=np.int64)
    chunks = rows = 0
    # Replace the CSV only after a complete successful write.
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.master_split_', suffix='.csv',
                                     mode='w', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        for cid, frame in iter_physical_chunks(master_index, read_rows):
            if not np.array_equal(frame['row_idx_in_chunk'].to_numpy(), np.arange(len(frame))):
                raise ValueError(f'Chunk {cid}: row indices must be consecutive from zero')
            frame['split_id'] = assign_row_splits(frame['y'].to_numpy(), seed, cid,
                                                  train_ratio, validation_ratio, test_ratio)
            frame.to_csv(temporary, mode='a', index=False, header=chunks == 0)
            for (split, label), n in frame.groupby(['split_id', 'y']).size().items():
                counts[int(split), int(label)] += int(n)
            chunks += 1
            rows += len(frame)
            if chunks <= 5 or chunks % 100 == 0:
                print(f'{master_index.parent}: chunks={chunks} rows={rows}', flush=True)
        if not rows:
            raise ValueError(f'{master_index}: no rows')
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'WROTE {output}: chunks={chunks}, rows={rows}, counts[split][class]={counts.tolist()}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train', type=float, default=.8)
    parser.add_argument('--validation', type=float, default=.1)
    parser.add_argument('--test', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--read-rows', type=int, default=500_000)
    parser.add_argument('--monitor-interval', type=int, default=0,
                        help='Legacy compatibility option; no background monitor is started.')
    args = parser.parse_args()
    if args.read_rows <= 0 or args.seed < 0:
        parser.error('--read-rows must be positive and --seed nonnegative')
    validate_ratios(args.train, args.validation, args.test)
    files = sorted(args.root.rglob('master_index.csv'))
    if not files:
        parser.error(f'No master_index.csv found under {args.root}')
    for path in files:
        process_dataset(path, args.train, args.validation, args.test, args.seed, args.read_rows)


if __name__ == '__main__':
    main()
