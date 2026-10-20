#!/usr/bin/env python3
"""Read-only validation of random row splits and optional training metadata.

Group/protein overlap between splits is allowed. Check original rows, per-class
80/10/10 counts within physical chunks, and agreement with training metadata.
"""
from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from split_data import split_counts, validate_ratios

REQUIRED = ['chunk_id', 'row_idx_in_chunk', 'y', 'group_id']


def validate_dataset(master_split: Path, read_rows: int = 500_000,
                     check_metadata: bool = True,
                     ratios: tuple[float, float, float] = (.8, .1, .1)) -> dict:
    """Raise ValueError on the first violation; never modify input data."""
    validate_ratios(*ratios)
    master_split = Path(master_split)
    master_index = master_split.with_name('master_index.csv')
    index_columns = list(pd.read_csv(master_index, nrows=0).columns)
    split_columns = list(pd.read_csv(master_split, nrows=0).columns)
    if not set(REQUIRED).issubset(index_columns):
        raise ValueError(f'{master_index}: missing required columns')
    if not set(index_columns + ['split_id']).issubset(split_columns):
        raise ValueError(f'{master_split}: missing source columns or split_id')
    metadata_dir = master_split.parent / 'chunk_metadata'
    verify_arrays = check_metadata and metadata_dir.is_dir()
    readers = [pd.read_csv(path, chunksize=read_rows)
               for path in (master_index, master_split)]
    counts = np.zeros((3, 2), dtype=np.int64)
    rows = 0
    seen_chunks = set()
    current_chunk = None
    expected_row = 0
    arrays = None
    chunk_counts = np.zeros((3, 2), dtype=np.int64)

    def check_chunk_ratios():
        for label in (0, 1):
            actual = chunk_counts[:, label]
            expected = np.array(split_counts(int(actual.sum()), ratios[0], ratios[1]))
            # Allow one row of rounding difference for legacy floor/round policies.
            if np.any(np.abs(actual - expected) > 1):
                raise ValueError(f'Chunk {current_chunk}, class {label}: counts {actual.tolist()} '
                                 f'differ from expected {expected.tolist()} beyond rounding tolerance')

    for source, assigned in zip_longest(*readers):
        if source is None or assigned is None or len(source) != len(assigned):
            raise ValueError('master_index/master_split row count mismatch')
        if not source.equals(assigned[index_columns]):
            raise ValueError(f'Source rows differ near global row {rows}')
        for column in REQUIRED + ['split_id']:
            values = assigned[column].to_numpy()
            if assigned[column].isna().any() or not np.all(values == values.astype(np.int64)):
                raise ValueError(f'Invalid integer values in {column}')
        if not assigned['y'].isin([0, 1]).all():
            raise ValueError('Labels must be 0 or 1')
        if not assigned['split_id'].isin([0, 1, 2]).all():
            raise ValueError('split_id must be 0, 1 or 2')
        if (assigned[['group_id', 'chunk_id', 'row_idx_in_chunk']] < 0).any().any():
            raise ValueError('Negative group/chunk/row identifier')
        for (split, label), n in assigned.groupby(['split_id', 'y']).size().items():
            counts[int(split), int(label)] += int(n)
        ids = assigned['chunk_id'].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            part = assigned.iloc[start:end]
            cid = int(ids[start])
            if cid != current_chunk:
                if current_chunk is not None:
                    check_chunk_ratios()
                chunk_counts.fill(0)
                if arrays is not None and expected_row != len(arrays['y']):
                    raise ValueError(f'Chunk {current_chunk}: metadata has extra rows')
                if cid in seen_chunks:
                    raise ValueError(f'Chunk {cid} appears discontinuously')
                seen_chunks.add(cid)
                current_chunk, expected_row, arrays = cid, 0, None
                if verify_arrays:
                    arrays = {key: np.load(metadata_dir / f'chunk_{cid:06d}_{key}.npy', mmap_mode='r')
                              for key in ('y', 'group_id', 'split_id')}
                    if any(a.ndim != 1 or len(a) != len(arrays['y']) for a in arrays.values()):
                        raise ValueError(f'Chunk {cid}: inconsistent metadata shapes')
            for (split, label), n in part.groupby(['split_id', 'y']).size().items():
                chunk_counts[int(split), int(label)] += int(n)
            stop = expected_row + len(part)
            if not np.array_equal(part['row_idx_in_chunk'].to_numpy(), np.arange(expected_row, stop)):
                raise ValueError(f'Chunk {cid}: row indices are not consecutive from zero')
            if arrays is not None:
                for key, array in arrays.items():
                    if not np.array_equal(array[expected_row:stop], part[key].to_numpy()):
                        raise ValueError(f'Chunk {cid}: {key} metadata differs from master_split')
            expected_row = stop
        rows += len(assigned)
    if rows == 0:
        raise ValueError('Empty master files')
    if arrays is not None and expected_row != len(arrays['y']):
        raise ValueError(f'Chunk {current_chunk}: metadata has extra rows')
    check_chunk_ratios()
    result = {'rows': rows, 'chunks': len(seen_chunks), 'row_counts_by_split_class': counts.tolist(),
              'metadata_checked': verify_arrays}
    print(f'PASS {master_split}: {result}', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/nap/lperez_nn/data/data_input_nn'))
    parser.add_argument('--read-rows', type=int, default=500_000)
    parser.add_argument('--train', type=float, default=.8)
    parser.add_argument('--validation', type=float, default=.1)
    parser.add_argument('--test', type=float, default=.1)
    parser.add_argument('--skip-metadata', action='store_true', help='Validate CSV files before generating metadata.')
    args = parser.parse_args()
    if args.read_rows <= 0:
        parser.error('--read-rows must be positive')
    files = sorted(args.root.rglob('master_split.csv'))
    if not files:
        parser.error(f'No master_split.csv found under {args.root}')
    failed = 0
    for path in files:
        try:
            validate_dataset(path, args.read_rows, not args.skip_metadata, (args.train, args.validation, args.test))
        except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
            failed += 1
            print(f'FAIL {path}: {exc}', flush=True)
    print(f'Datasets checked: {len(files)}; failed: {failed}')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
