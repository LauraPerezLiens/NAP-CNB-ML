#!/usr/bin/env python3
"""Archive exact existing split assignments without reading embedding matrices.

Export packs four row split IDs per byte (two bits per row), in gzip shards.
The manifest maps dataset/chunk/row position to the original assignments.
No assignment is regenerated and no input file is modified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np


def pack_splits(values):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, [0, 1, 2]).all():
        raise ValueError('Expected a one-dimensional array with split IDs 0, 1, 2')
    padded = np.zeros(((len(values) + 3) // 4) * 4, dtype=np.uint8)
    padded[:len(values)] = values
    return np.bitwise_or.reduce(padded.reshape(-1, 4) << np.array([0, 2, 4, 6], dtype=np.uint8), axis=1)


def unpack_splits(packed, n_rows):
    packed = np.frombuffer(packed, dtype=np.uint8)
    return ((packed[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)[:n_rows]


def export(root: Path, output: Path, chunks_per_shard: int = 1000):
    if chunks_per_shard < 1:
        raise ValueError('chunks_per_shard must be positive')
    inputs = sorted(root.rglob('chunk_metadata/chunk_*_split_id.npy'))
    if not inputs:
        raise ValueError(f'No prepared metadata under {root}')
    output.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for path in inputs:
        dataset = path.parent.parent.relative_to(root).as_posix()
        grouped.setdefault(dataset, []).append(path)
    manifest = []
    sources = []
    shard_hashes = {}
    total_rows = 0
    for dataset, paths in grouped.items():
        dataset_slug = dataset.replace('/', '__')
        source = root / dataset / 'master_split.csv'
        stat = source.stat()
        sources.append({'dataset': dataset, 'path_relative_to_root': f'{dataset}/master_split.csv',
                        'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'note': 'Size/mtime are provenance hints, not a hash or full CSV validation.'})
        for start in range(0, len(paths), chunks_per_shard):
            archive = f'{dataset_slug}/part_{start // chunks_per_shard:04d}.split2.gz'
            target = output / archive
            target.parent.mkdir(parents=True, exist_ok=True)
            offset = 0
            with target.open('wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=6, mtime=0) as stream:
                    for path in paths[start:start + chunks_per_shard]:
                        cid = int(path.name.split('_')[1])
                        splits = np.load(path, mmap_mode='r')
                        labels = np.load(path.with_name(f'chunk_{cid:06d}_y.npy'), mmap_mode='r')
                        if labels.shape != splits.shape or not np.isin(labels, [0, 1]).all():
                            raise ValueError(f'Invalid labels: {path}')
                        packed = pack_splits(splits).tobytes()
                        restored = unpack_splits(packed, len(splits))
                        if not np.array_equal(restored, splits):
                            raise ValueError(f'Round-trip mismatch: {path}')
                        stream.write(packed)
                        counts = np.bincount(splits, minlength=3)
                        positives = np.bincount(splits, weights=labels, minlength=3).astype(np.int64)
                        manifest.append(dict(dataset=dataset, chunk_id=cid, n_rows=len(splits),
                                             archive=archive, uncompressed_offset_bytes=offset,
                                             packed_bytes=len(packed),
                                             split_array_sha256=hashlib.sha256(np.asarray(splits, dtype=np.uint8).tobytes()).hexdigest(),
                                             labels_sha256=hashlib.sha256(np.asarray(labels, dtype=np.uint8).tobytes()).hexdigest(),
                                             **{f'{name}_{kind}': int(val) for name, n, p in zip(('train', 'validation', 'test'), counts, positives)
                                                for kind, val in (('rows', n), ('positive', p), ('negative', n-p))}))
                        offset += len(packed)
                        total_rows += len(splits)
            shard_hashes[archive] = hashlib.sha256(target.read_bytes()).hexdigest()
            print(f'Exported {dataset}, part {start // chunks_per_shard}: {target.stat().st_size} bytes', flush=True)
    with (output / 'chunks.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(manifest)
    metadata = {'format': 'NAP-CNB split2 gzip v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source': 'Existing chunk_metadata used by the training loader; no reassignment.',
                'split_codes': {'0': 'TRAIN', '1': 'VALIDATION', '2': 'TEST'},
                'encoding': 'Four consecutive row IDs per byte, least-significant two bits first; each chunk padded independently.',
                'datasets': len(grouped), 'chunks': len(manifest), 'rows': total_rows,
                'source_csv_provenance': sources, 'archive_sha256': shard_hashes,
                'validation': 'Each exported array decoded and compared exactly with source metadata; CSV contents not fully audited.',
                'historical_rng': 'Unknown; current split_data.py is not asserted to regenerate these historical assignments.'}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'COMPLETE: {len(manifest)} chunks, {total_rows} rows, {len(grouped)} datasets', flush=True)


def restore_chunk(bundle: Path, dataset: str, chunk_id: int):
    with (bundle / 'chunks.csv').open(newline='') as handle:
        record = next((r for r in csv.DictReader(handle)
                       if r['dataset'] == dataset and int(r['chunk_id']) == chunk_id), None)
    if record is None:
        raise KeyError((dataset, chunk_id))
    with gzip.open(bundle / record['archive'], 'rb') as stream:
        stream.seek(int(record['uncompressed_offset_bytes']))
        packed = stream.read(int(record['packed_bytes']))
    if len(packed) != int(record['packed_bytes']):
        raise ValueError('Truncated archive')
    splits = unpack_splits(packed, int(record['n_rows']))
    if hashlib.sha256(splits.tobytes()).hexdigest() != record['split_array_sha256']:
        raise ValueError('Assignment checksum mismatch')
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data_input_nn'))
    parser.add_argument('--output', type=Path, default=Path('split_assignments'))
    parser.add_argument('--chunks-per-shard', type=int, default=1000)
    args = parser.parse_args()
    export(args.root, args.output, args.chunks_per_shard)


if __name__ == '__main__':
    main()
