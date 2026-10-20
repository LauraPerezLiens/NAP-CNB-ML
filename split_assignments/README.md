# Exact split assignments

This snapshot contains the split of every row in the existing training metadata:
**11 datasets, 14503 chunks, 1,449,852,950 rows**.
Total bundle size at export: approximately 195.8 MiB.

See the repository-root README for the encoding, restoration example and scope.
`manifest.json` records format/version and hashes; `chunks.csv` maps row addresses
to gzip shard offsets and includes per-class/split counts.

Rows are indexed from zero within their original physical chunk. Assignments were
copied exactly, not regenerated. Codes: 0=TRAIN, 1=VALIDATION, 2=TEST. Shared groups
are allowed by the intended random-row policy. Original embeddings and identity
metadata are external inputs; the bundle contains neither sequences nor weights.
