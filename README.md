# NAP-CNB data preparation and LayerNorm training

## Split policy: random rows, without grouping

TRAIN/VALIDATION/TEST are assigned **randomly by row**, approximately 80/10/10
within each label class of each physical chunk. Neither `group_id` nor
`protein_group_id` constrains assignment. The same group or protein is allowed
in multiple splits; this is intentional, not a validation failure.

The generator shuffles row positions separately for labels 0 and 1 using
`np.random.default_rng(np.random.SeedSequence([seed, chunk_id]))`. Seed defaults
to 42. For each class of size N, TRAIN receives `round(0.8*N)`, VALIDATION
receives `round(0.1*N)` limited to remaining rows, and TEST receives the remainder.
Tiny classes may have an empty split because of rounding. This is random,
label-stratified row assignment, not a group-based split or a global 50/50 split.

Original CSV row order and columns are preserved, and `split_id` is appended:
TRAIN=0, VALIDATION=1, TEST=2. A physical chunk must appear contiguously, with
`row_idx_in_chunk` consecutive from zero. CSV read block size does not change
the assignments. Source row order, labels, chunk IDs and seed must remain fixed
for reproducibility. Group IDs are retained for secondary embedding lookup.

## Retained root-level scripts

| Script | Purpose |
| --- | --- |
| `split_data.py` | Generate random row assignments independently within each physical chunk and label class. Group IDs do not participate. |
| `check_master_split.py` | Read-only checks of source row preservation, valid labels/splits, row indices, per-chunk/class ratios, and existing metadata arrays. Group overlap is allowed. |
| `prepare_chunk_metadata.py` | Convert `master_split.csv` to the compact per-chunk arrays consumed by training. It does not reassign splits. |

The validator allows a one-row rounding difference from the expected counts to
support legacy floor/round conventions. It cannot establish randomness from
counts alone and does not require exact historical RNG assignments. It does not
reject a group appearing in multiple splits or having different row labels.
It verifies `group_id` integrity and CSV/array agreement, without using it as a
splitting unit. This is a row holdout, not a test of unseen groups or proteins.

## Current data and correction of the earlier interpretation

The earlier script cleanup incorrectly interpreted group overlap as an error.
The user clarified that random rows, without grouping, are the intended policy.
That group-based generator/validator interpretation has now been removed.
Observed shared TRAIN/TEST groups in HLA-C chunk 1897 (3619), HLA-B chunk 1849
(3613), and H2-D chunk 757 (3594) are allowed under the requested policy.
They do not require invalidating or restarting the LayerNorm run on that basis.

No existing CSV, embedding, metadata array, or checkpoint was regenerated during
this correction. LayerNorm was left running. A full real-data validation is not
claimed; implementation checks were performed on synthetic fixtures.
The retained `check_master_split.log` is a historical log, not output from the
new validator. The new RNG construction is not claimed to recreate every row
assignment made by older generators.

## Preparation workflow

Use the `nap_env` environment and run from the repository root. **Do not execute
split regeneration while training is reading the same data.** Existing data does
not need regenerating merely to permit groups shared across splits.

```bash
# Only when intentionally creating a NEW split, with all readers stopped:
python split_data.py \
  --root /data/nap/lperez_nn/data/data_input_nn \
  --train 0.80 --validation 0.10 --test 0.10 --seed 42

# Validate the new CSVs before comparing against old metadata.
python check_master_split.py \
  --root /data/nap/lperez_nn/data/data_input_nn --skip-metadata

# Rebuild metadata for the newly generated split.
python prepare_chunk_metadata.py \
  --root /data/nap/lperez_nn/data/data_input_nn --force

# Verify CSVs and generated arrays agree.
python check_master_split.py \
  --root /data/nap/lperez_nn/data/data_input_nn
```

The generator writes a temporary CSV and replaces `master_split.csv` after
successful completion of each dataset. Existing splits are overwritten when
that command is deliberately executed; preserve an external snapshot if needed.
If assignments actually change, begin a separate experiment with fresh model
weights and a standardizer fitted using the new TRAIN split.

The validator streams CSV blocks and checks ratios at physical chunk boundaries.
`--read-rows` controls memory use. It checks metadata automatically when a
`chunk_metadata` directory exists unless `--skip-metadata` is given. Its report
states whether arrays were checked, and failures produce a nonzero exit code.
Custom ratios can be passed with `--train`, `--validation`, and `--test` to both
the generator and validator. It does not audit embedding numerical quality or
cross-modal biological identity. Metadata conversion continues to preserve group
IDs because the training loader uses them to retrieve secondary embeddings.

## Cleanup and checks

The three preparation/validation scripts above and `export_split_assignments.py` are retained at repository root.
The obsolete embedding QC script and distribution analysis script were removed;
existing analysis artifacts were preserved. The metadata converter was retained.

Synthetic checks verified 80/10/10 counts per class/chunk, reproducibility across
CSV read block sizes, acceptance of shared groups across splits, unchanged split
assignments after changing group/protein IDs, metadata conversion, and rejection
of incorrect split counts.

## LayerNorm

See [EXPERIMENTS.md](NN/Multilayer_Percepton/EXPERIMENTS.md) for architecture,
hardware, optimization, sampling, checkpoints, and experimental results.

## Exact assignments used by the recorded experiments

The `split_assignments/` bundle records the **existing training metadata**, not
a newly generated split. The original `master_split.csv` files total tens of GB
and are excluded with `data_input_nn/`. Instead, the bundle stores the split of
every row, addressed by `(dataset path, chunk_id, row_idx_in_chunk)`.

- `manifest.json`: format, source provenance, total coverage, and archive hashes.
- `chunks.csv`: one record per physical chunk, archive location/offset, row count,
  SHA-256 of the decoded split and label arrays, and class counts per split.
- Dataset subdirectories: gzip shards containing two-bit split codes, four rows
  per byte. Each chunk is padded independently; row 0 uses the lowest two bits.

Codes are 0=TRAIN, 1=VALIDATION, 2=TEST. These files identify **each row's actual
assignment**, including the historical random choices that are not guaranteed to
be reproduced by the current generator. They do not contain embeddings, full
source CSV columns, group IDs, or biological sequences. Original source data is
still required to map a row address to its full biological identity.

Example: recover one chunk without modifying the training data:

```python
from pathlib import Path
from export_split_assignments import restore_chunk

splits = restore_chunk(
    Path("split_assignments"),
    "human/mhc-I/HLA-C/classification_human_mhc-I_HLA-C_blosum_ProtT5_chunks",
    1897,
)
print(splits[0])  # Exact split_id for row_idx_in_chunk == 0.
```

`restore_chunk` verifies the decoded array's SHA-256. The exporter also decodes
and compares every chunk before completing the bundle. Source CSV sizes and
modification times are provenance hints, not cryptographic validation of those
large CSV files. The exported assignments come from the NPY metadata actually
read by the current training loader.

To create a separate new snapshot after intentionally changing a split:

```bash
python export_split_assignments.py \
  --root data_input_nn --output split_assignments_new
```

The output directory must not already exist. Keep the corresponding original
input row ordering and labels to reuse an archived split correctly.
