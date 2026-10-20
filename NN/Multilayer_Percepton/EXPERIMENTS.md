# Technical record of the LayerNorm MLP experiments

This document describes the **current** implementation, the experiments performed,
and the limits of their interpretation. The reference implementation is
`mlp_baseline_layernorm.py`, together with `dataset.py` and
`run_chunk_ladder_layernorm.sh`. This document does not change the running experiment.

## 1. Objective and scope

The model performs binary classification from precomputed embeddings.
It receives one feature row and returns a logit for its 0/1 label.
The specific biological meaning of these labels comes from previously prepared
metadata; the MLP neither redefines labels nor generates embeddings.

The experiment studies performance as the number of available physical chunks
increases: 10, 20, 30, 40, …, 14500, and finally 14503. Each stage requests 100 epochs.
The first LayerNorm stage was trained from scratch; subsequent stages inherit
the weights of the best checkpoint from the preceding stage.

This is therefore a **cumulative learning** ladder, with increasing amounts of
data and computation across stages. It is not equivalent to training independent
models from scratch.

## 2. Pipeline files

| File or directory | Purpose |
| --- | --- |
| `mlp_baseline_layernorm.py` | Architecture, training, evaluation, metrics, and checkpoints. |
| `dataset.py` | Dataset discovery, embedding/metadata loading, secondary embedding lookup, sampling, and standardization. |
| `run_chunk_ladder_layernorm.sh` | Sequential stage launcher using one GPU. |
| `chunk_ladder_layernorm_cosine_100epochs_1GPU/` | Outputs from all LayerNorm stages. |
| `…/standardizer_primary_secondary_2chunks.npz` | Input statistics shared across the ladder. |
| `…/chunks_N/` | Checkpoints, history, and log for the stage using N chunks. |

Metadata and split preparation scripts are outside this directory.
Input embeddings and metadata are not stored inside checkpoints.

## 3. Hardware, GPUs, and environment

Observed server: `pascal`. It has four NVIDIA GeForce RTX 3090 cards, each with
24576 MiB (24 GiB). The observed physical indices are 0, 1, 2, and 3.
All four GPUs have not been used simultaneously for a single run of this ladder.

| Historical experiment | Physical GPU used | Status |
| --- | --- | --- |
| Initial BatchNorm baseline, 10 chunks / 100 epochs | 0 | Completed; directory subsequently deleted. |
| BatchNorm diagnostic, 10 chunks / 100 epochs | 1 | Completed; directory and script subsequently deleted. |
| Initial LayerNorm run, 10 chunks / 100 epochs | 2 | Completed; retained in `chunks_10`. |
| LayerNorm ladder, starting at 20 chunks | 2 | One stage at a time, one training process. |
| BatchNorm ladder with `…1GPU_prints` output | One GPU; insufficient evidence is retained here to establish its physical index | Stopped at the user's request during the 20-chunk stage and deleted locally. |

GPU 2 has UUID `GPU-a32f11f3-3561-4d1a-c364-20ebd84386c7`.
Indices may change if the server is reconfigured; the UUID identifies the card
observed during this session.

The ladder exports:

```bash
CUDA_VISIBLE_DEVICES=2
WORLD_SIZE=1
RANK=0
LOCAL_RANK=0
```

With this visibility setting, PyTorch refers to **physical GPU 2** as `cuda:0`.
This does not mean that physical GPU 0 is being used. One GPU does not imply one
CPU or one thread: this launcher sets neither CPU affinity nor a thread limit.

Python interpreter used:
`/home/nap/miniconda3/envs/nap_env/bin/python`.

| Component | Observed version |
| --- | --- |
| python | 3.10.13 |
| torch | 2.14.0+cu130 |
| cuda_runtime | 13.0 |
| numpy | 2.0.1 |
| pandas | 2.3.3 |
| sklearn | 1.7.2 |

These versions were inspected when the document was written; checkpoints do not
contain a complete environment export. The CUDA version shown is the PyTorch
runtime version, not a statement about the `nvcc` compiler version.

The code includes a DDP/NCCL path for `WORLD_SIZE>1`, but the current ladder uses
`WORLD_SIZE=1`, direct Python execution, and no distribution across multiple GPUs.
If CUDA is unavailable, the Python script's single-device path falls back to CPU;
the launcher does not impose an additional check to prevent this fallback.

## 4. Data, samples, batches, and chunks

Default data root:
`/data/nap/lperez_nn/data/data_input_nn`.
The recorded runs discovered **11 datasets and 14503 physical chunks** in total.

- **Sample:** one input row containing its features and binary label.
- **Batch:** a set of rows processed in one forward pass; 8192 by default.
- **Chunk:** a physical file/block containing many rows, not an individual sample.
- **Stage:** 100 epochs with a selected limit on physical chunks.
- **Epoch:** a pass through the selected chunks using the TRAIN sampling described
  below, followed by two complete evaluations over the fixed populations.

The current configuration uses `primary_secondary`:

| Modality | Dimension | Access |
| --- | ---: | --- |
| Primary / ProtT5 | 1024 | Primary embedding rows from the chunk. |
| Secondary / SecondaryBERT | 768 | Retrieved using the corresponding `group_id`. |
| Concatenated vector | 1792 | MLP input, `float32`. |

Embeddings are precomputed and **are not updated** during MLP training.
The code supports `primary` (1024 features, without this standardizer), but this
mode is not used in these experiments.

For each chunk, the prepared metadata includes at least:

```text
chunk_metadata/
    chunk_XXXXXX_y.npy
    chunk_XXXXXX_group_id.npy
    chunk_XXXXXX_split_id.npy
```

Positions in these arrays correspond to primary embedding rows.
`split_id` distinguishes `TRAIN=0`, `VALIDATION=1`, and `TEST=2`.
The same physical chunks supply TRAIN and TEST rows, filtered by this ID.
TEST rows are not used for weight updates. The current script **does not run
VALIDATION**. It does not read `master_split.csv` directly; it relies on prior
metadata preparation.

Reading uses memory-mapped files and loads the rows needed for each batch.
Secondary embeddings use an LRU cache configured with size 2 per store.
The code temporarily sorts indices when loading rows and restores the requested
order; it does not need to load all embeddings onto the GPU.

## 5. Reproducible chunk selection

`build_chunk_jobs` constructs `(dataset_index, chunk_id)` pairs.
`select_fixed_jobs` permutes the list using:

```python
np.random.default_rng(np.random.SeedSequence([42, 111111]))
```

Each stage takes the first N elements. Provided the files, discovery procedure,
and ordering remain unchanged, the selections are nested:
`chunks_10 ⊂ chunks_20 ⊂ chunks_30 …`.

**Chunk order does not change between epochs:** `ordered_epoch_jobs` returns
`list(fixed_jobs)` and does not use its `seed` or `epoch` arguments to reorder it.
Sample permutations within training do change with the epoch.

Selection reconstructed from the current data:

| Position | Dataset | Physical chunk |
| ---: | --- | ---: |
| 1 | human/mhc-I/HLA-C | 1897 |
| 2 | human/mhc-I/HLA-B | 1849 |
| 3 | mouse/mhc-I/H2-D | 757 |
| 4 | human/mhc-I/HLA-A | 1636 |
| 5 | human/mhc-II/HLA-DP | 1186 |
| 6 | mouse/mhc-I/H2-K | 1387 |
| 7 | human/mhc-I/HLA-B | 1081 |
| 8 | human/mhc-I/HLA-A | 1960 |
| 9 | mouse/mhc-I/H2-D | 345 |
| 10 | human/mhc-I/HLA-C | 1721 |
| 11 | mouse/mhc-I/H2-D | 14 |
| 12 | human/mhc-II/HLA-DP | 606 |
| 13 | human/mhc-II/HLA-DP | 646 |
| 14 | human/mhc-II/HLA-DP | 268 |
| 15 | human/mhc-I/HLA-C | 743 |
| 16 | human/mhc-I/HLA-B | 856 |
| 17 | human/mhc-II/HLA-DQ | 392 |
| 18 | human/mhc-I/HLA-B | 2307 |
| 19 | mouse/mhc-I/H2-K | 1536 |
| 20 | human/mhc-I/HLA-A | 1440 |

Positions 1–10 define the first stage and 1–20 define the second.
This list does not replace the metadata: future changes in dataset discovery
could change the selection even with the same seed.

## 6. Fixed input standardization

Before the MLP, each feature is transformed as follows:

```text
x_normalized[j] = (x[j] - TRAIN_mean[j]) / TRAIN_scale[j]
```

Means and standard deviations are computed separately for the 1024 primary and
768 secondary dimensions. Fitting uses all TRAIN rows from exactly **two physical
chunks**, chosen from candidates containing TRAIN rows, with seed 42.
Neither TEST labels nor VALIDATION rows are used to estimate these statistics.

Variance uses divisor N; means and moments are accumulated in `float64`, and the
final parameters are stored in `float32`. Scales below `1e-12` are replaced by 1.
Batches used to estimate statistics have size 8192.

The retained standardizer identifies these fitting chunks:

| Dataset | Chunk |
| --- | ---: |
| human/mhc-I/HLA-A | 1294 |
| human/mhc-II/HLA-DR | 1846 |

**Standardizer selection is independent of each stage's chunk limit.** Its two
chunks are not among the first 10/20 chunks listed above. Thus, “training with
10 chunks” refers to MLP weight fitting; preprocessing additionally accesses
TRAIN rows from these two chunks.

The same `.npz` is reused throughout the ladder, without refitting per epoch or
when moving from 10 to 20 chunks. It was copied from the initial experiment to
the shared directory. The script searches for it in the **parent of
`--output-dir`**. `--refit-scaler` forces refitting and is not used by the current
launcher. Loading verifies that two chunks are recorded but does not exhaustively
compare their provenance with the current data. Retaining the original file is
necessary.

Input standardization and LayerNorm are separate operations: the former uses
fixed parameters fitted on TRAIN; the latter operates inside the network on each
sample's activations.

## 7. MLP architecture

```text
Input [B, 1792]
  → Linear(1792, 512)
  → LayerNorm(512)
  → ReLU
  → Dropout(p=0.30)
  → Linear(512, 128)
  → LayerNorm(128)
  → ReLU
  → Dropout(p=0.30)
  → Linear(128, 1)
  → squeeze(1)
Output: logits [B]
```

Trainable parameters in `primary_secondary` mode: **985089**.
Linear layers include bias. LayerNorm learns scale and shift, with default
`eps=1e-5` in the observed version.

Each LayerNorm computes mean and variance over the 512 or 128 activations within
**one row**, without combining batch rows. It has no running means/variances,
and its normalization rule is the same in `train()` and `eval()`.

Dropout randomly removes 30% of activations during training, with PyTorch's
corresponding scaling. It is disabled during evaluation.
ReLU layers run with `inplace=True`.

There is no BatchNorm, convolution, attention, residual connection, or explicit
sigmoid in the forward pass. The current loop does not use AMP/autocast,
GradScaler, gradient clipping, or gradient accumulation; computation uses FP32.

## 8. Loss function and weight updates

The loss is `nn.BCEWithLogitsLoss()` with mean reduction and no `pos_weight` or
per-sample weights. Class balance is obtained through sampling.

For logit z and label y, the binary loss can be expressed as:

```text
L(z, y) = max(z, 0) - z*y + log(1 + exp(-abs(z)))
```

The function combines sigmoid and cross-entropy in a numerically stable form.
The minibatch step is:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(X_t)
loss = criterion(logits, y_t)
loss.backward()
optimizer.step()
```

ONLINE metrics use logits from this forward pass, obtained **before the weight
update**, even though they are accumulated after `optimizer.step()`.

Evaluation uses `binary_cross_entropy_with_logits(..., reduction="sum")`;
sums are accumulated and divided by the total number of samples. This is the
same per-sample loss, aggregated to avoid giving equally weighted batch means
to batches of different sizes.

## 9. AdamW and cosine annealing with warm restarts

| Parameter | Effective ladder value |
| --- | ---: |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Betas | (0.9, 0.999), default |
| AdamW epsilon | 1e-8, default |
| Scheduler | CosineAnnealingWarmRestarts |
| T_0 | 20 epochs |
| T_mult | 1 |
| eta_min | 0.00001 |
| Epochs per stage | 100 |
| Early stopping | Disabled |

The learning rate is fixed within each epoch. After both evaluations,
`scheduler.step(epoch + 1)` prepares the next epoch. The log distinguishes
`current_lr` (used in the completed epoch) from `next_lr` (next epoch).

With `T_mult=1`, cycles retain a duration of 20 epochs. Epochs 1, 21, 41, 61, and
81 start with LR 0.001. Epochs 20, 40, 60, 80, and 100 use approximately
0.0000160943. Because updates occur only at integer epoch boundaries, this is
the last value before a restart: `eta_min` itself is not observed as an epoch's
learning rate. A final `next_lr=0.001` at epoch 100 does not imply an epoch 101.

Scheduler restarts change the LR; **they do not reinitialize weights or AdamW
moments within a stage**. Starting a new stage through `--init-checkpoint`,
however, creates a new optimizer and scheduler.

## 10. TRAIN ONLINE sampling

Batches contain 8192 samples: **4096 negatives and 4096 positives**.
The batch size must be positive and even.

For a chunk containing both classes:

1. Only its TRAIN rows are retrieved and separated by label.
2. Each class has a `CyclingClassSampler`, which permutes its rows and recycles
   them when exhausted.
3. `ceil(max(n_negatives, n_positives) / 4096)` complete batches are generated.
4. Indices from both classes are shuffled within each batch.

This covers the chunk's TRAIN rows and allows repetition, particularly for the
minority class. The last batch is also complete and may repeat rows to fill its
quotas. ONLINE counts **sample presentations**, not unique examples.

For a chunk containing only one class, the code deterministically searches for
another selected chunk with TRAIN rows of the missing class. Each batch combines
half its samples from the source chunk and half from the donor. The donor may
belong to another dataset. TEST rows are never used as donors. If no valid donor
exists, an error is raised. A chunk with no TRAIN rows produces no batches.

Permutation seeds include the global seed, epoch, dataset, and chunk; donor
selection additionally incorporates the missing class and a specific salt.

## 11. TRAIN EVAL and TEST EVAL

Before the first epoch, two sampling plans are constructed, one per split:

- **All positive samples** from the split in the selected chunks are retained.
- An equal number of negatives is selected globally, without replacement.
- Rows are grouped by chunk and reproducibly shuffled within each chunk.
- These rows are reused throughout the stage's 100 epochs.

The 50/50 balance is **global**, not necessarily per chunk or per batch.
If there are no positives or too few negatives to match them, the code exits with
an explicit error. The last evaluation batch may contain fewer than 8192 samples.

Selection seeds are `[42, split, 987654]`; local ordering seeds include
`[42, split, dataset_index, chunk_id, 123456]`.
Plans are rebuilt when N increases: retaining previous chunks does not guarantee
that the exact same subset of negatives is retained.

At the end of each epoch, TRAIN EVAL runs first, followed by TEST EVAL.
Evaluation uses `model.eval()` and `torch.inference_mode()`: dropout is disabled,
with no backpropagation or optimizer update. The model's previous mode is then
restored. TEST is not periodically evaluated within training blocks.

| Characteristic | TRAIN ONLINE | TRAIN EVAL | TEST EVAL |
| --- | --- | --- | --- |
| Distribution | 50/50 per batch | 50/50 globally | 50/50 globally |
| Dropout | Enabled | Disabled | Disabled |
| Internal normalization | LayerNorm per sample | LayerNorm per sample | LayerNorm per sample |
| Weights | Updated | Frozen | Frozen |
| ROC-AUC | Exact | Exact | Exact |
| Timing | During training forward passes | End of epoch | End of epoch |
| Samples | With repetition | Fixed per stage, without repetition | Fixed per stage, without repetition |

ONLINE combines predictions from different versions of the weights and gives
more weight to recycled samples. EVAL uses one final model and global weighting.
Thus, ONLINE and TRAIN EVAL are not identical measurements; EVAL is not expected
to always be worse, since disabling dropout can improve TRAIN EVAL.

## 12. Metrics and classification threshold

Positive prediction: `logit >= 0`, equivalent to sigmoid probability >= 0.5.
There is no threshold search or tuning. TP, TN, FP, and FN are used to compute:

| Metric | Definition |
| --- | --- |
| Accuracy | `(TP + TN) / n` |
| Precision | `TP / (TP + FP)` |
| Recall / sensitivity | `TP / (TP + FN)` |
| Specificity | `TN / (TN + FP)` |
| Balanced accuracy | `(recall + specificity) / 2` |
| F1 | `2TP / (2TP + FP + FN)` |
| F2 | `5TP / (5TP + FP + 4FN)` |
| MCC | `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` |
| Loss | Sum of individual losses / n |

ROC-AUC is calculated **over all accumulated labels and logits**, rather than
averaging batch or chunk AUC values:

```python
roc_auc_score(y_true, logits)
```

It uses neither histograms nor probability rounding. With no samples or only
one class, it returns NaN. Ratios with a zero denominator also return NaN.
With exact 50/50 balance, accuracy and balanced accuracy coincide.
PR-AUC/average precision is not computed in this pipeline.

Labels and logits are stored in CPU RAM for exact AUC computation.
Memory grows with the number of sample presentations in the epoch, including
repeated ONLINE samples. “Streaming” in the class name does not imply constant
memory for AUC. Concatenation and sorting also require additional memory.
With DDP, arrays are gathered across ranks; that path is not currently used.

The log prints loss to 6 decimal places and other metrics to 4. A printed
`1.0000` may be a rounded value below 1. The JSONL history preserves greater
precision and also includes TP/TN/FP/FN.

## 13. Epoch workflow and recovery points

1. Visit chunks in their fixed order, in blocks of 10 global chunks.
2. Train each block and merge its ONLINE metrics into the epoch totals.
3. If chunks remain, save `recovery` at the block boundary.
4. After all chunks are processed, evaluate TRAIN and TEST using their fixed plans.
5. Compare TEST loss with the previous best value.
6. Prepare the LR for the next epoch.
7. Save BEST if improved; save LAST and the end-of-epoch RECOVERY checkpoint.
8. Append a history entry and run garbage collection / clear the CUDA cache.
   Clearing the cache does not delete weights or free tensors still in use.

With 10 chunks, there is no intermediate block checkpoint: saving occurs at epoch
end. With 20, recovery is saved after the first 10 chunks and at the end.
Checkpoints are not saved after every minibatch. The main report is printed per
epoch, not per block, despite what the name `report-every-chunks` might suggest.

## 14. BEST selection and checkpoint contents

BEST is selected when TEST loss is finite and strictly lower than the previous
best. Selection is not based on accuracy or AUC, and training does not stop before
100 epochs. If no epoch provides a finite TEST loss, the script fails.

Files per stage:

```text
best_mlp_primary_secondary.pt
last_mlp_primary_secondary.pt
recovery_mlp_primary_secondary.pt
history_mlp_primary_secondary.jsonl
run.log
completed_100_epochs.ok         # Created by the launcher when a new stage finishes.
```

Checkpoints contain version 4, `normalization="layernorm"`, weights, AdamW state,
scheduler state, features, input dimension, dropout, seed, position within the
epoch, processed chunks, completed epochs, best epoch, arguments, and, where
applicable, partial ONLINE metrics including labels/logits.

The historical field **`best_val_loss` actually stores the best TEST loss**.
Its name does not mean that VALIDATION is being evaluated.
The standardizer is stored in an external `.npz`, not inside the `.pt` file.
`epochs_without_improvement` exists for compatibility and does not enable early
stopping. Checkpoints do not store all embeddings or split arrays.

`--init-checkpoint`: loads weights, starts a stage at epoch 1, and resets AdamW,
the scheduler, and best loss. It checks features, input dimension, and dropout.

`--resume`: restores weights, AdamW, scheduler, best loss, and position. It also
requires the same seed. For an intermediate checkpoint, it restores accumulated
metrics and continues from the next block. `--epochs` is the total target, not
an additional number of epochs. `--resume` and `--init-checkpoint` are mutually
exclusive.

**Actual reproducibility limit:** Python, NumPy, and PyTorch are seeded with
`seed + rank`, but their RNG states are not saved/restored in current LayerNorm
checkpoints. After resuming, the dropout sequence may differ from an uninterrupted
run. Deterministic CUDA algorithms are not explicitly enabled either. Recovery
is operational, but bit-for-bit identity is not guaranteed.
The chunk/metadata set has no fingerprint validated on resume.
Checkpoints are written directly with `torch.save`, without atomic replacement;
an interruption during writing may leave an incomplete file.

## 15. Ladder launcher behavior

`run_chunk_ladder_layernorm.sh` starts at **20 chunks** and requires
`chunks_10/best_mlp_primary_secondary.pt` to exist. The first 10-chunk stage has
already completed and is not automatically retrained.

For each N:

1. Create `chunks_N`.
2. If a completion marker exists, check BEST and LAST and skip the stage.
3. Otherwise, prioritize RECOVERY; if absent, try LAST.
4. Read `completed_epochs` from the checkpoint found. If 100 epochs are already
   complete, create the marker and advance to the next stage.
5. If epochs remain, resume; if no checkpoint exists, initialize from the preceding BEST.
6. Run Python and copy its output to `chunks_N/run.log` using `tee -a`.
7. After successful completion, check that BEST/LAST exist, create the marker,
   and advance.

`set -euo pipefail` stops the ladder if a stage fails. `flock` on `ladder.lock`
prevents two instances of this launcher from writing simultaneously.
Stages are not assigned to separate GPUs: all run **sequentially on GPU 2**.
The lock prevents duplicate ladder launchers but does not reserve the GPU against
other external programs.

Only BEST weights are passed to the next stage; they are not necessarily the
weights from epoch 100. For example, the 20-chunk stage starts from the 10-chunk
BEST checkpoint, which corresponds to epoch 98.

## 16. Commands and monitoring

From the repository root:

```bash
bash NN/Multilayer_Percepton/run_chunk_ladder_layernorm.sh
```

Override the GPU, chunk limit, or interpreter while retaining the same output
directory:

```bash
CUDA_VISIBLE_DEVICES=2 FINAL_CHUNKS=100 \
PYTHON=/home/nap/miniconda3/envs/nap_env/bin/python \
bash NN/Multilayer_Percepton/run_chunk_ladder_layernorm.sh
```

Do not start another launcher while the current one is active. Changing these
variables does not automatically create an independent comparison or a separate
directory. The usual limit is 14503, and the increment is fixed at 10.

Equivalent example for a new 20-chunk stage, executed individually **only when
that stage is not already active**:

```bash
CUDA_VISIBLE_DEVICES=2 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
/home/nap/miniconda3/envs/nap_env/bin/python -u \
NN/Multilayer_Percepton/mlp_baseline_layernorm.py \
  --root /data/nap/lperez_nn/data/data_input_nn \
  --features primary_secondary --epochs 100 --seed 42 \
  --batch-size 8192 --eval-batch-size 8192 --dropout 0.30 \
  --learning-rate 0.001 --weight-decay 0.0001 \
  --scheduler-t0 20 --scheduler-tmult 1 --scheduler-min-lr 0.00001 \
  --max-chunks 20 --report-every-chunks 10 \
  --init-checkpoint NN/Multilayer_Percepton/chunk_ladder_layernorm_cosine_100epochs_1GPU/chunks_10/best_mlp_primary_secondary.pt \
  --output-dir NN/Multilayer_Percepton/chunk_ladder_layernorm_cosine_100epochs_1GPU/chunks_20
```

To recover an individual stage, replace `--init-checkpoint` with
`--resume path/to/recovery_mlp_primary_secondary.pt`; do not pass both.
The recommended way to continue the ladder is to run the launcher again.
Do not start an existing stage without `--resume` if its history must be preserved:
the Python script deletes the previous JSONL when starting a new run.

Monitoring:

```bash
tail -f NN/Multilayer_Percepton/chunk_ladder_layernorm_cosine_100epochs_1GPU/ladder.log
tail -f NN/Multilayer_Percepton/chunk_ladder_layernorm_cosine_100epochs_1GPU/chunks_20/run.log
nvidia-smi
```

`run.pid` at the ladder root identifies the launcher started during this session;
it is temporary process information, not a permanent experiment identifier.
`ladder.log` combines stage output; each `chunks_N/run.log` contains that stage's
output. JSONL is the preferred source for numerical analysis.
`train` and `train_online` in JSONL are aliases for the same result, as are `test`
and `test_eval`. They should not be counted as additional experiments.

Potentially misleading option names:

- `--auc-bins`: legacy compatibility option; it does not affect exact AUC.
- `--full-test-every-epoch`: accepted as an argument, but the current loop always
  evaluates TEST at epoch end, even though this flag defaults to false.
- `--monitor-test-chunks`: currently 0; this implementation has no active loop
  for TEST monitoring after each block.

## 17. Decision history and results

Normalization comparisons using 10 chunks / 100 epochs were performed on
October 20, 2026. The BatchNorm diagnostic evaluated the same weights and samples
with running statistics and with batch statistics, disabling dropout and using
a copy to avoid modifying the trained model.

Values at the end of epoch 100, recorded during this session before cleanup:

| TEST mode | Accuracy | Recall | ROC-AUC | Loss |
| --- | ---: | ---: | ---: | ---: |
| BatchNorm, running statistics | 0.769167 | 0.538600 | 0.986282 | 1.880200 |
| BatchNorm, batch statistics (diagnostic) | 0.985450 | 0.983391 | 0.998883 | 0.050603 |
| LayerNorm, standard evaluation | 0.999170 | 0.998538 | 0.999991 | 0.003121 |

The controlled BatchNorm comparison shows an effect of its inference statistics.
The subsequent LayerNorm experiment replaced both normalization layers while
retaining the rest of the training configuration.
The architecture comparison is not a multi-seed experiment and does not by itself
establish absence of leakage or generalization to new datasets.

The user decided to retain only LayerNorm. The BatchNorm ladder was stopped,
and its directories and scripts within this directory were removed locally.
Previously published versions remain in Git history.
The original `chunk_lader_cosine_100epoch_layernorm` directory was **moved**, without
duplicating files, to `chunk_ladder_layernorm_cosine_100epochs_1GPU/chunks_10`.
It is no longer a symbolic link. As a result of this move, historical logs and
arguments may contain the old path; this does not indicate a second copy or
another training run.

### Retained 10-chunk results

| Selection | Epoch | TEST loss | TEST accuracy | TEST AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BEST by TEST loss | 98 | 0.002926016 | 0.999269200 | 0.999989795 | 3 | 19 |
| LAST | 100 | 0.003121437 | 0.999169546 | 0.999991146 | 3 | 22 |

Per-epoch populations: ONLINE 1,400,832 sample presentations; TRAIN EVAL 240,856
rows; TEST EVAL 30,104 rows. Commas in these counts are thousands separators.

### Progress snapshot at the time of writing

| Stage | Last complete epoch observed |
| --- | ---: |
| chunks_10 | 100 |
| chunks_20 | 90 |

This table is a snapshot, not a live status indicator. The ladder may have
advanced while this document was being written. Refer to logs and JSONL for the
current state. No total completion time is promised: work per stage increases,
and there are I/O and exact-AUC computation costs.

## 18. Limits on result interpretation

1. **BEST is selected using TEST loss.** The current TEST participates in model
   selection; its results are exploratory. Final evaluation should select models
   and parameters on VALIDATION and reserve an independent TEST.
2. **TRAIN and TEST share physical chunks.** Their rows are separated by split_id,
   but this run does not check for shared groups/proteins/sequences. Biological
   independence requires auditing split preparation.
3. **TEST is artificially balanced.** Precision, accuracy, and loss may change in
   a population with a different prevalence. Balancing also changes which
   negatives enter evaluation.
4. **The standardizer uses TRAIN outside the small subset.** Account for this when
   describing experiments with few chunks or designing a holdout by chunk.
5. **EVAL populations change between stages.** To isolate the effect of more data,
   evaluate all checkpoints on a common fixed set.
6. **Training is cumulative.** Improvement from N to N+10 may reflect more data,
   more optimization, or both; this is not a controlled comparison from scratch.
7. **Chunks retain their order.** Ordering effects and differences between
   datasets may exist; LayerNorm avoids dependence on batch statistics but does
   not automatically remove all distribution effects.
8. **One seed and specific hardware.** There is no estimate of variation across
   seeds or a bit-for-bit guarantee after recovery.
9. **Nearly perfect results require external checks.** They do not by themselves
   demonstrate generalization to unseen groups, chunks, species, or families.
10. **Exact AUC has a growing cost.** RAM usage and intermediate RECOVERY size grow
    with retained labels/logits; peak consumption for the entire ladder up to
    14503 chunks has not been validated.

## 19. Checks performed for this documentation

Architecture, arguments, launcher behavior, sampling plans, metrics, loss,
scheduler, and checkpoints were checked against the current files.
GPU/UUID information was queried with `nvidia-smi`, along with `nap_env` versions,
standardizer metadata, and retained histories. Previous tests checked LayerNorm
forward/backward execution and independence from batch companions in evaluation
mode, as well as exact AUC, balance, and preservation of weights during EVAL.
No hyperparameter, checkpoint, or training process was changed to write this
document. No new leakage audit or end-to-end multi-GPU validation was performed.

Technical review date: 2026-10-20 15:00 CEST.

## 20. Clarified split policy and preparation scripts

The intended policy, clarified by the user after the initial cleanup, is
**random splitting by rows, without grouping by group_id or protein_group_id**.
Sharing groups between TRAIN, VALIDATION and TEST is allowed. The earlier
interpretation that such overlap violated the desired split was incorrect.
The observed overlaps in HLA-C chunk 1897 (3619 TRAIN/TEST groups), HLA-B chunk
1849 (3613), and H2-D chunk 757 (3594) are compatible with this policy and do not
by themselves invalidate or require restarting the current LayerNorm run.

The retained root-level `split_data.py` now assigns rows randomly within each
physical chunk and label class, approximately 80/10/10, using seed 42 by default.
The RNG is constructed from `[seed, chunk_id]`; group and protein IDs are never
inputs to assignment. Source order and identifiers are preserved. The current
RNG implementation is not claimed to reconstruct every assignment produced by
historical scripts. `check_master_split.py` permits overlap and checks row
integrity, per-class/chunk counts and existing metadata arrays instead.
`prepare_chunk_metadata.py` copies assignments into arrays without changing them.

Synthetic checks confirmed ID independence, acceptance of shared groups,
reproducibility across CSV block sizes, balanced split proportions within classes,
and correct metadata conversion. No real-data split or metadata was regenerated,
and the active LayerNorm process was not interrupted. The repository README
provides the preparation workflow for a future intentional split rebuild.

The distinction remains relevant to interpretation: these experiments measure
performance on held-out rows whose groups/proteins may also appear in training.
They do not establish generalization to entirely unseen groups/proteins. This is
a description of the intended evaluation scope, not a failure of the requested
random-row splitting policy.

## 21. Archived exact row assignments

The repository now provides `split_assignments/` at its root, exported by
`export_split_assignments.py` from the existing `chunk_metadata` arrays. This
archives the actual historical assignments consumed by training without changing
them or assuming that the current generator reproduces the historical RNG.
Each sample is addressed by dataset path, physical chunk ID, and row position.

`chunks.csv` records per-chunk counts, class composition, compressed shard offsets,
and hashes of the split and label arrays. `manifest.json` identifies the format,
coverage, source CSV sizes/timestamps, and compressed archive hashes. Gzip shards
encode four split IDs per byte. Every chunk was decoded and compared to its source
array during export. The bundle contains assignments, not embeddings or complete
biological identifiers; the original inputs remain required for full sample
identity. The repository README gives an exact per-chunk restoration example.

Full `master_split.csv` files are excluded because they total tens of GB. Source
CSV size/timestamp entries are provenance hints, not a full CSV consistency audit.
The exported arrays directly reflect what the training loader reads. Training
was not interrupted or reassigned to produce this archive.
