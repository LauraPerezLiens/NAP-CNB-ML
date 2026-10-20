# MLP normalization experiments

All three runs use primary + secondary embeddings, seed 42, the same 10 selected
physical chunks, 100 requested epochs, AdamW and CosineAnnealingWarmRestarts
(T_0=20, T_mult=1, eta_min=1e-5). The baseline output folder contains `10epoch`
for historical naming reasons; its actual target is 100 epochs.

## Metrics and sampling

- TRAIN ONLINE: 50/50 batches with repetition, dropout enabled, weights updated.
- TRAIN EVAL and TEST EVAL: fixed global 50/50 populations without replacement,
  evaluated at epoch end with dropout disabled and weights frozen.
- ROC-AUC is exact, calculated by scikit-learn from labels and raw logits.
- ONLINE balances within chunks; EVAL balances globally. These procedures can
  assign different weights to individual chunks despite both being 50/50.
- The BatchNorm diagnostic additionally evaluates the same weights, rows and
  batches using batch statistics on a deep copy, with dropout disabled.
- LayerNorm replaces both BatchNorm layers and normalizes each sample independently.

## Reproducing the runs

Run from the repository root with a Python environment providing PyTorch, NumPy
and scikit-learn. `nap_env.yml` records an available environment export; adapt
hardware-specific packages and local paths to your system. Input embeddings and
split metadata are external to this repository; pass their location via `--root`.
The small fitted standardizers are included alongside the experiment artifacts.

### BatchNorm baseline

```bash
CUDA_VISIBLE_DEVICES=0 python NN/Multilayer_Percepton/mlp_baseline.py \
  --root /path/to/data_input_nn --features primary_secondary \
  --epochs 100 --max-chunks 10 --report-every-chunks 10 --seed 42 \
  --output-dir NN/Multilayer_Percepton/chunk_lader_cosine_10epoch_trainytesteval
```

### BatchNorm diagnostic

```bash
CUDA_VISIBLE_DEVICES=1 python NN/Multilayer_Percepton/mlp_baseline_bn_diagnostic.py \
  --root /path/to/data_input_nn --features primary_secondary \
  --epochs 100 --max-chunks 10 --report-every-chunks 10 --seed 42 \
  --output-dir NN/Multilayer_Percepton/chunk_lader_cosine_100epoch_bn_diagnostic
```

### LayerNorm

```bash
CUDA_VISIBLE_DEVICES=2 python NN/Multilayer_Percepton/mlp_baseline_layernorm.py \
  --root /path/to/data_input_nn --features primary_secondary \
  --epochs 100 --max-chunks 10 --report-every-chunks 10 --seed 42 \
  --output-dir NN/Multilayer_Percepton/chunk_lader_cosine_100epoch_layernorm
```

## Result snapshot

Committed histories are a snapshot of experiments that may still be running.
The table uses the last complete epoch available when this note was generated;
compare matched epochs in the histories when comparing architectures.

| Run | Completed epoch in snapshot | TEST accuracy (running/default stats) | TEST AUC | TEST accuracy (BN batch stats) |
| --- | ---: | ---: | ---: | ---: |
| BatchNorm baseline | 95 | 0.7735 | 0.9856 | — |
| BatchNorm diagnostic | 60 | 0.7695 | 0.9875 | 0.9837 |
| LayerNorm | 27 | 0.8387 | 0.9892 | — |

## Interpretation and limits

The diagnostic isolates a substantial effect of BatchNorm running statistics;
batch-stat evaluation is diagnostic and depends on batch composition. It is not
a replacement for a fixed inference protocol. TRAIN and TEST evaluation results
being close does not by itself prove independence or rule out data leakage.

The current training scripts select BEST using TEST loss. These TEST results are
therefore exploratory; a final comparison should select checkpoints and tune
hyperparameters on VALIDATION and reserve an untouched TEST for final reporting.

Large checkpoints, input datasets, raw logs and process IDs are not included.
Other tracked historical results are preserved even if absent in this workspace.
