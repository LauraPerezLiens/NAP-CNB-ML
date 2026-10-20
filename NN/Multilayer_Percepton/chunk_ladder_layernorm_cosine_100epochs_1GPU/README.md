# LayerNorm chunk ladder

100 epochs per stage, seed 42, primary_secondary features, one GPU (physical GPU 2).
Stages: 20, 30, 40, ... 14500, 14503 physical chunks, matching the existing ladder limit.
`chunks_10` contains the completed original LayerNorm experiment, moved here without copying data.
Each new stage starts from the preceding stage's BEST weights and resets the
optimizer and scheduler. Interrupted stages resume from their checkpoint.
The fitted standardizer is copied from the original run and shared across stages.

This compares cumulative training with increasing data and compute, not independent
models trained from scratch. Evaluation populations also grow with the selected
chunks; metrics across stages are not measured on a common fixed test population.
BEST currently uses TEST loss, so these are exploratory results.

Start/restart with `bash ../run_chunk_ladder_layernorm.sh` from this directory,
or invoke the launcher using its absolute path. Per-stage logs and histories are
in chunks_N; ladder.log contains combined output.
