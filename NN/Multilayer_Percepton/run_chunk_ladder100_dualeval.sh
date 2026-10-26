#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/nap/miniconda3/envs/nap_env/bin/python}"
RESULTS="${BASE_DIR}/chunk_ladder100_layernorm_cosine_shuffled_allscaler_dualeval_100epochs_1GPU"
SCRIPT="${BASE_DIR}/mlp_layernorm_shuffled_ladder100_dualeval.py"
FEATURES=primary_secondary
EPOCHS=100
STEP=100
FINAL_CHUNKS="${FINAL_CHUNKS:-14503}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MALLOC_ARENA_MAX=2

mkdir -p "$RESULTS"
# Prevent two launchers from writing to the same stage.
exec 9>"${RESULTS}/ladder.lock"
flock -n 9 || { echo 'Another LayerNorm ladder is already running.' >&2; exit 1; }

# Start the 100-chunk stage with random weights. Later stages use this run's best.
PREVIOUS_BEST=""
[[ "$FINAL_CHUNKS" =~ ^[0-9]+$ ]] && (( FINAL_CHUNKS >= 100 )) || {
    echo 'FINAL_CHUNKS must be an integer >= 100.' >&2; exit 1;
}

run_stage() {
    local n="$1"
    local out="${RESULTS}/chunks_${n}"
    local best="${out}/best_mlp_${FEATURES}.pt"
    local last="${out}/last_mlp_${FEATURES}.pt"
    local recovery="${out}/recovery_mlp_${FEATURES}.pt"
    local done="${out}/completed_${EPOCHS}_epochs.ok"
    mkdir -p "$out"
    if [[ -f "$done" ]]; then
        [[ -s "$best" && -s "$last" ]] || { echo "Missing checkpoints: $out" >&2; exit 1; }
        PREVIOUS_BEST="$best"
        echo "Skipping completed stage: $n chunks"
        return
    fi
    local -a cmd=("$PYTHON" -u "$SCRIPT"
        --features "$FEATURES" --epochs "$EPOCHS" --seed 42
        --learning-rate 0.001 --weight-decay 0.0001
        --scheduler-t0 20 --scheduler-tmult 1 --scheduler-min-lr 0.00001
        --max-chunks "$n" --report-every-chunks 1 --secondary-cache-size 1 --eval-batch-size 1024
        --eval-target-per-class 29006 --shared-eval-dir "$RESULTS/shared_global_eval" --output-dir "$out")
    # A completed stage may have finished before its marker was written.
    local resume_path=""
    if [[ -s "$recovery" ]]; then
        resume_path="$recovery"
    elif [[ -s "$last" ]]; then
        resume_path="$last"
    fi
    if [[ -n "$resume_path" ]]; then
        local completed
        completed="$("$PYTHON" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("completed_epochs",0))' "$resume_path")"
        if (( completed >= EPOCHS )); then
            [[ -s "$best" && -s "$last" ]] || { echo "Missing completed checkpoints: $out" >&2; exit 1; }
            touch "$done"
            PREVIOUS_BEST="$best"
            return
        fi
        cmd+=(--resume "$resume_path")
    elif [[ -n "$PREVIOUS_BEST" ]]; then
        [[ -s "$PREVIOUS_BEST" ]] || { echo "Missing previous checkpoint: $PREVIOUS_BEST" >&2; exit 1; }
        cmd+=(--init-checkpoint "$PREVIOUS_BEST")
    fi
    echo "========== LAYERNORM SHUFFLED STAGE $n chunks / $EPOCHS epochs / GPU $CUDA_VISIBLE_DEVICES =========="
    printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
    "${cmd[@]}" 2>&1 | tee -a "${out}/run.log"
    [[ -s "$best" && -s "$last" ]] || { echo "Missing checkpoints: $out" >&2; exit 1; }
    touch "$done"
    PREVIOUS_BEST="$best"
}

for ((n=100; n<=FINAL_CHUNKS; n+=STEP)); do
    run_stage "$n"
done
if (( FINAL_CHUNKS > 100 && FINAL_CHUNKS % STEP != 0 )); then
    run_stage "$FINAL_CHUNKS"
fi
echo "LAYERNORM LADDER COMPLETE. FINAL BEST: $PREVIOUS_BEST"
