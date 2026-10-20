#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

FEATURES="primary_secondary"

SEED=42
EPOCHS=100

FINAL_CHUNKS=14503
STEP=10
RECOVERY_BLOCK=10

# Optimizer
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0001

# CosineAnnealingWarmRestarts
SCHEDULER_T0=20
SCHEDULER_TMULT=1
SCHEDULER_MIN_LR=0.00001

# Paths
BASE_DIR="/data/nap/lperez_nn/data/NN/Multilayer_Percepton"
SCRIPT="${BASE_DIR}/mlp_baseline.py"
RESULTS="${BASE_DIR}/chunk_ladder_cosine_100epochs_1GPU_balancedtest"

mkdir -p "$RESULTS"

# ============================================================
# TRAINING
# ============================================================

PREVIOUS_BEST=""

for ((N=STEP; N<=FINAL_CHUNKS; N+=STEP)); do

    OUT="${RESULTS}/chunks_${N}"

    BEST="${OUT}/best_mlp_${FEATURES}.pt"
    LAST="${OUT}/last_mlp_${FEATURES}.pt"
    RECOVERY="${OUT}/recovery_mlp_${FEATURES}.pt"

    DONE="${OUT}/completed_100_epochs.ok"

    mkdir -p "$OUT"

    echo "========== STAGE ${N} chunks / ${EPOCHS} epochs =========="

    # Skip completed stages
    if [[ -f "$DONE" ]]; then

        [[ -s "$BEST" && -s "$LAST" ]] || {
            echo "ERROR: completed marker but missing checkpoints: $OUT"
            exit 1
        }

        PREVIOUS_BEST="$BEST"

        echo "Already completed; skipping."
        continue
    fi

    # Training command
    CMD=(
        torchrun
        --standalone
        --nproc_per_node=1
        "$SCRIPT"

        --features "$FEATURES"
        --epochs "$EPOCHS"
        --seed "$SEED"

        --learning-rate "$LEARNING_RATE"
        --weight-decay "$WEIGHT_DECAY"

        --scheduler-t0 "$SCHEDULER_T0"
        --scheduler-tmult "$SCHEDULER_TMULT"
        --scheduler-min-lr "$SCHEDULER_MIN_LR"

        --max-chunks "$N"
        --report-every-chunks "$RECOVERY_BLOCK"

        --output-dir "$OUT"
    )

    # Checkpoint handling
    if [[ -s "$RECOVERY" ]]; then

        CMD+=(--resume "$RECOVERY")

    elif [[ -s "$LAST" ]]; then

        CMD+=(--resume "$LAST")

    elif [[ -n "$PREVIOUS_BEST" ]]; then

        CMD+=(--init-checkpoint "$PREVIOUS_BEST")

    fi

    # Execute training
    "${CMD[@]}" 2>&1 | tee -a "${OUT}/run.log"

    # Verify checkpoints
    [[ -s "$BEST" && -s "$LAST" ]] || {
        echo "ERROR: missing checkpoints in $OUT"
        exit 1
    }

    touch "$DONE"

    PREVIOUS_BEST="$BEST"

done

# ============================================================
# FINAL STAGE: REMAINING CHUNKS
# ============================================================

if (( FINAL_CHUNKS % STEP != 0 )); then

    N="$FINAL_CHUNKS"

    OUT="${RESULTS}/chunks_${N}"

    BEST="${OUT}/best_mlp_${FEATURES}.pt"
    LAST="${OUT}/last_mlp_${FEATURES}.pt"
    RECOVERY="${OUT}/recovery_mlp_${FEATURES}.pt"

    DONE="${OUT}/completed_100_epochs.ok"

    mkdir -p "$OUT"

    if [[ -f "$DONE" ]]; then

        [[ -s "$BEST" && -s "$LAST" ]] || {
            echo "ERROR: completed marker but missing checkpoints: $OUT"
            exit 1
        }

        echo "Final stage already completed; skipping."

    else

        echo "========== FINAL STAGE ${N} chunks / ${EPOCHS} epochs =========="

        CMD=(
            torchrun
            --standalone
            --nproc_per_node=1
            "$SCRIPT"

            --features "$FEATURES"
            --epochs "$EPOCHS"
            --seed "$SEED"

            --learning-rate "$LEARNING_RATE"
            --weight-decay "$WEIGHT_DECAY"

            --scheduler-t0 "$SCHEDULER_T0"
            --scheduler-tmult "$SCHEDULER_TMULT"
            --scheduler-min-lr "$SCHEDULER_MIN_LR"

            --max-chunks "$N"
            --report-every-chunks "$RECOVERY_BLOCK"

            --output-dir "$OUT"
        )

        if [[ -s "$RECOVERY" ]]; then

            CMD+=(--resume "$RECOVERY")

        elif [[ -s "$LAST" ]]; then

            CMD+=(--resume "$LAST")

        elif [[ -n "$PREVIOUS_BEST" ]]; then

            CMD+=(--init-checkpoint "$PREVIOUS_BEST")

        fi

        "${CMD[@]}" 2>&1 | tee -a "${OUT}/run.log"

        [[ -s "$BEST" && -s "$LAST" ]] || {
            echo "ERROR: missing checkpoints in $OUT"
            exit 1
        }

        touch "$DONE"

    fi

    PREVIOUS_BEST="$BEST"

fi

echo "CHUNK LADDER COMPLETE. FINAL BEST: $PREVIOUS_BEST"