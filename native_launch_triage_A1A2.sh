#!/bin/bash
# ============================================================
# TRIAGE RUN — A1 / A2 (40 steps each) + post-run GCS upload.
#
# Usage:
#   ./native_launch_triage_A1A2.sh A1   # rho=0.8 (hybrid LOO)
#   ./native_launch_triage_A1A2.sh A2   # rho=1.0 (strict per-sample Brier)
#
# Both variants use the same reward formula
#   r_ijk = C − λ_cal·(q − T)²
#         − λ_ret·A·(1−C)                      # retention (anchor only)
#         − λ_wc·A·(1−C)·q²                    # wrong-conf-anchor (anchor only)
#         + γ_fmt·valid_fmt − γ_bad·invalid − λ_trunc·truncated
# with T = ρ·C + (1−ρ)·C̄_{-i,j}. A1 sets ρ=0.8, A2 sets ρ=1.0.
#
# Per-prompt centering + GLOBAL-STD-FLOOR normalization prevents
# tiny-std steps from blowing up into fitness_clip.
#
# Batch composition: 50 % anchors (base-model solve rate ≥ 0.75) +
# 50 % ordinary DeepScaler prompts.
#
# After each stage finishes (success or fail), the logs, eval JSONs,
# and last checkpoint are copied to
#   gs://esvpg-experiments/es_exp/triage_${VARIANT}_${TS}/
# ============================================================

set -u

if [[ $# -lt 1 ]]; then
    echo "usage: $0 {A1|A2}" >&2
    exit 2
fi
VARIANT="$1"

case "$VARIANT" in
    A1) RHO="0.8"; STAGE_NAME="triage-A1-rho08";;
    A2) RHO="1.0"; STAGE_NAME="triage-A2-rho10";;
    *)  echo "unknown variant $VARIANT (want A1 or A2)" >&2; exit 2;;
esac

# ============================================================
# Shared hyperparameters (A1 and A2 only differ in rho)
# ============================================================
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"
reward_variant="rlcr_hybrid_loo_retention"
prompt_template="conf_tags"

# Penalty coefficients (expert plan)
lambda_cal="1.0"
lambda_retention="0.20"
lambda_wrong_conf_anchor="0.50"
lambda_trunc="0.25"
gamma_fmt="0.02"
gamma_bad="1.0"

# Stability knobs
sigma="0.0005"
learning_rate="0.0005"
global_std_floor="0.05"
fitness_clip="3.0"

# ES topology (kept identical to the stage-1 run for like-for-like comparison)
population_size="256"
steps_per_adapter="4"
lora_r="1"

# Batch composition
prompt_batch_size="4"
samples_per_prompt="1"
anchor_frac="0.5"

# Generation
max_tokens="2048"
temperature="0.7"
enable_thinking="False"

# Schedule
num_iterations="40"
steps_per_eval="10"

# Seed — MUST match whatever seed was used for precompute_anchor_set.py.
base_seed="42"

# Paths
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
mkdir -p checkpoints logs wandb cache/huggingface

ANCHOR_SET_PATH="${REPO_ROOT}/data/anchor_set_deepscaler40k_seed${base_seed}.json"
if [[ ! -f "$ANCHOR_SET_PATH" ]]; then
    echo "[fatal] anchor set not found at $ANCHOR_SET_PATH"
    echo "        Run: python precompute_anchor_set.py --seed ${base_seed} "
    echo "             --num-prompts 4000 --out $ANCHOR_SET_PATH"
    exit 3
fi

export WANDB_DIR="${REPO_ROOT}/wandb"
export CUDA_VISIBLE_DEVICES="0"

ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

# GCS destination for artefacts.
TS=$(date +%s)
GCS_STAGE_PREFIX="gs://esvpg-experiments/es_exp/triage_${VARIANT}_${TS}"

# ============================================================
# Helpers (copied from native_launch_overnight_rlcr.sh)
# ============================================================
hard_cleanup() {
    echo "[cleanup] Stopping Ray..."
    ray stop --force > /dev/null 2>&1 || true
    sleep 2
    echo "[cleanup] Killing lingering es_lora / ray / vllm processes..."
    pkill -9 -f "es_lora_multinode.py"  2>/dev/null || true
    pkill -9 -f "ray::"                 2>/dev/null || true
    pkill -9 -f "vllm"                  2>/dev/null || true
    sleep 3
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        for attempt in $(seq 1 12); do
            local gpu_mem_mb
            gpu_mem_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
            if [[ -z "$gpu_mem_mb" ]] || [[ "$gpu_mem_mb" -lt 2000 ]]; then
                echo "[cleanup] GPU memory drained: ${gpu_mem_mb:-?} MiB"
                return 0
            fi
            echo "[cleanup] GPU still holding ${gpu_mem_mb} MiB, waiting... ($attempt/12)"
            sleep 5
        done
        echo "[cleanup] WARNING: GPU did not fully drain after 60s; proceeding anyway"
    fi
    return 0
}

start_ray_with_retry() {
    for attempt in 1 2 3; do
        if ray start --head --port=6379 --dashboard-host=0.0.0.0 > /tmp/ray_start.log 2>&1; then
            echo "[ray] started on attempt $attempt"
            return 0
        fi
        echo "[ray] start failed on attempt $attempt; retrying in 10s"
        ray stop --force > /dev/null 2>&1 || true
        sleep 10
    done
    echo "[ray] FATAL: could not start Ray after 3 attempts"
    cat /tmp/ray_start.log
    return 1
}

# ============================================================
# Run
# ============================================================
RUN_NAME="${STAGE_NAME}-s${sigma}-lr${learning_rate}-${TS}"
LOG_FILE="${REPO_ROOT}/logs/${RUN_NAME}.log"

echo ""
echo "==============================================================="
echo "TRIAGE: $VARIANT  (rho=$RHO)"
echo "  sigma=$sigma lr=$learning_rate iters=$num_iterations"
echo "  lambda_cal=$lambda_cal lambda_ret=$lambda_retention"
echo "  lambda_wc=$lambda_wrong_conf_anchor lambda_trunc=$lambda_trunc"
echo "  anchor_frac=$anchor_frac global_std_floor=$global_std_floor fitness_clip=$fitness_clip"
echo "  base_seed=$base_seed anchor_set=$ANCHOR_SET_PATH"
echo "  run_name=$RUN_NAME"
echo "  gcs_dest=$GCS_STAGE_PREFIX"
echo "  start=$(date -Iseconds)"
echo "==============================================================="

hard_cleanup

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
fi

if ! start_ray_with_retry; then
    echo "[triage $VARIANT] ABORTED: Ray could not start"
    exit 2
fi

TRAIN_START=$(date +%s)

python es_lora_multinode.py \
    --sigma "$sigma" \
    --learning-rate "$learning_rate" \
    --max-tokens "$max_tokens" \
    --model-name "$model_name" \
    --population-size "$population_size" \
    --steps-per-adapter "$steps_per_adapter" \
    --lora-r "$lora_r" \
    --num-iterations "$num_iterations" \
    --task "$task" \
    --reward-variant "$reward_variant" \
    --prompt-template "$prompt_template" \
    --lambda-cal "$lambda_cal" \
    --rho "$RHO" \
    --gamma-fmt "$gamma_fmt" \
    --gamma-bad "$gamma_bad" \
    --lambda-retention "$lambda_retention" \
    --lambda-wrong-conf-anchor "$lambda_wrong_conf_anchor" \
    --lambda-trunc "$lambda_trunc" \
    --anchor-set-path "$ANCHOR_SET_PATH" \
    --anchor-frac "$anchor_frac" \
    --global-std-floor "$global_std_floor" \
    --fitness-clip "$fitness_clip" \
    --format-reward-enabled \
    --per-prompt-normalize \
    --no-normalize-with-std \
    --no-scale-lr-in-grad \
    $ENABLE_THINKING_FLAG \
    --prompt-batch-size "$prompt_batch_size" \
    --samples-per-prompt "$samples_per_prompt" \
    --temperature "$temperature" \
    --steps-per-eval "$steps_per_eval" \
    --base-seed "$base_seed" \
    --in-run-dcpo-eval \
    --in-run-dcpo-math500-n 200 \
    --in-run-dcpo-math500-repeats 2 \
    --in-run-dcpo-aime24-repeats 8 \
    --in-run-dcpo-amc24-repeats 4 \
    --name-prefix "$RUN_NAME" \
    --checkpoint-dir "${REPO_ROOT}/checkpoints/${RUN_NAME}" \
    --use-wandb 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
TRAIN_END=$(date +%s)
ELAPSED=$((TRAIN_END - TRAIN_START))

SUMMARY_PATH="${REPO_ROOT}/logs/${RUN_NAME}_summary.txt"
cat > "$SUMMARY_PATH" <<EOF
variant=${VARIANT}
rho=${RHO}
run_name=${RUN_NAME}
sigma=${sigma}
learning_rate=${learning_rate}
lambda_cal=${lambda_cal}
lambda_retention=${lambda_retention}
lambda_wrong_conf_anchor=${lambda_wrong_conf_anchor}
lambda_trunc=${lambda_trunc}
gamma_fmt=${gamma_fmt}
gamma_bad=${gamma_bad}
global_std_floor=${global_std_floor}
fitness_clip=${fitness_clip}
anchor_frac=${anchor_frac}
base_seed=${base_seed}
anchor_set=${ANCHOR_SET_PATH}
num_iterations=${num_iterations}
steps_per_eval=${steps_per_eval}
train_wall_seconds=${ELAPSED}
exit_code=${EXIT_CODE}
EOF
echo "Triage $VARIANT: exit=$EXIT_CODE, wall=${ELAPSED}s, end=$(date -Iseconds)"

# ============================================================
# Upload artefacts to GCS
# (Runs even on failure — partial data is still useful.)
# ============================================================
echo ""
echo "[gcs] uploading artefacts to $GCS_STAGE_PREFIX"
if command -v gcloud >/dev/null 2>&1; then
    # Logs + summary
    gcloud storage cp "$LOG_FILE" "$SUMMARY_PATH" "$GCS_STAGE_PREFIX/logs/" 2>&1 | tail -3 || true
    # In-run eval JSONs
    if [[ -d "${REPO_ROOT}/checkpoints/${RUN_NAME}/in_run_dcpo_eval" ]]; then
        gcloud storage cp --recursive \
            "${REPO_ROOT}/checkpoints/${RUN_NAME}/in_run_dcpo_eval" \
            "$GCS_STAGE_PREFIX/eval_jsons/" 2>&1 | tail -3 || true
    fi
    # Anchor set (once — idempotent upload).
    gcloud storage cp "$ANCHOR_SET_PATH" \
        "gs://esvpg-experiments/es_exp/anchor_sets/$(basename "$ANCHOR_SET_PATH")" \
        2>&1 | tail -3 || true
    # Last checkpoint (big — skip if disk is low).
    LAST_CKPT=$(ls -dt "${REPO_ROOT}/checkpoints/${RUN_NAME}/checkpoint_step_"* 2>/dev/null | head -n1)
    if [[ -n "$LAST_CKPT" ]] && [[ -d "$LAST_CKPT" ]]; then
        echo "[gcs] uploading $LAST_CKPT"
        gcloud storage cp --recursive "$LAST_CKPT" "$GCS_STAGE_PREFIX/checkpoints/" 2>&1 | tail -3 || true
    fi
    # Wandb offline data (if any).
    if [[ -d "${REPO_ROOT}/wandb" ]]; then
        gcloud storage cp --recursive "${REPO_ROOT}/wandb" "$GCS_STAGE_PREFIX/wandb/" 2>&1 | tail -3 || true
    fi
else
    echo "[gcs] WARNING: gcloud not found on PATH — skipping upload"
fi

hard_cleanup

echo ""
echo "==============================================================="
echo "TRIAGE $VARIANT COMPLETE — exit=$EXIT_CODE, wall=${ELAPSED}s"
echo "Artefacts: $GCS_STAGE_PREFIX"
echo "==============================================================="

exit $EXIT_CODE
