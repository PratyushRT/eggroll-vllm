#!/bin/bash
# ============================================================
# PAPER RLCR run — plain RLCR reward for the optimizer-question
# paper. Fitness = correctness + 0.5·(-Brier):
#
#     R_i = C_i − 0.5 · (q_i − C_i)^2
#
# No anchor set, no LOO retention, no format bonus. The simplest
# possible calibrated reward, which DCPO uses as one of their
# ablation targets ("RLCR") and for which they publish numbers.
#
# Usage:
#   ./native_launch_paper_rlcr.sh smoke    # 3 iters, ~15 min — smoke test
#   ./native_launch_paper_rlcr.sh paper    # 100 iters, full run
#
# Diagnostics wired in (new this run):
#   • fitness/global_std_pre_floor        — pre-floor raw σ of rewards
#   • fitness/antithetic_pair_diff_mean   — mean |f(+ε) − f(−ε)|
#   • conf_hist_bin{0..9}_rate            — 10-bin confidence histogram
#   • wrong_conf_mean, wrong_frac_q_ge_0p8
#   • pass_at_1 / pass_at_4 (Chen et al. 2021 unbiased)
#   • 5 benchmarks at every eval: MATH-500, AIME24, AIME25, AMC23, AMC24
# ============================================================

set -u

if [[ $# -lt 1 ]]; then
    echo "usage: $0 {smoke|paper}" >&2
    exit 2
fi
MODE="$1"

case "$MODE" in
    smoke)
        # Smoke: reduced-scale end-to-end validation. Exercises every new
        # diagnostic path (raw σ, antithetic pair-diff, conf histogram,
        # pass@1/4, all 5 benchmarks) without burning a full 13 min/step.
        num_iterations="2"
        steps_per_eval="100"          # eval only at step 0
        math500_n="40"
        math500_repeats="1"
        aime24_repeats="1"
        amc24_repeats="1"
        aime25_repeats="1"
        amc23_repeats="1"
        # Scale overrides (applied after the block) for faster smoke:
        smoke_pop="64"
        smoke_spp="2"
        smoke_pb="2"
        smoke_max_tokens="1024"
        STAGE_NAME="paper-rlcr-smoke"
        ;;
    paper)
        num_iterations="100"
        steps_per_eval="10"
        math500_n="200"
        math500_repeats="2"
        aime24_repeats="4"
        amc24_repeats="4"
        aime25_repeats="4"
        amc23_repeats="4"
        STAGE_NAME="paper-rlcr-100it"
        ;;
    *)
        echo "unknown mode $MODE (want smoke or paper)" >&2
        exit 2
        ;;
esac

# ============================================================
# Hyperparameters (plain RLCR, user-specified)
# ============================================================
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"
reward_variant="rlcr"
prompt_template="dcpo_verbose"

# RLCR calibration coefficient: R = C − λ·(q−C)². λ=0.5 ≡ +0.5·(−Brier).
lambda_cal="0.5"

# ES noise + learning rate.
sigma="0.001"
learning_rate="0.001"

# Normalization: per-prompt center, global-std denom with a hard floor at
# 0.01 so tiny-std steps do NOT blow z into the clip band. Raw σ is logged
# separately each step (fitness/global_std_pre_floor) as the mode-collapse
# diagnostic.
global_std_floor="0.01"
fitness_clip="3.0"

# ES topology.
population_size="256"
steps_per_adapter="4"
lora_r="1"

# Batch composition. samples_per_prompt=4 gives enough rollouts per prompt
# for pass@4 and for the RLCR per-rollout reward to have meaningful
# population-level gradient signal.
prompt_batch_size="4"
samples_per_prompt="4"

# Generation.
max_tokens="3000"
temperature="0.7"
enable_thinking="False"

# Seed.
base_seed="42"

# Smoke-mode scale overrides (pop / spp / prompt_batch / max_tokens).
if [[ "$MODE" == "smoke" ]]; then
    population_size="$smoke_pop"
    samples_per_prompt="$smoke_spp"
    prompt_batch_size="$smoke_pb"
    max_tokens="$smoke_max_tokens"
fi

# Paths.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
mkdir -p checkpoints logs wandb cache/huggingface

export WANDB_DIR="${REPO_ROOT}/wandb"
export CUDA_VISIBLE_DEVICES="0"

ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

TS=$(date +%s)
GCS_STAGE_PREFIX="gs://esvpg-experiments/es_exp/paper_rlcr_${MODE}_${TS}"
RUN_NAME="${STAGE_NAME}-s${sigma}-lr${learning_rate}-${TS}"
LOG_FILE="${REPO_ROOT}/logs/${RUN_NAME}.log"

# ============================================================
# Helpers (same pattern as native_launch_triage_A1A2.sh)
# ============================================================
hard_cleanup() {
    echo "[cleanup] Stopping Ray..."
    ray stop --force > /dev/null 2>&1 || true
    sleep 2
    echo "[cleanup] Killing lingering es_lora / ray / vllm processes..."
    pkill -9 -f "es_lora_multinode.py" 2>/dev/null || true
    pkill -9 -f "ray::"                2>/dev/null || true
    pkill -9 -f "python.*vllm"         2>/dev/null || true
    pkill -9 -f "EngineCore_DP"        2>/dev/null || true
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
echo ""
echo "==============================================================="
echo "PAPER RLCR ($MODE)"
echo "  reward: R = C − ${lambda_cal}·(q−C)²  (plain RLCR, no format bonus)"
echo "  sigma=$sigma lr=$learning_rate iters=$num_iterations"
echo "  pop=$population_size spp=$samples_per_prompt pb=$prompt_batch_size"
echo "  max_tokens=$max_tokens temp=$temperature"
echo "  global_std_floor=$global_std_floor fitness_clip=$fitness_clip"
echo "  steps_per_eval=$steps_per_eval"
echo "  eval benches: MATH-500(n=$math500_n×$math500_repeats) AIME24(×$aime24_repeats) AIME25(×$aime25_repeats) AMC23(×$amc23_repeats) AMC24(×$amc24_repeats)"
echo "  run_name=$RUN_NAME"
echo "  gcs_dest=$GCS_STAGE_PREFIX"
echo "  start=$(date -Iseconds)"
echo "==============================================================="

hard_cleanup

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
fi

if ! start_ray_with_retry; then
    echo "[paper-rlcr $MODE] ABORTED: Ray could not start"
    exit 2
fi

# ============================================================
# Pre-emption-resilient background uploader.
#
# Every ${UPLOAD_INTERVAL_SECS} seconds, rsync the full run directory
# (checkpoints/${RUN_NAME}/) to GCS. `gcloud storage rsync` is
# idempotent — unchanged files are skipped — so:
#   • Every 10th-step checkpoint lands in GCS within ≤5 min of being
#     written (history, permanent).
#   • The LATEST 10th-step checkpoint is always <5 min stale in GCS,
#     giving us a warm-start point after a spot pre-emption.
#   • in_run_dcpo_eval/step_*.json snapshots ride along — so even if
#     the run dies between eval and next save, the eval JSON is safe.
#   • The live training log is uploaded alongside.
#
# The uploader runs in the background, is killed on EXIT, and logs to
# a dedicated file so its stderr doesn't pollute the training log.
# ============================================================
UPLOAD_INTERVAL_SECS="${UPLOAD_INTERVAL_SECS:-300}"
UPLOADER_LOG="${REPO_ROOT}/logs/${RUN_NAME}_uploader.log"
background_uploader() {
    while true; do
        sleep "$UPLOAD_INTERVAL_SECS"
        {
            echo "[uploader $(date -Iseconds)] rsyncing checkpoints + eval JSONs"
            if command -v gcloud >/dev/null 2>&1; then
                if [[ -d "${REPO_ROOT}/checkpoints/${RUN_NAME}" ]]; then
                    # Rsync the full run directory — catches every
                    # checkpoint_step_* dir AND in_run_dcpo_eval/*.json.
                    gcloud storage rsync --recursive \
                        "${REPO_ROOT}/checkpoints/${RUN_NAME}" \
                        "$GCS_STAGE_PREFIX/run/" 2>&1 | tail -3 || true
                fi
                # Always ship the latest live log so remote diagnostics are current.
                if [[ -f "$LOG_FILE" ]]; then
                    gcloud storage cp "$LOG_FILE" "$GCS_STAGE_PREFIX/logs/" 2>&1 | tail -1 || true
                fi
            fi
        } >> "$UPLOADER_LOG" 2>&1 || true
    done
}

background_uploader &
UPLOADER_PID=$!
echo "[uploader] started (pid=$UPLOADER_PID, interval=${UPLOAD_INTERVAL_SECS}s, log=$UPLOADER_LOG)"
# Kill the uploader whenever this script exits (success or fail).
trap 'kill "$UPLOADER_PID" 2>/dev/null || true' EXIT

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
    --global-std-floor "$global_std_floor" \
    --fitness-clip "$fitness_clip" \
    --no-format-reward-enabled \
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
    --in-run-dcpo-math500-n "$math500_n" \
    --in-run-dcpo-math500-repeats "$math500_repeats" \
    --in-run-dcpo-aime24-repeats "$aime24_repeats" \
    --in-run-dcpo-amc24-repeats "$amc24_repeats" \
    --in-run-dcpo-aime25-repeats "$aime25_repeats" \
    --in-run-dcpo-amc23-repeats "$amc23_repeats" \
    --save-freq 10 \
    --name-prefix "$RUN_NAME" \
    --checkpoint-dir "${REPO_ROOT}/checkpoints/${RUN_NAME}" \
    --use-wandb 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
TRAIN_END=$(date +%s)
ELAPSED=$((TRAIN_END - TRAIN_START))

SUMMARY_PATH="${REPO_ROOT}/logs/${RUN_NAME}_summary.txt"
cat > "$SUMMARY_PATH" <<EOF
mode=${MODE}
run_name=${RUN_NAME}
reward_variant=${reward_variant}
lambda_cal=${lambda_cal}
sigma=${sigma}
learning_rate=${learning_rate}
global_std_floor=${global_std_floor}
fitness_clip=${fitness_clip}
population_size=${population_size}
samples_per_prompt=${samples_per_prompt}
prompt_batch_size=${prompt_batch_size}
max_tokens=${max_tokens}
temperature=${temperature}
num_iterations=${num_iterations}
steps_per_eval=${steps_per_eval}
math500_n=${math500_n}
math500_repeats=${math500_repeats}
aime24_repeats=${aime24_repeats}
amc24_repeats=${amc24_repeats}
aime25_repeats=${aime25_repeats}
amc23_repeats=${amc23_repeats}
base_seed=${base_seed}
train_wall_seconds=${ELAPSED}
exit_code=${EXIT_CODE}
EOF
echo "paper-rlcr $MODE: exit=$EXIT_CODE, wall=${ELAPSED}s, end=$(date -Iseconds)"

# ============================================================
# Final upload to GCS (runs even on failure / pre-emption trap).
#
# The background uploader has been rsync'ing every
# ${UPLOAD_INTERVAL_SECS}s throughout the run, so most data is
# already in GCS. This final pass:
#   • Kills the uploader so it doesn't race with us.
#   • Rsyncs the run directory one last time (catches the final
#     checkpoint + the last eval JSON + any new log lines).
#   • Ships the summary file, uploader log, and wandb offline data.
# ============================================================
echo ""
echo "[gcs-final] stopping background uploader and doing final sync"
kill "$UPLOADER_PID" 2>/dev/null || true
wait "$UPLOADER_PID" 2>/dev/null || true
trap - EXIT   # clear trap so we don't double-kill below

if command -v gcloud >/dev/null 2>&1; then
    # Log + summary.
    gcloud storage cp "$LOG_FILE" "$SUMMARY_PATH" "$GCS_STAGE_PREFIX/logs/" 2>&1 | tail -3 || true
    if [[ -f "$UPLOADER_LOG" ]]; then
        gcloud storage cp "$UPLOADER_LOG" "$GCS_STAGE_PREFIX/logs/" 2>&1 | tail -1 || true
    fi
    # Final rsync of the full run dir — catches the last checkpoint
    # + eval JSON + any newly-written files the periodic uploader
    # missed. Idempotent (skips unchanged).
    if [[ -d "${REPO_ROOT}/checkpoints/${RUN_NAME}" ]]; then
        echo "[gcs-final] rsync checkpoints/${RUN_NAME} -> $GCS_STAGE_PREFIX/run/"
        gcloud storage rsync --recursive \
            "${REPO_ROOT}/checkpoints/${RUN_NAME}" \
            "$GCS_STAGE_PREFIX/run/" 2>&1 | tail -5 || true
    fi
    # Wandb offline data (if present).
    if [[ -d "${REPO_ROOT}/wandb" ]]; then
        gcloud storage cp --recursive "${REPO_ROOT}/wandb" "$GCS_STAGE_PREFIX/wandb/" 2>&1 | tail -3 || true
    fi
else
    echo "[gcs-final] WARNING: gcloud not found on PATH — skipping upload"
fi

hard_cleanup

echo ""
echo "==============================================================="
echo "PAPER RLCR $MODE COMPLETE — exit=$EXIT_CODE, wall=${ELAPSED}s"
echo "Artefacts: $GCS_STAGE_PREFIX"
echo "==============================================================="

exit $EXIT_CODE
