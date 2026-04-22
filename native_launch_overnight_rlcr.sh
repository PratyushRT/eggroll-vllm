#!/bin/bash
# OVERNIGHT RUN — Expert 3 plan, full.
#
# Sequence (single GPU, all runs serial):
#   Stage-0/sweep-1: sigma=1e-3  × 20 iters, eval every 5  (smoke + anchor)
#   Sweep-2:         sigma=5e-4  × 20 iters, eval every 5
#   Sweep-3:         sigma=2e-3  × 20 iters, eval every 5
#   Stage-1:         sigma=1e-3  × 100 iters, eval every 10 (signal run, anchor)
#
# Total expected wall-clock: ~11-13 hr on 1×H100 spot.
# If Stage-1 shows a clear climb at 100 iters, separate Stage-2 run (200 iters)
# can be launched in the morning.
#
# All runs use:
#   - reward_variant=rlcr_hybrid_loo (LOO cross-population group mean, rho=0.5)
#   - prompt_template=conf_tags  (<conf>...</conf> format)
#   - per-prompt normalization with GLOBAL std (GRPO-style, Expert 3 spec)
#   - format reward: gamma_fmt=0.05, gamma_bad=1.0
#   - lambda_cal=1.0 (up from 0.5 — Expert 3 anchor)
#   - normalize_with_std=OFF, scale_lr_in_grad=OFF (per-prompt norm replaces them)
#   - pop=256, r=1 (only combination that fits + uses torch.mm fast path)
#   - temperature=0.7  (was 1.0; matches DCPO eval temp)
#   - samples_per_prompt=1  (expert default; LOO group mean across population covers variance)
#   - steps_per_adapter=4  (LoRA reuse 4, EGGROLL default)
#   - max_tokens=2048 for sweep, 2048 for Stage-1 too (keep cost tractable)
#
# In-run eval (every N steps):
#   - MATH-500 first 100 × 2 samples
#   - AIME24 30 × 4 samples
#   - AMC24 45 × 2 samples (skipped gracefully if data/amc24.parquet missing)
#   - Full DCPO metric suite (Acc, ECE, PCE, Brier, AUROC, pass@k, conf_entropy)

set -u

# ============================================================
# Common config (shared across all stages)
# ============================================================
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"
reward_variant="rlcr_hybrid_loo"
prompt_template="conf_tags"
lambda_cal="1.0"
rho="0.5"
gamma_fmt="0.05"
gamma_bad="1.0"
learning_rate="0.001"
population_size="256"
steps_per_adapter="4"
lora_r="1"
prompt_batch_size="4"        # more prompts per step → more LOO denominator
samples_per_prompt="1"        # Expert 3 anchor
max_tokens="2048"
temperature="0.7"
enable_thinking="False"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"
mkdir -p checkpoints logs wandb cache/huggingface

export WANDB_DIR="${REPO_ROOT}/wandb"
export CUDA_VISIBLE_DEVICES="0"

ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

# ============================================================
# Robustness helpers
# ============================================================

# Force-kill anything holding the GPU from a previous run. Idempotent.
hard_cleanup() {
    echo "[cleanup] Stopping Ray..."
    ray stop --force > /dev/null 2>&1 || true
    sleep 2

    echo "[cleanup] Killing lingering es_lora / ray / vllm processes..."
    pkill -9 -f "es_lora_multinode.py"  2>/dev/null || true
    pkill -9 -f "ray::"                 2>/dev/null || true
    pkill -9 -f "vllm"                  2>/dev/null || true
    sleep 3

    echo "[cleanup] Clearing /dev/shm artifacts..."
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true

    # Wait up to 60s for GPU memory to drain.
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

# Start Ray with retry.
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

# Pre-flight: min 50GB free disk, GPU visible.
preflight_check() {
    local free_gb
    free_gb=$(df -BG "${REPO_ROOT}" | awk 'NR==2 {gsub("G","",$4); print $4}')
    if [[ -n "$free_gb" ]] && [[ "$free_gb" -lt 30 ]]; then
        echo "[preflight] WARNING: only ${free_gb}G free on ${REPO_ROOT}; stage may fail to checkpoint"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1
    else
        echo "[preflight] WARNING: nvidia-smi not found"
    fi
}

# ============================================================
# Launch helper
# ============================================================
run_stage() {
    local stage_name="$1"
    local sigma="$2"
    local num_iterations="$3"
    local steps_per_eval="$4"

    echo ""
    echo "==============================================================="
    echo "STAGE: $stage_name"
    echo "  sigma=$sigma  iters=$num_iterations  eval_every=$steps_per_eval"
    echo "  start=$(date -Iseconds)"
    echo "==============================================================="

    hard_cleanup
    preflight_check

    if ! start_ray_with_retry; then
        echo "[stage $stage_name] ABORTED: Ray could not start"
        return 2
    fi

    local RUN_NAME="overnight-${stage_name}-s${sigma}-$(date +%s)"
    local LOG_FILE="${REPO_ROOT}/logs/${RUN_NAME}.log"

    local TRAIN_START=$(date +%s)

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
        --rho "$rho" \
        --gamma-fmt "$gamma_fmt" \
        --gamma-bad "$gamma_bad" \
        --format-reward-enabled \
        --per-prompt-normalize \
        --no-normalize-with-std \
        --no-scale-lr-in-grad \
        $ENABLE_THINKING_FLAG \
        --prompt-batch-size "$prompt_batch_size" \
        --samples-per-prompt "$samples_per_prompt" \
        --temperature "$temperature" \
        --steps-per-eval "$steps_per_eval" \
        --in-run-dcpo-eval \
        --in-run-dcpo-math500-n 100 \
        --in-run-dcpo-math500-repeats 2 \
        --in-run-dcpo-aime24-repeats 4 \
        --in-run-dcpo-amc24-repeats 2 \
        --name-prefix "$RUN_NAME" \
        --checkpoint-dir "${REPO_ROOT}/checkpoints" \
        --use-wandb 2>&1 | tee "$LOG_FILE"

    local EXIT_CODE=${PIPESTATUS[0]}
    local TRAIN_END=$(date +%s)
    local ELAPSED=$((TRAIN_END - TRAIN_START))

    cat > "${REPO_ROOT}/logs/${RUN_NAME}_summary.txt" <<EOF
run_name=${RUN_NAME}
stage=${stage_name}
sigma=${sigma}
num_iterations=${num_iterations}
steps_per_eval=${steps_per_eval}
train_wall_seconds=${ELAPSED}
exit_code=${EXIT_CODE}
EOF

    echo "Stage $stage_name: exit=$EXIT_CODE, wall=${ELAPSED}s, end=$(date -Iseconds)"

    # Always cleanup after a stage — even on success, so the next stage
    # starts from a guaranteed-clean state.
    hard_cleanup

    return $EXIT_CODE
}

# Wrap run_stage so a single stage failure does NOT abort the whole sequence.
# Logs to a manifest so morning review can see which stages succeeded.
MANIFEST="${REPO_ROOT}/logs/overnight_manifest_$(date +%s).tsv"
echo -e "stage\tsigma\titers\texit_code\twall_seconds\tstart\tend" > "$MANIFEST"

run_stage_safe() {
    local stage_name="$1"
    local sigma="$2"
    local num_iterations="$3"
    local steps_per_eval="$4"

    local start_iso=$(date -Iseconds)
    local start_epoch=$(date +%s)
    set +e  # never let a stage failure abort the script
    run_stage "$stage_name" "$sigma" "$num_iterations" "$steps_per_eval"
    local ec=$?
    set -e || true
    set -u
    local end_epoch=$(date +%s)
    local end_iso=$(date -Iseconds)
    echo -e "${stage_name}\t${sigma}\t${num_iterations}\t${ec}\t$((end_epoch - start_epoch))\t${start_iso}\t${end_iso}" >> "$MANIFEST"

    if [[ $ec -ne 0 ]]; then
        echo "[orchestrator] Stage ${stage_name} failed (exit=$ec) — continuing to next stage anyway"
    fi
    return 0
}

# ============================================================
# Stage orchestration
# ============================================================

OVERNIGHT_START=$(date +%s)
echo "OVERNIGHT RUN START: $(date -Iseconds)"
echo "Manifest: $MANIFEST"

# Sweep 1 (also serves as smoke + anchor point): sigma=1e-3, 20 iters
run_stage_safe "sweep-s1e3"  "0.001"  20  5

# Sweep 2: sigma=5e-4, 20 iters
run_stage_safe "sweep-s5e4"  "0.0005" 20  5

# Sweep 3: sigma=2e-3, 20 iters
run_stage_safe "sweep-s2e3"  "0.002"  20  5

# Stage 1: main signal run, 100 iters with anchor sigma=1e-3
# (If morning review shows another sigma won the sweep, rerun this stage.)
run_stage_safe "stage1-signal" "0.001" 100 10

OVERNIGHT_END=$(date +%s)
TOTAL_WALL=$((OVERNIGHT_END - OVERNIGHT_START))

echo ""
echo "==============================================================="
echo "OVERNIGHT RUN COMPLETE"
echo "Total wall-clock: ${TOTAL_WALL}s = $(python3 -c "print(round(${TOTAL_WALL}/3600.0, 2))") hr"
echo "==============================================================="
echo ""
echo "Manifest:"
cat "$MANIFEST"
