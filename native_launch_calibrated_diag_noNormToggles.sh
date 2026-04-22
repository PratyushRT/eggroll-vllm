#!/bin/bash
# DIAGNOSTIC v3 — toggles OFF + lower temperature.
#
# Post-mortem on lr=1e-3 and lr=5e-4 runs: both oscillated ±0.4 to +0.99 in lockstep.
# Halving lr changed NOTHING because normalize_with_std makes the update direction
# scale-invariant (fitness z-scored before weighting noise). lr only scales
# magnitude, not direction — and the direction itself was bad.
#
# Misalignment audit vs author's 8B email:
#   - Author's tested 8B configs: pop=512 or pop=16k (we're at 256, below floor)
#   - Author's temperature: 0.0 greedy (we're at 1.0)
#   - Author's samples_per_prompt: 1 (we're at 4 for Hybrid group mean)
#   - Author: normalize_with_std + scale_lr_in_grad "always ON, EXCEPT base models"
#     → Qwen3-8B may qualify as "base" (pre-trained + mild instruct-tuned, no deep RLHF)
#   - Author's observed pattern: truncation exponentially DECREASES over steps
#   - Our pattern: truncation spikes from 1% to 55% at step 1, oscillates.
#     This is the "toggles backfiring on base models" failure mode.
#
# This run: test Path A (cheap, 1-GPU, 1hr):
#   - normalize_with_std OFF (removes std-division that amplifies noise when std→0)
#   - scale_lr_in_grad OFF (removes sqrt(pop) amplification)
#   - temperature 1.0 → 0.3 (closer to author's greedy, less per-rollout variance)
#   - lr back to 1e-3, sigma 1e-3 (author's anchor)
#
# If this shows monotone or near-monotone trajectory, the toggles were the
# instability source — great, we document this as "base model exception"
# validated empirically. If still chaotic, the issue is pop/temp/samples
# mismatch and we go to Path B (2-GPU pop=512 + greedy + Instance reward).

set -u

# --- Core model + task ---
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"

# --- DCPO-style reward config ---
reward_variant="hybrid"
lambda_cal="0.5"
instance_weight="0.3"
enable_thinking="False"

# --- ES hyperparameters ---
sigma="0.001"
learning_rate="0.001"       # back to author's anchor
population_size="256"
steps_per_adapter="4"
lora_r="1"
num_iterations="15"

# --- Rollout / batch ---
prompt_batch_size="1"
samples_per_prompt="4"
max_tokens="2048"
temperature="0.3"           # ← was 1.0; author used 0.0 greedy, 0.3 is minimum-variance-that-still-gives-hybrid-signal

# --- ES normalization toggles: BOTH OFF (Path A hypothesis test) ---
normalize_with_std=""       # ← was "normalize-with-std"; base-model exception per author
scale_lr_in_grad=""         # ← was "scale-lr-in-grad"; removes sqrt(pop) amplifier
pass_at_k=""

# --- Eval ---
steps_per_eval="999"
sub_dataset_size="null"

# --- Misc ---
name_prefix="native-calibrated-diag-noToggles-temp3e-1"
GPU_DEVICES="0"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints logs wandb cache/huggingface

DATASET_SIZE_CMD=""
if [[ "$sub_dataset_size" != "None" ]] && [[ "$sub_dataset_size" != "null" ]] && [[ -n "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

NORMALIZE_FLAG=$([[ -n "$normalize_with_std" ]] && echo "--${normalize_with_std}" || echo "")
SCALE_LR_FLAG=$([[ -n "$scale_lr_in_grad" ]] && echo "--${scale_lr_in_grad}" || echo "")
PASSATK_FLAG=$([[ -n "$pass_at_k" ]] && echo "--${pass_at_k}" || echo "")
ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

export WANDB_DIR="${REPO_ROOT}/wandb"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

echo "Cleaning up /dev/shm..."
rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true

echo "Starting local Ray cluster..."
ray stop > /dev/null 2>&1 || true
ray start --head --port=6379 --dashboard-host=0.0.0.0

RUN_NAME="${name_prefix}_$(date +%s)"
echo "---------------------------------"
echo "Run name:           $RUN_NAME"
echo "Target GPU:         $GPU_DEVICES"
echo "Model:              $model_name"
echo "Task:               $task"
echo "Reward:             $reward_variant (λ=$lambda_cal)"
echo "Pop×Prompt×Sample:  $population_size × $prompt_batch_size × $samples_per_prompt"
echo "lora_r:             $lora_r"
echo "temperature:        $temperature (down from 1.0)"
echo "normalize_w_std:    ${normalize_with_std:-OFF}"
echo "scale_lr_in_grad:   ${scale_lr_in_grad:-OFF}"
echo "max_tokens:         $max_tokens"
echo "Iterations:         $num_iterations"
echo "---------------------------------"

TRAIN_START_EPOCH=$(date +%s)

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
    --lambda-cal "$lambda_cal" \
    --instance-weight "$instance_weight" \
    $ENABLE_THINKING_FLAG \
    $NORMALIZE_FLAG \
    $SCALE_LR_FLAG \
    --prompt-batch-size "$prompt_batch_size" \
    --samples-per-prompt "$samples_per_prompt" \
    --temperature "$temperature" \
    $PASSATK_FLAG \
    --steps-per-eval "$steps_per_eval" \
    $DATASET_SIZE_CMD \
    --name-prefix "$RUN_NAME" \
    --checkpoint-dir "${REPO_ROOT}/checkpoints" \
    --use-wandb

EXIT_CODE=$?
TRAIN_END_EPOCH=$(date +%s)
TRAIN_WALL_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))

N_GPUS=$(python - <<PY
import torch
print(torch.cuda.device_count())
PY
)
TRAIN_GPU_HOURS=$(python - <<PY
print(round(${TRAIN_WALL_SECONDS} / 3600.0 * ${N_GPUS}, 3))
PY
)

GPU_HOURS_FILE="${REPO_ROOT}/logs/${RUN_NAME}_gpu_hours.txt"
cat > "$GPU_HOURS_FILE" <<EOF
run_name=${RUN_NAME}
train_wall_seconds=${TRAIN_WALL_SECONDS}
n_gpus=${N_GPUS}
train_gpu_hours=${TRAIN_GPU_HOURS}
exit_code=${EXIT_CODE}
note=Diagnostic v3 (Path A) with normalize_with_std+scale_lr_in_grad OFF, temp=0.3.
EOF

echo "Python script finished with exit code $EXIT_CODE"
echo "Training wall-clock: ${TRAIN_WALL_SECONDS}s on ${N_GPUS} GPU(s) = ${TRAIN_GPU_HOURS} GPU-hours"

ray stop || true

exit "$EXIT_CODE"
