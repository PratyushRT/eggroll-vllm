#!/bin/bash
# DIAGNOSTIC v2 — halved learning rate to damp oscillations.
#
# The pop=256 r=1 diagnostic run validated:
#   - config fits in 80GB H100 (r=1 uses torch.mm fast path at L326-330)
#   - truncation IS controlled (prop_truncated stabilised <5% from step 5 onward)
#   - max-member fitness hits 0.99+ repeatedly → valid directions exist
# But mean fitness oscillated wildly (−0.4 ↔ +0.99), textbook "step size too large".
# Step 6 peaked at 0.991 (> step-0 baseline 0.829) but step 7 crashed back to 0.014.
#
# Analysis: with scale_lr_in_grad + normalize_with_std ON, effective per-dim step is
#   lr × sqrt(pop) / (pop × sigma) = 1e-3 × 16 / (256 × 1e-3) = 6.3e-5
# vs author's pop=512 anchor: 1e-3 × 22.6 / (512 × 1e-3) = 4.4e-5
# → our step was 41% larger per-dim. Combined with deepscaler40k being harder
# than whatever they tuned on, overshoot is plausible.
#
# This run: halve lr (1e-3 → 5e-4) → effective step 3.2e-5, 27% *below* author anchor.
# Overcorrect intentionally to isolate step-size as the variable.
#
# Expected signal if step-size was the culprit:
#   - mean fitness monotone or near-monotone
#   - max fitness still ≥ 0.99 (exploration preserved)
#   - prop_truncated stays ≤ 5% throughout

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
learning_rate="0.0005"      # ← HALVED from 1e-3 to damp oscillations
population_size="256"
steps_per_adapter="4"
lora_r="1"                  # keep r=1 for torch.mm fast path at L326-330
num_iterations="15"

# --- Rollout / batch ---
prompt_batch_size="1"
samples_per_prompt="4"
max_tokens="2048"
temperature="1.0"

# --- ES normalization toggles ---
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"
pass_at_k=""

# --- Eval ---
steps_per_eval="999"
sub_dataset_size="null"

# --- Misc ---
name_prefix="native-calibrated-diag-pop256-r1-lr5e4"
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
echo "Run name:      $RUN_NAME"
echo "Target GPU:    $GPU_DEVICES"
echo "Model:         $model_name"
echo "Task:          $task"
echo "Reward:        $reward_variant (λ=$lambda_cal)"
echo "Pop × Prompts × Samples = $population_size × $prompt_batch_size × $samples_per_prompt"
echo "lora_r:        $lora_r"
echo "learning_rate: $learning_rate (halved from 1e-3)"
echo "max_tokens:    $max_tokens"
echo "Iterations:    $num_iterations"
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
note=Diagnostic v2 with halved lr (5e-4) to damp oscillations from pop256_r1 run.
EOF

echo "Python script finished with exit code $EXIT_CODE"
echo "Training wall-clock: ${TRAIN_WALL_SECONDS}s on ${N_GPUS} GPU(s) = ${TRAIN_GPU_HOURS} GPU-hours"

ray stop || true

exit "$EXIT_CODE"
