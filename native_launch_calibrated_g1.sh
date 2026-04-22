#!/bin/bash
# Native (no-docker) 1-GPU launcher for DCPO-calibrated ES training.
#
# Mirrors docker_launch_calibrated_g1.sh but assumes the user's venv is
# already active (python resolves to the venv python with vllm + ray + torch
# CUDA). Use when docker is unavailable (e.g. GCE DLVM that ships its own
# CUDA toolchain).
#
# Usage (on the VM):
#   cd ~/ESvPG/eggroll-vllm
#   source .venv/bin/activate
#   export HF_TOKEN=$(cat ~/.cache/huggingface/token)
#   export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
#   bash native_launch_calibrated_g1.sh            # run in foreground
#   # or for background:
#   nohup bash native_launch_calibrated_g1.sh > logs/train.log 2>&1 &

set -u  # fail on unset variables

# -----------------------------------------
# User-settable parameters (edit these)
# -----------------------------------------

# --- Core model + task ---
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"

# --- DCPO-style reward config ---
reward_variant="hybrid"     # hybrid | instance | rlcr
lambda_cal="0.5"
instance_weight="0.3"
enable_thinking="False"

# --- ES hyperparameters (match EGGROLL 1-GPU anchor exactly) ---
sigma="0.001"
learning_rate="0.001"
population_size="64"
steps_per_adapter="4"
lora_r="1"
num_iterations="50"         # DEBUG: plumbing validation only

# --- Rollout / batch ---
prompt_batch_size="2"
samples_per_prompt="8"
max_tokens="3000"
temperature="1.0"

# --- ES normalization toggles ---
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"
pass_at_k=""

# --- Eval ---
steps_per_eval="25"
sub_dataset_size="null"

# --- Misc ---
name_prefix="native-calibrated-debug"
GPU_DEVICES="0"
# -----------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# 1. Folders
mkdir -p checkpoints logs wandb cache/huggingface

# 2. Optional flags
DATASET_SIZE_CMD=""
if [[ "$sub_dataset_size" != "None" ]] && [[ "$sub_dataset_size" != "null" ]] && [[ -n "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

NORMALIZE_FLAG=$([[ -n "$normalize_with_std" ]] && echo "--${normalize_with_std}" || echo "")
SCALE_LR_FLAG=$([[ -n "$scale_lr_in_grad" ]] && echo "--${scale_lr_in_grad}" || echo "")
PASSATK_FLAG=$([[ -n "$pass_at_k" ]] && echo "--${pass_at_k}" || echo "")
ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

# 3. Env
export WANDB_DIR="${REPO_ROOT}/wandb"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

# 4. Shared-mem cleanup (EGGROLL uses /dev/shm for population buffers)
echo "Cleaning up /dev/shm..."
rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true

# 5. Ray cluster (local head)
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

# GPU-hours (primary fairness axis per EGGROLL §G.3)
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
note=Primary compute-fairness axis per EGGROLL paper §G.3.
EOF

echo "Python script finished with exit code $EXIT_CODE"
echo "Training wall-clock: ${TRAIN_WALL_SECONDS}s on ${N_GPUS} GPU(s) = ${TRAIN_GPU_HOURS} GPU-hours"
echo "GPU-hours report:    $GPU_HOURS_FILE"

echo "Stopping Ray..."
ray stop || true

exit "$EXIT_CODE"
