#!/bin/bash
# DIAGNOSTIC 1-GPU launcher — post-author-feedback config sweep.
#
# After our first 50-step debug run regressed (gsm8k 57%→8%, prop_truncated 6%→40%),
# we asked the EGGROLL author directly. Their key feedback:
#   - For 8B, pop=256 was their **conservative floor**, pop=512 recommended.
#     Our pop=64 was 4× below the minimum they ever tested.
#   - lr=1e-3, σ=1e-3 is confirmed for 8B at pop=512.
#   - scale_lr_in_grad + normalize_with_std: always ON (except on base models).
#   - Their native reward zeroes out truncated rollouts — we ALREADY do this
#     (tasks.py:478). The truncation penalty isn't missing; the signal-to-noise
#     is just too low at pop=64 × lora_r=1 on an 8B model.
#
# This diagnostic sweep is SHORT (15 steps, no eval beyond step-0) to verify
# the new config produces a healthy training dynamic BEFORE we commit to a
# paper-grade 240-step run.
#
# Key changes from docker_launch_calibrated_g1.sh / native_launch_calibrated_g1.sh:
#   - population_size 64 → 256    (author's 8B 1-GPU anchor)
#   - lora_r 1 → 4                (4× more trainable params; rank 1 on 8B is too low)
#   - prompt_batch_size 2 → 1     (keep completions/step ≈ current at 1024)
#   - samples_per_prompt 8 → 4    (still ≥2 for Hybrid group mean, half variance cost)
#   - max_tokens 3000 → 2048      (reduce truncation floor, faster iterations)
#   - num_iterations 50 → 15      (diagnostic: confirm direction, not converge)
#   - steps_per_eval 25 → 999     (skip in-run eval; we'll run eval_dcpo_benchmarks.py after)
#
# Total completions/step: 256 × 1 × 4 = 1024 (same as previous run)
# Expected step time: ~3-5 min (similar, smaller max_tokens offsets pop adapter cost)
# Expected wall-clock: 15 × 4 min + 1 × 7 min eval @ step 0 = ~1.2 hr
# Rough cost on spot H100: ~$4.40

set -u

# -----------------------------------------
# User-settable parameters
# -----------------------------------------

# --- Core model + task ---
model_name="Qwen/Qwen3-8B"
task="calibrated-math:deepscaler40k"

# --- DCPO-style reward config ---
reward_variant="hybrid"
lambda_cal="0.5"
instance_weight="0.3"
enable_thinking="False"

# --- ES hyperparameters (post-author-feedback, 8B 1-GPU floor) ---
sigma="0.001"
learning_rate="0.001"
population_size="256"       # ← matches author's 8B anchor exactly (their conservative floor).
steps_per_adapter="4"
lora_r="1"                  # ← CRITICAL: r=1 uses torch.mm path in apply_lora_es_update (L326-330);
                            #   r≥2 uses torch.bmm which materializes [pop, out_dim, in_dim] tensor
                            #   (pop × 64MB per q_proj layer!). All three prior r≥2 runs OOMed at L333
                            #   regardless of pop. Match author's anchor (r=1 + pop=256) for the fast path.
num_iterations="15"         # ← was 50; short diagnostic

# --- Rollout / batch ---
prompt_batch_size="1"       # ← was 2; with larger pop we want fewer prompts per member
samples_per_prompt="4"      # ← was 8; still ≥2 for Hybrid, half variance overhead
max_tokens="2048"           # ← was 3000; reduce truncation floor

# Temperature must stay > 0 for samples_per_prompt > 1 (asserted at es_lora_multinode.py:1119)
temperature="1.0"

# --- ES normalization toggles ---
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"
pass_at_k=""

# --- Eval ---
steps_per_eval="999"        # ← was 25; skip in-run eval for faster diagnostic
sub_dataset_size="null"

# --- Misc ---
name_prefix="native-calibrated-diag-pop256-r1"
GPU_DEVICES="0"
# -----------------------------------------

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
note=Diagnostic run at author-recommended 8B 1-GPU config (pop=256, lora_r=4).
EOF

echo "Python script finished with exit code $EXIT_CODE"
echo "Training wall-clock: ${TRAIN_WALL_SECONDS}s on ${N_GPUS} GPU(s) = ${TRAIN_GPU_HOURS} GPU-hours"

ray stop || true

exit "$EXIT_CODE"
