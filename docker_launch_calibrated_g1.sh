#!/bin/bash
# Single-GPU DCPO-calibrated ES training launcher.
#
# 50-step debug run on Qwen3-8B with DCPO-Hybrid reward (λ=0.5), group-mean
# calibration target across samples_per_prompt=8 rollouts. Fits on 1×H100
# (80GB) or 1×H200 (141GB). See /Users/pratyush/.claude/plans/lexical-riding-bachman.md
# for the full plan and feasibility estimates.
#
# After training, run eval_dcpo_benchmarks.py against the saved checkpoint:
#     python eval_dcpo_benchmarks.py \
#         --model-path checkpoints/<run>/checkpoint_step_50 \
#         --output-json results_step50.json \
#         --output-md results_step50.md

# -----------------------------------------
# User-settable parameters (edit these)
# -----------------------------------------

# --- Core model + task ---
model_name="Qwen/Qwen3-8B"  # EGGROLL's scripts ship Qwen3-4B configs only; 8B hyperparameters are open (see AUTHOR_QUESTIONS.md).
task="calibrated-math:deepscaler40k"

# --- DCPO-style reward config ---
reward_variant="hybrid"     # hybrid | instance | rlcr
lambda_cal="0.5"
instance_weight="0.3"
enable_thinking="False"     # Qwen3 non-thinking mode (DCPO default)

# --- ES hyperparameters ---
# Defaults follow EGGROLL's own 1-GPU launcher (docker_launch_base_g1.sh):
#   sigma=0.001, learning_rate=0.001, lora_r=1, steps_per_adapter=4,
#   normalize_with_std=ON, scale_lr_in_grad=ON.
# We halve population_size (256 -> 64) to stay within 1×H100 VRAM at 8B with
# samples_per_prompt=8 (DCPO group structure). See AUTHOR_QUESTIONS.md for the
# open question on population/LR scaling at 8B.
sigma="0.001"
learning_rate="0.001"
population_size="64"
steps_per_adapter="4"
lora_r="1"                  # EGGROLL default everywhere; was 4 in early debug
num_iterations="50"         # DEBUG: plumbing validation only (not paper-grade)

# --- Rollout / batch ---
prompt_batch_size="2"
samples_per_prompt="8"      # matches DCPO's G=8 so Hybrid reward's group mean is well-defined
max_tokens="3000"           # matches DCPO's max_response_length
# NOTE: temperature=1.0 deviates from EGGROLL's production default of 0.0.
# EGGROLL relies on population perturbations for exploration and uses greedy
# sampling; DCPO trains at 1.0 because PG needs stochastic sampling. For
# calibration we want the model to see response variance, so we keep 1.0 here.
# A paper ablation sweep over {0.0, 0.7, 1.0} is cheap and worth running.
temperature="1.0"

# --- ES normalization toggles ---
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"  # ON: matches EGGROLL's 1-GPU launcher
pass_at_k=""                         # disabled (mean-reduce across samples for Hybrid reward)

# --- Eval ---
steps_per_eval="25"
sub_dataset_size="null"

# --- Misc ---
name_prefix="docker-calibrated-debug"
GPU_DEVICES="0"
# -----------------------------------------

# 1. Generate necessary folders
mkdir -p checkpoints logs wandb cache/huggingface

# 2. Build the image (uses cache instantly if already built)
echo "Ensuring Docker image is built..."
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${USER}_eggroll .

# 3. Setup optional flags
DATASET_SIZE_CMD=""
if [[ "$sub_dataset_size" != "None" ]] && [[ "$sub_dataset_size" != "null" ]] && [[ -n "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

NORMALIZE_FLAG=$([[ -n "$normalize_with_std" ]] && echo "--${normalize_with_std}" || echo "")
SCALE_LR_FLAG=$([[ -n "$scale_lr_in_grad" ]] && echo "--${scale_lr_in_grad}" || echo "")
PASSATK_FLAG=$([[ -n "$pass_at_k" ]] && echo "--${pass_at_k}" || echo "")
ENABLE_THINKING_FLAG=$([[ "${enable_thinking}" == "True" ]] && echo "--enable-thinking" || echo "--no-enable-thinking")

# 4. Create the execution script for inside the container
cat << 'EOF' > run_inside_docker.sh
#!/bin/bash
export WANDB_DIR="/app/wandb"

echo "Cleaning up local shared memory..."
rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true

echo "Starting local Ray cluster..."
ray start --head --port=6379 --dashboard-host=0.0.0.0

echo "Starting DCPO-calibrated ES training..."
python es_lora_multinode.py \
    --sigma "$SIGMA" \
    --learning-rate "$LEARNING_RATE" \
    --max-tokens "$MAX_TOKENS" \
    --model-name "$MODEL_NAME" \
    --population-size "$POPULATION_SIZE" \
    --steps-per-adapter "$STEPS_PER_ADAPTER" \
    --lora-r "$LORA_R" \
    --num-iterations "$NUM_ITERATIONS" \
    --task "$TASK" \
    --reward-variant "$REWARD_VARIANT" \
    --lambda-cal "$LAMBDA_CAL" \
    --instance-weight "$INSTANCE_WEIGHT" \
    $ENABLE_THINKING_FLAG \
    $NORMALIZE_FLAG \
    $SCALE_LR_FLAG \
    --prompt-batch-size "$PROMPT_BATCH_SIZE" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --temperature "$TEMPERATURE" \
    $PASSATK_FLAG \
    --steps-per-eval "$STEPS_PER_EVAL" \
    $DATASET_SIZE_CMD \
    --name-prefix "$NAME_PREFIX" \
    --checkpoint-dir "/app/checkpoints" \
    --use-wandb

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE"

echo "Stopping Ray..."
ray stop || true
EOF
chmod +x run_inside_docker.sh

# 5. Launch the Docker container
CONTAINER_NAME="${USER}_${name_prefix}_$(date +%s)"

echo "---------------------------------"
echo "Launching Job: $CONTAINER_NAME"
echo "Target GPU:    $GPU_DEVICES"
echo "Model:         $model_name"
echo "Task:          $task"
echo "Reward:        $reward_variant (λ=$lambda_cal)"
echo "Pop × Prompts × Samples = $population_size × $prompt_batch_size × $samples_per_prompt"
echo "---------------------------------"

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus "\"device=$GPU_DEVICES\"" \
    --shm-size=64g \
    -v $(pwd):/app \
    -e SIGMA="$sigma" \
    -e LEARNING_RATE="$learning_rate" \
    -e MAX_TOKENS="$max_tokens" \
    -e MODEL_NAME="$model_name" \
    -e POPULATION_SIZE="$population_size" \
    -e STEPS_PER_ADAPTER="$steps_per_adapter" \
    -e LORA_R="$lora_r" \
    -e NUM_ITERATIONS="$num_iterations" \
    -e TASK="$task" \
    -e REWARD_VARIANT="$reward_variant" \
    -e LAMBDA_CAL="$lambda_cal" \
    -e INSTANCE_WEIGHT="$instance_weight" \
    -e ENABLE_THINKING_FLAG="$ENABLE_THINKING_FLAG" \
    -e PROMPT_BATCH_SIZE="$prompt_batch_size" \
    -e SAMPLES_PER_PROMPT="$samples_per_prompt" \
    -e TEMPERATURE="$temperature" \
    -e STEPS_PER_EVAL="$steps_per_eval" \
    -e NAME_PREFIX="$name_prefix" \
    -e NORMALIZE_FLAG="$NORMALIZE_FLAG" \
    -e SCALE_LR_FLAG="$SCALE_LR_FLAG" \
    -e PASSATK_FLAG="$PASSATK_FLAG" \
    -e DATASET_SIZE_CMD="$DATASET_SIZE_CMD" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    ${USER}_eggroll /app/run_inside_docker.sh > "logs/${CONTAINER_NAME}.log" 2>&1

echo "Job launched in background."
echo "View stdout logs:  tail -f logs/${CONTAINER_NAME}.log"
echo "View docker logs:  docker logs -f $CONTAINER_NAME"
