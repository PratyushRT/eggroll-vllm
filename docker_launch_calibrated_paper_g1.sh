#!/bin/bash
# Single-GPU DCPO-calibrated ES training launcher — PAPER-GRADE config.
#
# ---------------------------------------------------------------------------
# Fairness protocol (critical — EGGROLL paper §G.3, arXiv:2511.16652)
# ---------------------------------------------------------------------------
# The EGGROLL authors compare against PG baselines on **wall-clock time at
# matched hardware**, not total completions. Figure 5 x-axes are literally
# "Relative wall-clock time (hours)"; §G.3 line 2941 calls out a prior paper
# for NOT reporting wall-clock/hardware as "difficult to establish a fair
# comparison." Matching total-completions alone silently hands ES a large
# wall-clock advantage because EGGROLL gets ~1024 parallel gens/GPU vs PG's
# ~32 at the same model scale.
#
# We therefore report BOTH axes:
#   1. Total completions (DCPO-matched budget: ~245,760)
#   2. GPU-hours on 1×H100/H200 (measured at run time; logged to WandB
#      via the runner's built-in timer + written to train_stats_gpu_hours.txt)
#
# The paper narrative will quote GPU-hours as the primary axis and total
# completions as secondary. Both numbers are reproduced from WandB and the
# log file produced by `run_inside_docker.sh` below.
# ---------------------------------------------------------------------------
#
# Compute targets:
#   DCPO Qwen3-8B paper:  batch=256, G=8, ~120 steps  -> ~245,760 completions
#                                                         ~737M completion tokens
#                                                         published wall-clock not
#                                                         reported, but using 8×H100
#                                                         for ~24h is typical.
#   ES (this launcher):   pop × prompt × samples × iter ≈ same completion budget,
#                                                         1×H100 for ~12–15h expected.
#
# REWARD_VARIANT presets (all hit 1024 completions/step × 240 iter = 245,760):
#
#   1. hybrid    : pop=64, prompt_batch=2, samples_per_prompt=8, 240 iter
#                  (Hybrid reward REQUIRES samples≥2 for its group mean;
#                   confirmed by es_lora_multinode.py:1278-1281 warning.)
#   2. instance  : pop=256, prompt_batch=4, samples_per_prompt=1, 240 iter
#                  (ES-native shape per EGGROLL's own 1-GPU launcher.)
#   3. rlcr      : pop=256, prompt_batch=4, samples_per_prompt=1, 240 iter
#                  (Same as instance; RLCR reward doesn't need groups.)
#
# Landing-zone rationale for 1×H100 80GB, max_tokens=3000:
#   Paper Tables 26/28 report 1024-1536 parallel gens/GPU at seq len 1000.
#   Our seq len 3000 shrinks this 3-6×, so ~1024 concurrent generations is
#   the ragged-edge VRAM target. All three presets fit this envelope exactly.
#
# Expected wall-clock on 1×H100 80GB: ~12–15h at max_tokens=3000.
# On 1×H200 141GB: same, with headroom to raise population_size.
#
# After training, run eval_dcpo_benchmarks.py against the final checkpoint:
#     python eval_dcpo_benchmarks.py \
#         --model-path checkpoints/<run>/checkpoint_step_240 \
#         --output-json results_paper.json \
#         --output-md results_paper.md

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
enable_thinking="False"     # Qwen3 non-thinking mode (DCPO default)

# --- ES hyperparameters (EGGROLL 1-GPU defaults) ---
sigma="0.001"
learning_rate="0.001"       # EGGROLL 1-GPU default; may need retune for pop=256 at 8B (see AUTHOR_QUESTIONS.md)
steps_per_adapter="4"
lora_r="1"                  # EGGROLL default everywhere

# --- Compute budget ---
# 240 iterations × (pop × prompt × samples) = ~245k completions (DCPO-matched)
num_iterations="240"
steps_per_eval="60"         # eval 4× through training (steps 60/120/180/240)
save_freq="60"              # checkpoint at same cadence as eval

# --- Reward-variant-dependent rollout shape ---
# Auto-select pop × prompt × samples based on reward_variant to hit ~245k completions
if [[ "$reward_variant" == "hybrid" ]]; then
    # Group-mean reward needs samples_per_prompt ≥ 2; use DCPO's G=8
    population_size="64"
    prompt_batch_size="2"
    samples_per_prompt="8"
else
    # instance / rlcr: no group dependency → ES-native shape (larger pop)
    population_size="256"
    prompt_batch_size="4"
    samples_per_prompt="1"
fi
# All three configs: 64*2*8 = 256*4*1 = 1024 completions/step
# 1024 × 240 = 245,760 completions — matches DCPO Qwen3-8B budget.

max_tokens="3000"           # matches DCPO max_response_length
# NOTE: temperature=1.0 is DCPO training temp. EGGROLL's own default is 0.0
# (ES gets exploration from population perturbations, not sampling). Ablation
# sweep over {0.0, 0.7, 1.0} is cheap and tracked as an open question.
temperature="1.0"

# --- ES normalization toggles ---
normalize_with_std="normalize-with-std"
scale_lr_in_grad="scale-lr-in-grad"  # ON: matches EGGROLL's 1-GPU launcher; OPEN Q for pop=256 at 8B
pass_at_k=""                         # disabled (mean-reduce across samples)

# --- Data ---
sub_dataset_size="null"

# --- Misc ---
name_prefix="docker-calibrated-paper-${reward_variant}"
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

# --- GPU-hours fairness reporting ---
# Paper compares ES vs PG on wall-clock × GPU count, not total completions.
# Record start time and compute elapsed GPU-hours on exit.
TRAIN_START_EPOCH=$(date +%s)
NUM_GPUS_VISIBLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Train start: $(date -u +%Y-%m-%dT%H:%M:%SZ)  |  visible GPUs: $NUM_GPUS_VISIBLE" \
     | tee /app/logs/${NAME_PREFIX}_gpu_hours.txt

echo "Starting DCPO-calibrated ES training (paper-grade)..."
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
    --save-freq "$SAVE_FREQ" \
    $DATASET_SIZE_CMD \
    --name-prefix "$NAME_PREFIX" \
    --checkpoint-dir "/app/checkpoints" \
    --use-wandb

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE"

# --- GPU-hours fairness report ---
TRAIN_END_EPOCH=$(date +%s)
ELAPSED_SEC=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))
ELAPSED_HR=$(awk "BEGIN {printf \"%.3f\", $ELAPSED_SEC/3600.0}")
GPU_HOURS=$(awk "BEGIN {printf \"%.3f\", ($ELAPSED_SEC/3600.0) * $NUM_GPUS_VISIBLE}")
{
  echo "Train end:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Elapsed:     ${ELAPSED_SEC}s  (${ELAPSED_HR}h wall-clock)"
  echo "GPU-hours:   ${GPU_HOURS}  (wall-clock × $NUM_GPUS_VISIBLE GPUs)"
  echo "Exit code:   ${EXIT_CODE}"
} | tee -a /app/logs/${NAME_PREFIX}_gpu_hours.txt

echo "Stopping Ray..."
ray stop || true
EOF
chmod +x run_inside_docker.sh

# 5. Launch the Docker container
CONTAINER_NAME="${USER}_${name_prefix}_$(date +%s)"

TOTAL_COMPLETIONS=$((population_size * prompt_batch_size * samples_per_prompt * num_iterations))

echo "---------------------------------"
echo "Launching PAPER-GRADE Job: $CONTAINER_NAME"
echo "Target GPU:        $GPU_DEVICES"
echo "Model:             $model_name"
echo "Task:              $task"
echo "Reward:            $reward_variant (λ=$lambda_cal)"
echo "Rollout shape:     pop=$population_size × prompts=$prompt_batch_size × samples=$samples_per_prompt = $((population_size * prompt_batch_size * samples_per_prompt)) completions/step"
echo "Iterations:        $num_iterations"
echo ""
echo "--- Fairness axes reported (see paper §G.3) ---"
echo "  Completions budget (DCPO-matched): $TOTAL_COMPLETIONS  (DCPO target ~245,760)"
echo "  GPU-hours: measured at run end, written to logs/${name_prefix}_gpu_hours.txt"
echo "  Primary axis for paper: GPU-hours. Secondary: completions."
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
    -e SAVE_FREQ="$save_freq" \
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
