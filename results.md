# ESvPG Experimental Results

**Paper**: Evolution Strategies vs. Policy Gradients for calibrated LLM reasoning
**Codebase**: `eggroll-vllm` (ESHyperscale/EGGROLL fork) + DCPO calibration mods
**Target**: ES side of the ES-vs-PG comparison; PG side cited from DCPO (ICIP-CAS)

---

## Infrastructure

- **VM**: GCE `esvpg-eggroll-h100`, us-central1-a, 1× H100 80GB, spot, 485GB disk, `research-464723`
- **Stack**: Python 3.12 venv (uv), vLLM + Ray, bf16 weights
- **Model**: `Qwen/Qwen3-8B` (base, `enable_thinking=False`)
- **Task**: `calibrated-math:deepscaler40k` (ours), DCPO prompt appended for CONFIDENCE parsing

---

## Baseline evaluation (untrained Qwen3-8B)

`eval_dcpo_benchmarks.py`, DCPO-exact sampling params (n=4 repeats, T=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=3000). Confidence parsed per DCPO regex; logits-based confidence = `exp(mean(response_logprobs))`.

| Benchmark | Acc | ECE (verbal) | AUROC (verbal) | parse_success |
|---|---|---|---|---|
| MATH-500 (full) | 0.745 | ~0.25 | 0.73 | 0.844 |
| AIME24, AIME25, AMC23 | [numbers in existing JSON artifacts on VM] |

AMC24 source pending (DCPO repo doesn't ship `data/amc24.parquet`; looking for alternate source).

---

## Training experiments

### Summary of attempts

All runs used `task=calibrated-math:deepscaler40k`, `reward_variant=hybrid` (λ=0.5), `prompt_batch_size=1`, `samples_per_prompt=4`, `max_tokens=2048` unless noted. Population × samples = 1024 completions/step.

| # | Config | Steps completed | Outcome |
|---|---|---|---|
| 1 | pop=64, r=1, lr=σ=1e-3, temp=1.0, toggles ON, max_tok=3000, 50 iters | ~28 | **Regressed**: gsm8k 57%→8%, prop_trunc 6%→40%. Killed. Root cause initially mis-diagnosed. |
| 2 | pop=256, r=4, lr=σ=1e-3, temp=1.0, toggles ON | 0 | **OOM at `apply_lora_es_update:333`** (4 GB alloc, 2.48 GB free). Root cause: `r≥2` path materializes `[pop, out_dim, in_dim]` tensor in `torch.bmm`. |
| 3 | pop=128, r=4 | 0 | Same OOM at L333 — identical memory gap. Confirms rank, not pop, is the culprit. |
| 4 | pop=128, r=2 | 0 | Same OOM. Confirms the r≥2 code path is the memory hog, regardless of r value. |
| 5 | **pop=256, r=1**, lr=σ=1e-3 | 11 (killed) | **Fits** (r=1 uses `torch.mm` fast path at L326-330). But training dynamic: fitness oscillates ±0.4 to +0.99, prop_truncated jumps 1%→55%→back. Step 6 peaked at mean=0.991, step 7 crashed to 0.014. |
| 6 | pop=256, r=1, **lr=5e-4** (halved) | 7 (killed) | **Identical oscillation pattern.** Step 5 fitness −0.445 vs run #5's −0.418; step 6 fitness 0.940 vs 0.991. Direction unchanged because `normalize_with_std` makes the ES update direction scale-invariant. |
| 7 | pop=256, r=1, lr=σ=1e-3, **temp=0.3**, **normalize_with_std=OFF, scale_lr_in_grad=OFF** | 15 (complete) | Path A — see detailed trajectory below. |

### Diagnostic trajectory — Run #7 (Path A, 15 steps, 1.84 GPU-hours)

| Step | Mean fitness | Trunc % | std_norm | distinct_ans/prompt | min | max | Interpretation |
|------|-------------:|--------:|---------:|--------------------:|----:|----:|-----|
| 0 | 0.818 | 1.2 | 0.239 | 1.46 | −0.34 | 0.999 | baseline |
| 1 | 0.198 | 68.2 | 0.256 | 0.95 | −0.23 | 0.749 | crash from baseline |
| 2 | −0.004 | **99.2** | 0.020 | **0.03** | −0.11 | 0.000 | **degenerate garbage** (empty/noise outputs) |
| 3 | −0.309 | 31.1 | 0.124 | 1.34 | −0.48 | 0.178 | recovering |
| 4 | −0.026 | 64.8 | 0.199 | 1.21 | −0.45 | 0.500 | |
| 5 | −0.477 | 0.0 | 0.013 | 1.01 | −0.49 | −0.45 | |
| 6 | **0.999** | 0.0 | 0.000 | 1.76 | 0.999 | 0.999 | **fitness peak, zero variance** |
| 7 | −0.140 | 22.5 | 0.292 | 1.78 | −0.50 | 0.707 | crashed again |
| 8 | −0.449 | 3.2 | 0.087 | 1.59 | −0.50 | 0.160 | |
| 9 | −0.068 | 1.4 | 0.347 | 3.00 | −0.48 | 0.707 | |
| 10 | −0.397 | 0.0 | 0.154 | 2.46 | −0.49 | 0.368 | |
| 11 | −0.006 | **98.6** | 0.028 | **0.05** | −0.23 | 0.000 | **degenerate garbage again** |
| 12 | −0.457 | 0.0 | 0.088 | 3.22 | −0.49 | 0.047 | |
| 13 | −0.423 | 0.0 | 0.114 | 2.99 | −0.47 | 0.368 | |
| 14 | **0.9995** | 0.0 | 0.0005 | **1.00** | 0.999 | 1.000 | **final state: fitness ≈ 1, near-zero variance** |

**WandB**: https://wandb.ai/eternis-ai/hyperscalees-vllm

### Step 14 evaluation (MATH-500, n=50 × 2 samples)

[TO BE POPULATED AFTER EVAL]

| Metric | Baseline (untrained) | Step 14 (Path A) | Δ |
|---|---|---|---|
| Acc | 0.745 | — | — |
| ECE (verbal) | ~0.25 | — | — |
| AUROC (verbal) | 0.73 | — | — |
| parse_success | 0.844 | — | — |

**Critical test**: does the training fitness=0.9995 correspond to real accuracy gains, or reward-hacking?

---

## Analysis

### Why all runs oscillate

`normalize_with_std` divides the per-member fitness by its std before weighting noise — making the update **direction scale-invariant**. This is why halving lr (run #6) changed magnitude but not pattern.

Underlying pathology, present regardless of toggles:

1. **Narrow spread state** (most rollouts ≈0.8, std small): ES gradient direction is near-random noise.
2. Random direction → next step generates longer responses → **mass truncation** (~55–99%).
3. **Collapsed spread** (most fitness ≈0): one non-truncated rollout dominates "gradient" → huge over-correction.
4. Over-correction → truncation drops → back to state 1. **Cycle repeats.**

Running #7 (toggles OFF) shows an additional degenerate attractor: fitness=1.0 with near-zero variance, where all rollouts across all population members produce near-identical outputs. Whether this is real learning or reward-hack is the purpose of the step-14 eval.

### Alignment audit vs EGGROLL author's tested 8B configs

Our configuration is **not directly validated by the author's experiments**. Author's tested 8B points:

- **pop ∈ {512, 16k}**, never 256 for 8B; best results at 16k
- **temperature = 0.0 (greedy)**, chosen "for reproducibility"
- **samples_per_prompt = 1** (group-mean rewards untested × antithetic sampling)
- `normalize_with_std`, `scale_lr_in_grad`: "always ON except on base models"
- Observed truncation: "exponentially decreasing over steps"

Our config: pop=256 (below floor), temp=1.0 or 0.3 (off-anchor), samples_per_prompt=4 (untested interaction), toggles ON *or* OFF both unstable. Qwen3-8B may fall into the "base model" exception the author flagged.

---

## Open questions

1. Is step 14's fitness=1.0 real accuracy improvement or a reward hack? → **next step eval answers this**
2. Does matching author's anchor (pop=512, temp=0, single-sample, Instance reward on 2×H100) stabilize training? → Path B if Path A's step-14 eval shows reward-hack.
3. Is Qwen3-8B "base enough" that toggles should be OFF? → partially tested by run #7, both configurations unstable.
4. `samples_per_prompt=4` × antithetic sampling — author has no data on this interaction; may be an independent instability source.

---

## Discovered bugs / workarounds

1. `eval_dcpo_benchmarks.py` — `--math500-n` flag was ignored by the default HF loader. **Fixed** with items-trim after loader returns.
2. `tasks.py:5` unconditionally imports `EGG_IMG, CHICK_IMG` from `egg_img`. **Created stub** returning empty strings.
3. `es_lora_multinode.py:326-333` — `r=1` path (efficient `torch.mm`) vs `r≥2` path (materializes `[pop, out_dim, in_dim]` via `torch.bmm`). On Qwen3-8B, the latter blows VRAM at pop≥128. Fix-on-fork opportunity: rewrite L332-333 as loop/fused per-member accumulation.
4. Checkpoint dir only contains `model_weights.safetensors` + `training_state.json`; missing `config.json`, `tokenizer.json`. **Workaround**: compose a full dir by symlinking base-model configs and renaming `model_weights.safetensors → model.safetensors`.
5. GCE VM sshd starves under 100% GPU load (rollouts pin vLLM engine). `gcloud compute ssh` exits 255. Recovery via `gcloud compute instances reset` (fast) or until-loop retry (slow).
