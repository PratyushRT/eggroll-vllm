# Questions for the EGGROLL authors

**Status:** most of the originally-drafted questions were answered by reading
the paper (arXiv:2511.16652) and the repo directly. This doc has been pruned
to the genuine remaining unknowns. See **"Answered"** section at the bottom
for the resolved questions and the sources.

Context: we're running ES on `eggroll-vllm` for the **ESvPG paper** — comparing
ES against DCPO / GRPO / RLCR on a calibrated-reasoning task (Qwen3-8B,
DeepScaler40k training set, MATH-500 / AIME / AMC eval suite). Paper-grade
compute is GPU-hours-matched on 1×H100 (primary axis per EGGROLL §G.3) with
total completions matched against DCPO's ~245k Qwen3-8B budget as a
secondary axis.

---

## Remaining open questions

### Q1. Training temperature > 0 for samples_per_prompt=1

Your launchers all use `temperature=0.0` (greedy) and `samples_per_prompt=1`.
Your code in `es_lora_multinode.py:1119-1120` asserts that
`samples_per_prompt > 1` *requires* `temperature > 0` (otherwise greedy samples
would collapse).

For reward variants that work at `samples_per_prompt=1` (DCPO-Instance, RLCR),
is there any reason to prefer non-greedy? Specifically:

- Does the extra per-token stochasticity help when the task reward itself is
  stochastic (verbalized calibration), where the model must learn to produce
  varied responses and label them with differing confidence?
- Any interaction between `temperature > 0` and `normalize_with_std` or
  `scale_lr_in_grad` that you've tested?

We'll sweep `{0.0, 0.7, 1.0}` as an ablation regardless, but a prior from
your experience would save a run.

---

### Q2. `samples_per_prompt > 1` interaction with antithetic ±σ pairs

Your code enforces `population_size % 2 == 0` (line 1115), consistent with
antithetic sampling described in §D.3 ("only 1 bit is extracted per antithetical
pair"). You haven't published an ablation on `samples_per_prompt > 1` for
reasoning tasks.

Our DCPO-Hybrid reward variant *requires* `samples_per_prompt ≥ 2` (for the
group-mean term inside the Brier penalty). Our config uses `samples_per_prompt=8`
to match DCPO's G=8.

- Do multiple rollouts per prompt × per pop-member cause over-correlation
  within an antithetical pair, which would shrink the effective antithetic
  benefit?
- Is there a regime where you've seen `samples_per_prompt > 1` cause
  gradient-estimator pathologies (e.g., NaN in `normalize_with_std`, or
  population fitness collapsing to a point mass)?
- If we wanted the minimum `samples_per_prompt` that makes the group-mean
  Brier well-defined without stressing the antithetic machinery, would you
  pick 2, 4, or 8?

---

### Q3. Open-ended

We're running ES-on-LLM in a regime your released launchers don't directly
cover (Qwen3-8B, 1×H100, calibrated-reasoning reward, ~245k-completion
budget). What's the one thing you'd want us to know that isn't in the paper
or repo?

---

## Answered from paper + code

Each of these was closed during our codebase/paper audit.

| Original question | Answer | Source |
|---|---|---|
| Population size for Qwen3-8B on 1×H100 | Pop is bottlenecked by **parallel-gen capacity per GPU** (~1024 concurrent gens/GPU at 7B, seq len 1000), NOT by model weight count. At `max_tokens=3000` this halves. Our `pop × prompt × samples = 1024` is the correct landing zone. | Paper Tables 26/28; repo launchers |
| LR vs. population-size rule | **Hand-tuned via linear search** around a √pop backbone. Their ηscale follows pop^~0.2 empirically. With `scale_lr_in_grad=ON`, effective step ∝ `lr/(√pop·σ)`. Matching their pop=256 anchor (lr=1e-3, flag ON) is principled. | Paper §G.4; `es_lora_multinode.py:340-343` |
| Compute-fairness axis | **Wall-clock on matched hardware → GPU-hours.** Paper explicitly calls out total-completions-without-GPU-hours as a red flag (§G.3 line 2941). We now report both axes, primary = GPU-hours. | Paper §G.3, Fig. 5 caption |
| `scale_lr_in_grad` criterion | Toggles √pop scaling on the update. **ON** at their 1-GPU pop=256 config, **OFF** at their multinode pop≥1024. Our pop=64 and pop=256 at 8B both sit in the ON regime. | `es_lora_multinode.py:340-343` + launcher comparison |
| σ scaling with pop | σ hand-bumped 2.86× when pop scaled 16× (~pop^0.27). All their launchers use σ=1e-3 regardless; σ=1e-3 is consistent for our pop ∈ {64, 256}. | Paper §G.3 line 3117 |
| Group-structure analog in ES | Paper §6.3: ES's "group" is z-scoring across population members per question with global variance, NOT across per-prompt rollouts. So DCPO's G=8 maps onto `population_size`, not `samples_per_prompt`. | Paper §6.3 lines 677-688 |
| `lora_r=1` rationale | Code constraint: adapter VRAM × pop must fit on 1 GPU. Rank 1 leaves room for pop=256 at 4B; paper never ablates higher ranks. We default to 1. | Repo launcher consistency |
| `steps_per_adapter=4` | Default across every config. No ablation published. We keep at 4. | Launcher consistency |
| Training temperature = 0 | Tables 26/28 confirm ES always trained greedy; GRPO baselines trained at temp=1.0. This is the first real DCPO-vs-EGGROLL knob disagreement. | Paper Tables 26/28 |
| Checkpoint selection | No published best-checkpoint rule. We default to last checkpoint, will report mid-training eval as rows in the results table. | Inferred from code |

---

## Our final configuration (post-audit)

| Knob | Debug | Paper Hybrid | Paper Instance/RLCR | EGGROLL 1-GPU anchor |
|---|---|---|---|---|
| model | Qwen3-8B | Qwen3-8B | Qwen3-8B | Qwen3-4B |
| population_size | 64 | 64 | 256 | 256 |
| prompt_batch_size | 2 | 2 | 4 | 16 |
| samples_per_prompt | 8 | 8 | 1 | 1 |
| concurrent gens/step | 1024 | 1024 | 1024 | 4096 (but 4B model) |
| num_iterations | 50 | 240 | 240 | 300 (default) |
| total completions | 51,200 | 245,760 | 245,760 | 1.23M |
| max_tokens | 3000 | 3000 | 3000 | 1024 |
| temperature | 1.0 | 1.0 | 1.0 | 0.0 |
| lora_r | 1 | 1 | 1 | 1 |
| sigma | 0.001 | 0.001 | 0.001 | 0.001 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 |
| steps_per_adapter | 4 | 4 | 4 | 4 |
| scale_lr_in_grad | ON | ON | ON | ON |
| normalize_with_std | ON | ON | ON | ON |

Every ES-native knob (σ, LR, lora_r, steps_per_adapter, scale_lr_in_grad,
normalize_with_std) matches EGGROLL's 1-GPU anchor exactly. The two
documented deviations are **model scale** (8B vs their 4B, at matched
concurrent-gen budget) and **temperature** (1.0 vs 0.0, required by
Hybrid's samples_per_prompt=8 and kept consistent across variants for
ablation-cleanness).
