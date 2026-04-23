# A1 / A2 Triage Run Analysis — `rlcr_hybrid_loo_retention`

Two 40-step ES triage runs on `Qwen/Qwen3-8B` with the new anchor-retention reward
variant, differing only in the LOO-target mixing coefficient `ρ`.

| | **A1** | **A2** |
|---|---|---|
| Role | LOO-blended target | Strict per-sample Brier |
| `ρ` | **0.8** | **1.0** |
| VM | `esvpg-eggroll-h100-e5a-2` | `esvpg-eggroll-h100-e5a` |
| Zone | `us-east5-a` | `us-east5-a` |
| Run name | `triage-A1-rho08-s0.0005-lr0.0005-1776873821` | `triage-A2-rho10-s0.0005-lr0.0005-1776873944` |
| Start (UTC) | 2026-04-22 16:03 | 2026-04-22 16:06 |
| End (UTC) | 2026-04-22 21:02 | 2026-04-22 21:06 |
| Wall clock | **~4.92 h** (17,703 s, sum of per-step `TIMES: total`) | **~4.92 h** (17,723 s) |
| Median step time | 438.9 s | 437.8 s |
| Final checkpoint | `checkpoint_step_39/model_weights.safetensors` (16.38 GB) | `checkpoint_step_39/model_weights.safetensors` (16.38 GB) |
| GCS prefix | `gs://esvpg-experiments/es_exp/triage_A1_1776873821/` | `gs://esvpg-experiments/es_exp/triage_A2_1776873944/` |
| Exit status | Clean completion, GCS upload succeeded | Clean completion, GCS upload succeeded |

---

## 1. Reward formula (both runs)

```
r_ijk = C
      − λ_cal · (q − T)²                         # calibration, T = ρ·C + (1−ρ)·C̄_{-i,j}
      − λ_ret · A · (1 − C)                       # retention penalty (anchors only)
      − λ_wc  · A · (1 − C) · q²                  # wrong-conf-anchor penalty (anchors only)
      + γ_fmt · valid_fmt
      − γ_bad · invalid_fmt
      − λ_trunc · truncated
```

where
- `C ∈ {0,1}` = correctness from DCPO `is_equiv` grader
- `q ∈ [0,1]` = parsed `<conf>...</conf>` verbal confidence (0.0 if unparseable)
- `A ∈ {0,1}` = is the prompt in the pre-computed base-solved anchor set
- `C̄_{-i,j}` = leave-one-out mean correctness across the population, same prompt
- `T` = calibration target; `ρ=1` is strict per-sample Brier, `ρ<1` blends in the group's view

After per-sample scoring, fitness is **per-prompt-centered** with a **global-std floor**:
`F_population ← (F − mean_per_prompt) / max(global_std, 0.05)`, clipped to `±3.0`.

---

## 2. Exact configuration

Both runs used the **same** launcher script (`native_launch_triage_A1A2.sh`) with the
`A1` or `A2` positional arg as the only caller-level difference.

### Reward / calibration

| Knob | A1 | A2 |
|---|---|---|
| `reward_variant` | `rlcr_hybrid_loo_retention` | `rlcr_hybrid_loo_retention` |
| `prompt_template` | `conf_tags` (`<conf>…</conf>`) | `conf_tags` |
| `lambda_cal` (λ for `(q−T)²`) | 1.0 | 1.0 |
| **`rho`** (LOO mix) | **0.8** | **1.0** |
| `lambda_retention` (anchor wrong-answer penalty) | 0.20 | 0.20 |
| `lambda_wrong_conf_anchor` (anchor high-conf-when-wrong penalty) | 0.50 | 0.50 |
| `lambda_trunc` | 0.25 | 0.25 |
| `gamma_fmt` (valid-format bonus) | 0.02 | 0.02 |
| `gamma_bad` (invalid-format penalty) | 1.00 | 1.00 |
| `anchor_frac` (fraction of each batch that must be anchors) | 0.50 | 0.50 |
| Anchor set | `data/anchor_set_deepscaler40k_seed42.json` (1556 anchors / 4000 scanned, threshold 0.75, K=4) | same |

### ES / optimization

| Knob | A1 | A2 |
|---|---|---|
| `sigma` | 0.0005 | 0.0005 |
| `learning_rate` | 0.0005 | 0.0005 |
| `global_std_floor` | 0.05 | 0.05 |
| `fitness_clip` | 3.0 | 3.0 |
| `population_size` | 256 | 256 |
| `lora_r` / `lora_alpha` | 1 / 1 | 1 / 1 |
| `steps_per_adapter` (LoRA reuse) | 4 | 4 |
| `normalize_with_std` | **false** | false |
| `scale_lr_in_grad` | **false** | false |
| `per_prompt_normalize` | **true** | true |
| `cosine_decay_sigma_final` | 0.0 (disabled) | 0.0 |
| `cosine_decay_lr_final` | 0.0 (disabled) | 0.0 |

### Generation / batching

| Knob | A1 | A2 |
|---|---|---|
| `model_name` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-8B` |
| `task` | `calibrated-math:deepscaler40k` | `calibrated-math:deepscaler40k` |
| `prompt_batch_size` | 4 | 4 |
| `samples_per_prompt` | 1 | 1 |
| `max_tokens` | 2048 | 2048 |
| `temperature` | 0.7 | 0.7 |
| `enable_thinking` | false | false |
| `base_seed` | 42 | 42 |

### Schedule / eval

| Knob | A1 | A2 |
|---|---|---|
| `num_iterations` | 40 | 40 |
| `steps_per_eval` | 10 | 10 |
| In-run eval sets | MATH-500 (n=200, 2 reps), AIME24 (8 reps), AMC24 (4 reps) | same |
| `save_freq` | 50 (so only end-of-training checkpoint saved) | same |

---

## 3. DCPO in-run eval — head-to-head

In-run eval fires at steps **0, 10, 20, 30** (not 40 — `num_iterations=40` means indices
0..39; the `steps_per_eval=10` guard did not trigger again after the final optimizer step).

### MATH-500 (n=200, 2 repeats → 400 generations per eval)

| Step | A1 acc | A2 acc | A1 ECE (verbal) | A2 ECE (verbal) | A1 ECE (logits) | A2 ECE (logits) | A1 AUROC (verbal) | A2 AUROC (verbal) | A1 pass@k | A2 pass@k |
|---|---|---|---|---|---|---|---|---|---|---|
|  0 | 0.7975 | 0.7975 | 0.1460 | 0.1460 | 0.0827 | 0.0827 | 0.6587 | 0.6587 | 0.8550 | 0.8550 |
| 10 | 0.7900 | 0.7825 | 0.1474 | 0.1527 | 0.0913 | 0.1033 | 0.7005 | 0.6970 | 0.8500 | 0.8200 |
| 20 | 0.7975 | 0.7900 | 0.1424 | 0.1379 | 0.0972 | 0.0922 | 0.7094 | 0.7259 | 0.8700 | 0.8600 |
| 30 | 0.7900 | **0.8050** | 0.1503 | **0.1378** | 0.0891 | **0.0774** | 0.6990 | 0.7098 | 0.8450 | 0.8500 |

### AIME24 (n=30, 8 repeats → 240 generations per eval)

| Step | A1 acc | A2 acc | A1 ECE (verbal) | A2 ECE (verbal) | A1 AUROC (verbal) | A2 AUROC (verbal) | A1 parse | A2 parse | A1 pass@k | A2 pass@k |
|---|---|---|---|---|---|---|---|---|---|---|
|  0 | 0.2125 | 0.2125 | 0.3931 | 0.3931 | 0.8078 | 0.8078 | 0.6208 | 0.6208 | 0.4000 | 0.4000 |
| 10 | **0.2583** | 0.2417 | 0.4037 | 0.4295 | 0.7604 | 0.7669 | 0.6792 | 0.6875 | 0.4667 | 0.3667 |
| 20 | 0.2250 | 0.2375 | 0.3795 | 0.3810 | 0.7517 | 0.7564 | 0.6250 | 0.6333 | 0.3333 | 0.4000 |
| 30 | 0.2375 | 0.2167 | 0.3878 | 0.3910 | 0.7800 | **0.8032** | 0.6417 | 0.6250 | 0.4000 | 0.4000 |

### AMC24 (n=45, 4 repeats → 180 generations per eval)

| Step | A1 acc | A2 acc | A1 ECE (verbal) | A2 ECE (verbal) | A1 AUROC (verbal) | A2 AUROC (verbal) | A1 parse | A2 parse | A1 pass@k | A2 pass@k |
|---|---|---|---|---|---|---|---|---|---|---|
|  0 | 0.5000 | 0.5000 | 0.3547 | 0.3547 | 0.7734 | 0.7734 | 0.8722 | 0.8722 | 0.6222 | 0.6222 |
| 10 | 0.4111 | 0.4389 | 0.4318 | 0.3748 | 0.7460 | 0.7668 | 0.8611 | 0.8278 | 0.5778 | 0.6000 |
| 20 | 0.4333 | 0.4556 | 0.3994 | 0.4017 | 0.7328 | 0.7162 | 0.8500 | 0.8778 | 0.6000 | 0.6444 |
| 30 | 0.4556 | **0.4889** | 0.4090 | 0.3846 | 0.7524 | 0.7283 | 0.8833 | 0.8944 | 0.7111 | 0.6667 |

### Ensemble read

- **Accuracy**: essentially flat on MATH-500 (both within ±1 pp of baseline 0.7975).
  AIME24 and AMC24 drifted **down** from baseline (AMC24: 0.50 → 0.46–0.49, AIME24 stable ~0.22).
  No run meaningfully improved accuracy; the training signal has not moved the policy on held-out math.
- **Calibration (verbal ECE)**: MATH-500 stable at 0.14–0.15 for both. **Neither run reduced ECE.**
  AIME24/AMC24 verbal ECE actually worse than baseline on some steps.
- **Calibration (logits ECE)**: MATH-500 slight drift 0.0827 → 0.09 (A1), 0.0827 → 0.077 (A2).
  A2 is marginally better on logits ECE at step 30.
- **AUROC (verbal, discrimination)**: MATH-500 improved slightly for both (0.659 → 0.70).
  The model's confidence ordering is a bit better, but since ECE didn't move, this is
  scale-only, not resolution.

**Bottom line**: Neither variant produced a defensible calibration improvement on held-out
DCPO benchmarks in 40 ES steps.

---

## 4. Training-step telemetry (from the rollout batch itself)

Selected rows. `anchor_wrong_conf_mean` and `anchor_wrong_frac_q_ge_0p8` are
computed only over anchor prompts that were answered **incorrectly** in the batch; when
all anchors are solved correctly they degenerate to `0.0` (empty-set convention).

### A1 (ρ=0.8)

| Step | correct_rate | mean_q | base_solved_retention | anchor_wrong_conf_mean | anchor_wrong_frac_q≥0.8 | truncation | parse_success | loo_reward_std | confidence_std |
|---|---|---|---|---|---|---|---|---|---|
|  0 | 0.6299 | 0.9785 | 0.9824 | 0.9978 | 1.00 | 0.011 | 0.989 | 0.9393 | 0.0000 |
|  5 | 0.7188 | 0.9888 | 0.9123 | 0.9999 | 1.00 | 0.004 | 0.996 | 0.9194 | 0.0000 |
| 10 | 0.7520 | 0.9672 | 0.6815 | 0.9814 | 0.99 | 0.005 | 0.993 | 1.0713 | 0.0000 |
| 15 | 0.5205 | 0.7685 | 0.7680 | 0.9610 | 1.00 | 0.217 | 0.781 | 1.1367 | 0.0000 |
| 20 | 0.7520 | 0.9909 | 1.0000 | 0.0000¹ | 0.00¹ | 0.001 | 0.999 | 0.8476 | 0.0000 |
| 25 | 0.7051 | 0.8583 | 0.8507 | 0.9435 | 0.98 | 0.122 | 0.876 | 1.0344 | 0.0000 |
| 30 | 0.8037 | 0.9942 | 0.8045 | 0.9948 | 1.00 | 0.001 | 0.999 | 0.9783 | 0.0000 |
| 35 | 0.5000 | 0.9748 | 1.0000 | 0.0000¹ | 0.00¹ | 0.011 | 0.989 | 0.9913 | 0.0000 |
| 38 | 0.4922 | 0.7674 | 0.9724 | 0.9721 | 1.00 | 0.223 | 0.777 | 1.0539 | 0.0000 |
| 39 | 1.0000 | 0.9996 | 1.0000 | 0.0000¹ | 0.00¹ | 0.000 | 1.000 | **0.0002²** | 0.0000 |

### A2 (ρ=1.0)

| Step | correct_rate | mean_q | base_solved_retention | anchor_wrong_conf_mean | anchor_wrong_frac_q≥0.8 | truncation | parse_success | loo_reward_std | confidence_std |
|---|---|---|---|---|---|---|---|---|---|
|  0 | 0.6318 | 0.9838 | 0.9805 | 0.9900 | 1.00 | 0.007 | 0.993 | 0.9683 | 0.0000 |
|  5 | 0.7051 | 0.9858 | 0.8976 | 0.9986 | 1.00 | 0.006 | 0.994 | 0.9977 | 0.0000 |
| 10 | 0.7451 | 0.9654 | 0.6824 | 0.9795 | 0.99 | 0.006 | 0.992 | 1.1440 | 0.0000 |
| 15 | 0.5059 | 0.7570 | 0.7540 | 0.9672 | 1.00 | 0.230 | 0.770 | 1.1843 | 0.0000 |
| 20 | 0.7568 | 0.9889 | 1.0000 | 0.0000¹ | 0.00¹ | 0.001 | 0.999 | 0.8390 | 0.0000 |
| 25 | 0.7139 | 0.8554 | 0.8670 | 0.9375 | 0.98 | 0.126 | 0.872 | 1.0776 | 0.0000 |
| 30 | 0.7979 | 0.9946 | 0.7979 | 0.9936 | 1.00 | 0.000 | 1.000 | 1.0769 | 0.0000 |
| 35 | 0.5000 | 0.9731 | 1.0000 | 0.0000¹ | 0.00¹ | 0.011 | 0.989 | 0.9894 | 0.0000 |
| 38 | 0.4902 | 0.7758 | 0.9723 | 0.9857 | 1.00 | 0.214 | 0.786 | 1.0593 | 0.0000 |
| 39 | 1.0000 | 0.9996 | 1.0000 | 0.0000¹ | 0.00¹ | 0.000 | 1.000 | **0.0002²** | 0.0000 |

¹ `0.0000` here is an **empty-set artifact**: the batch contained no *wrong-answer anchor*,
so the mean-confidence-on-wrong-anchors is computed over zero samples. It is **not** evidence
that the model downweighted confidence on wrong anchors.

² `loo_reward_std: 0.0002` at step 39 = **effectively zero population variance**. The entire
population of 256 LoRA perturbations produced near-identical rewards on this batch.

### The two structural problems visible in telemetry

1. **The wrong-conf-anchor penalty is not firing signal.**
   On every step where anchors *were* answered incorrectly (0, 5, 10, 15, 25, 30, 38),
   `anchor_wrong_conf_mean` is stuck at **0.94–0.998**, and `anchor_wrong_frac_q≥0.8` is
   pinned at **1.00**. λ_wc = 0.5 with `q²` multiplier is not producing enough signal to
   pull confidence down on wrong anchors. Both runs are identical on this axis.

2. **Periodic truncation crashes every ~10 steps.**
   Steps 15, 25, 38 show truncation spiking from ~0.5% to **22–23%** with parse rate
   dropping to 77–78%. `mean_q` also drops (0.97 → 0.77) at these steps. This is likely
   the `steps_per_adapter=4` reuse cycle interacting with how prompts are drawn — the
   policy wanders into a long-response mode once every ~10 steps, most generations hit
   the 2048-token cap, and `q` becomes unparseable. The ES update then pulls the policy
   back on the next step.

---

## 5. ES fitness signal — near-total collapse

`training_state.json → fitnesses_so_far` contains one summary fitness per step (40 values).
Counting steps where `|fitness| > 1e-5`:

| | **A1 (ρ=0.8)** | **A2 (ρ=1.0)** |
|---|---|---|
| Steps with meaningful fitness (\|f\| > 1e-5) | **4 / 40** | **4 / 40** |
| Max abs fitness | 0.036 | 0.016 |
| Non-noise values (sorted) | 0.036, 0.0019, 1.7e-4, 8.8e-5 | 0.016, 0.0064, 1.3e-4, 8.6e-5 |
| All other steps (36/40) | floored at **1e-16 to 1e-14** (f64 eps) | same |

**This is ES mode collapse**: for 36 of 40 steps, every perturbation in the population of 256
produced rewards that are identical up to double-precision rounding, so the ES gradient
estimate is zero and the update is a no-op. Real learning happened on only 4 steps in each run.

Combined with step-39 telemetry (`loo_reward_std=0.0002`, `confidence_std=0`,
`pass_at_k_fitness=1.0000`, batch-`correct_rate=1.0000`), this is consistent with:

- The policy has converged to a narrow output mode that says `"<boxed_answer>, <conf>≈1.00</conf>"`
  regardless of correctness.
- On batches where the anchor prompts happen to be solvable by this narrow mode, every
  population perturbation scores the same → zero variance → zero gradient.
- The *only* steps with nonzero fitness are the ones where some perturbations hit the
  truncation/parse cliff (steps 15, 25, 38) and others don't — i.e. the gradient is being
  driven by **the format/truncation penalty**, not by the calibration objective.

---

## 6. Rejection-first selection verdict

Thresholds from the pre-registered plan vs. observed step-30 results (step-40 eval
was not produced by the in-run hook):

| Threshold | A1 step-30 | A2 step-30 | A1? | A2? |
|---|---|---|---|---|
| MATH-500 acc retention ≥ 0.93 of base (0.7975 → ≥ 0.7417) | 0.7900 | 0.8050 | ✓ | ✓ |
| AIME24/25 acc retention ≥ 0.75 of base (0.2125 → ≥ 0.159) | 0.2375 | 0.2167 | ✓ | ✓ |
| AMC24 acc retention ≥ 0.80 of base (0.5000 → ≥ 0.40) | 0.4556 | 0.4889 | ✓ | ✓ |
| MATH-500 format_valid ≥ 0.95 | 0.9475 | 0.9450 | ✗ (borderline) | ✗ (borderline) |
| AIME24 format_valid ≥ 0.95 | 0.6417 | 0.6250 | ✗ | ✗ |
| Anchor wrong-conf excess ≤ 0.20 (training-step, anchor_wrong_conf_mean − 0.2) | ~0.79 | ~0.79 | ✗✗ | ✗✗ |
| Truncation ≤ 0.10 (training-step median across last 20) | median ~0.01 but **max 0.22** | median ~0.00 but **max 0.21** | ✗ on cycle peaks | ✗ on cycle peaks |

**Verdict: Neither A1 nor A2 passes.** Both fail the two hardest gates:
1. **Wrong-conf-anchor calibration never bent down** — the central hypothesis of this
   triage (that λ_wc = 0.5 with the `q²` multiplier would teach the model to downweight
   confidence on wrong anchors) is empirically refuted at these λ settings.
2. **Periodic truncation blow-ups** (≥20% every ~10 steps) violate the truncation gate
   and drive the *only* meaningful ES fitness signal, which contaminates the objective.

A1 vs A2 are statistically indistinguishable on every held-out DCPO metric. The LOO-blend
choice (ρ = 0.8 vs 1.0) does not move the needle at these λ settings — the wrong-conf
penalty is too weak for either to matter.

---

## 7. Diagnosis — why the penalty didn't work

At baseline (step 0), with Qwen3-8B's well-known overconfidence: on anchor prompts that
the *trained* model gets wrong, the model outputs `q ≈ 0.99`. The reward contribution from
the wrong-conf-anchor term is then:

```
−λ_wc · A · (1 − C) · q²   =  −0.5 · 1 · 1 · 0.99² ≈ −0.49
```

That is a healthy penalty in isolation. But it competes with:
- The calibration term `−λ_cal·(q−T)² = −1.0·(0.99−1.0)² ≈ −0.0001` — negligible
  when the policy is correct, but when `ρ=1.0` on a wrong anchor, `T = C = 0`, so
  `(q−T)² = 0.99²` and λ_cal pushes a **symmetric** penalty of 0.98. For A1 (ρ=0.8),
  `T = 0.8·0 + 0.2·C̄_{-i,j} ≈ 0.2·0.5 = 0.1`, giving `(0.99−0.1)² ≈ 0.79`, still large.
- The retention penalty `−λ_ret·A·(1−C) = −0.2·1·1 = −0.20`.
- Correctness reward `+C = 0`.

So the total reward on a wrong, high-confidence anchor is dominated by `−λ_cal·(q−T)²`
(≈ −0.8 to −0.98), which is **already** penalizing wrong-with-high-confidence via Brier.
Adding λ_wc = 0.5 on top of that in an ES setting doesn't substantially change the
*ordering* of perturbations — each perturbation's score is dominated by Brier, not by the
wrong-conf-anchor term, so the ES gradient estimate is driven by Brier too.

This explains why:
- Runs behave identically (ρ doesn't matter when Brier dominates both rewards),
- Wrong-conf never bent (the supposed-to-dominate penalty is actually secondary to Brier), and
- ES fitness signal collapsed to noise on 36/40 steps (population members produce nearly
  identical Brier scores when they converge to `q ≈ 1.0` on every prompt).

---

## 8. Recommendations for the next run

Before launching any 120-step stability run, one of the following must change:

**(A) Boost the wrong-conf-anchor penalty significantly.**
Increase `λ_wc` to **2.0–4.0** and change the multiplier from `q²` to `q` (linear, so it
doesn't vanish on small q) or `q⁴` (sharper gating on high-q wrongs). Current λ_wc=0.5
with q² is dominated by Brier; the term needs to be O(1) at `q=1, C=0` comparable to Brier's
`(q−T)²` contribution.

**(B) Cap per-sample Brier influence on anchors.**
Set `λ_cal = 0.2` on anchor prompts (or zero it out entirely) so the wrong-conf-anchor and
retention penalties can carry the signal on anchors, while keeping `λ_cal = 1.0` on
non-anchor prompts.

**(C) Fix the periodic truncation cliff.**
The 22% truncation spike every ~10 steps is either a `steps_per_adapter=4` LoRA-reuse
artifact or `max_tokens=2048` being too aggressive for CoT expansions. Try
`max_tokens=3000` (matches DCPO) and/or `steps_per_adapter=2`.

**(D) Lower the fitness-variance floor.**
`global_std_floor=0.05` is saving us from divide-by-zero explosions but is also the
reason the fitness collapses to ~1e-15 on 36/40 steps — perturbations that all produce
near-identical raw rewards are still scaled by a std of 0.05, producing floor-precision
outputs. Consider dropping the floor to `0.01` or replacing with a rank-based normalizer.

**(E) Ablate: re-run A1 without the new penalties.**
The `rlcr_hybrid_loo` baseline (no retention/wrong-conf-anchor terms) on the same 40-step
budget would tell us whether the new penalties are even active vs. just ignored. If that
run also shows 36/40 zero-fitness steps, the pathology is in the ES pipeline (global_std_floor,
batch size, lr), not the reward shape.

**Do not launch the 120-step stability run yet.** The 40-step triage produced the signal
it was designed to produce: the current recipe doesn't move wrong-conf-anchor calibration,
and scaling to 120 steps would just be a more expensive no-op.

---

## 9. Artifacts

Local (for this analysis):
- `analysis_A1A2/A1/{training.log, training_state.json, in_run_eval/step_{0,10,20,30}{,_eval}.json}`
- `analysis_A1A2/A2/{training.log, training_state.json, in_run_eval/step_{0,10,20,30}{,_eval}.json}`
- `analysis_A1A2/summary.json` — consolidated parse of the above

GCS (full uploads):
- `gs://esvpg-experiments/es_exp/triage_A1_1776873821/{logs/, eval_jsons/, checkpoints/checkpoint_step_39/, wandb/}`
- `gs://esvpg-experiments/es_exp/triage_A2_1776873944/{logs/, eval_jsons/, checkpoints/checkpoint_step_39/, wandb/}`
- `gs://esvpg-experiments/es_exp/anchor_sets/anchor_set_deepscaler40k_seed42.json` (shared)

WandB:
- Run A1: `hyperscalees-vllm/triage-A1-rho08-s0.0005-lr0.0005-1776873821-…-1776873837`
- Run A2: `hyperscalees-vllm/triage-A2-rho10-s0.0005-lr0.0005-1776873944-…-1776873960`
