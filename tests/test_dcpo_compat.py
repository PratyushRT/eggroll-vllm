"""
CPU-only unit tests for the DCPO-compat modules added to eggroll-vllm:

    - dcpo_prompt   (directive, regex, parse_conf_for_{eval,training}, strip_conf)
    - dcpo_grader   (is_equiv, last_boxed_only_string, compute_score, grade)
    - dcpo_metrics  (ece, pce, brier, mce, auroc, logits_confidence_from_token_logprobs)
    - tasks.CalibratedMathTask reward variants (hybrid / instance / rlcr)

Run from the repo root:
    pytest -xvs tests/test_dcpo_compat.py

These tests do NOT require a GPU, vLLM, transformers, or network access.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# Make repo root importable regardless of how pytest is invoked.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dcpo_grader as G
import dcpo_metrics as M
import dcpo_prompt as P


# ---------------------------------------------------------------------------
# dcpo_prompt
# ---------------------------------------------------------------------------

class TestDCPOPrompt:
    def test_directive_contains_dcpo_markers(self):
        # Sanity: directive is the literal DCPO string (typo + full-width colon)
        assert "CONFIDENCE:" in P.DCPO_CONFIDENCE_DIRECTIVE
        assert "singal" in P.DCPO_CONFIDENCE_DIRECTIVE      # DCPO typo
        assert "：" in P.DCPO_CONFIDENCE_DIRECTIVE           # full-width colon
        assert "\\boxed{}" in P.DCPO_CONFIDENCE_DIRECTIVE

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Blah. CONFIDENCE: 0.83", 0.83),
            ("CONFIDENCE: 1", 1.0),
            ("CONFIDENCE: 0", 0.0),
            ("Some garbage first CONFIDENCE: 0.5 trailing", 0.5),
            ("CONFIDENCE:.5", 0.5),  # regex allows leading-dot-less forms
        ],
    )
    def test_parse_conf_for_training_ok(self, text, expected):
        q, status = P.parse_conf_for_training(text)
        assert status == "ok"
        assert math.isclose(q, expected)

    def test_parse_conf_for_training_missing_returns_zero(self):
        q, status = P.parse_conf_for_training("no conf here")
        assert status == "no_match"
        assert q == 0.0

    def test_parse_conf_for_training_clamps_out_of_range(self):
        # The regex only captures values like "1" or "0.x" — but sanity check clamp.
        q, status = P.parse_conf_for_training("CONFIDENCE: 1")
        assert q == 1.0 and status == "ok"

    def test_parse_conf_for_eval_missing_returns_negative_sentinel(self):
        q, status = P.parse_conf_for_eval("nothing here")
        assert status == "no_match"
        assert q == P.EVAL_DEFAULT_NO_MATCH
        assert q < 0  # so ECE/Brier masks exclude it

    def test_parse_conf_for_eval_match_returns_value(self):
        q, status = P.parse_conf_for_eval("CONFIDENCE: 0.7")
        assert status == "ok"
        assert q == 0.7

    def test_strip_conf_removes_conf_line(self):
        s = "My answer is \\boxed{42}. CONFIDENCE: 0.83"
        cleaned = P.strip_conf(s)
        assert "CONFIDENCE" not in cleaned
        assert "\\boxed{42}" in cleaned


# ---------------------------------------------------------------------------
# dcpo_grader
# ---------------------------------------------------------------------------

class TestDCPOGrader:
    def test_last_boxed_only_string(self):
        s = "Some text \\boxed{42} and more \\boxed{7}"
        assert G.last_boxed_only_string(s) == "\\boxed{7}"

    def test_last_boxed_none_when_missing(self):
        assert G.last_boxed_only_string("no box here") is None

    def test_remove_boxed_brace_form(self):
        assert G.remove_boxed("\\boxed{42}") == "42"

    def test_remove_boxed_space_form(self):
        assert G.remove_boxed("\\boxed 42") == "42"

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("42", "42", True),
            ("42", "43", False),
            # LaTeX normalisation equivalences from DCPO's strip_string:
            (r"\frac{1}{2}", r"\dfrac{1}{2}", True),    # dfrac -> frac
            (r"\frac{1}{2}", r"\tfrac{1}{2}", True),    # tfrac -> frac
            (r"\sqrt{2}", r"\sqrt2", True),             # sqrt2 -> sqrt{2}
            (r"\frac 1 2", r"\frac{1}{2}", True),       # fix_fracs
            (r"\left(3\right)", "(3)", True),           # \left/\right stripped
            (r"45^{\circ}", "45", True),                # degrees stripped
        ],
    )
    def test_is_equiv(self, a, b, expected):
        assert G.is_equiv(a, b) == expected, f"is_equiv({a!r},{b!r}) != {expected}"

    def test_compute_score_correct(self):
        gen = "We reason... so the answer is \\boxed{42}."
        assert G.compute_score(gen, "42") == 1.0

    def test_compute_score_wrong(self):
        gen = "\\boxed{7}"
        assert G.compute_score(gen, "42") == 0.0

    def test_compute_score_no_boxed(self):
        gen = "The answer is 42."
        assert G.compute_score(gen, "42") == 0.0

    def test_grade_list_ground_truth_any_match(self):
        gen = "\\boxed{7}"
        assert G.grade(gen, ["42", "7", "3"]) is True

    def test_grade_int_ground_truth(self):
        gen = "\\boxed{42}"
        assert G.grade(gen, 42) is True
        assert G.grade(gen, 43) is False

    def test_grade_none_ground_truth_is_false(self):
        assert G.grade("\\boxed{1}", None) is False


# ---------------------------------------------------------------------------
# dcpo_metrics
# ---------------------------------------------------------------------------

class TestDCPOMetrics:
    def test_ece_all_correct_at_perfect_confidence(self):
        # All samples: conf=1.0, acc=1.0 → ECE = 0
        conf = np.ones(100)
        acc = np.ones(100)
        assert M.compute_ece(conf, acc, n_bins=15) == 0.0

    def test_ece_all_wrong_at_full_confidence(self):
        # conf=1.0, acc=0.0 → ECE = 1.0 (single bin, full overconfidence)
        conf = np.ones(100)
        acc = np.zeros(100)
        assert math.isclose(M.compute_ece(conf, acc, n_bins=15), 1.0, abs_tol=1e-9)

    def test_pce_zero_when_underconfident(self):
        # conf=0.2, acc=0.9 (underconfident) → PCE = 0
        conf = np.full(100, 0.2)
        acc = np.full(100, 0.9)
        assert M.compute_overconf_ece(conf, acc, n_bins=15) == 0.0

    def test_pce_positive_when_overconfident(self):
        conf = np.full(100, 0.9)
        acc = np.full(100, 0.2)
        pce = M.compute_overconf_ece(conf, acc, n_bins=15)
        assert math.isclose(pce, 0.7, abs_tol=1e-9)

    def test_brier_perfect(self):
        # conf perfectly matches acc ∈ {0,1} → Brier = 0
        conf = np.array([1.0, 0.0, 1.0, 0.0])
        acc = np.array([1.0, 0.0, 1.0, 0.0])
        assert M.compute_brier(conf, acc) == 0.0

    def test_brier_known_value(self):
        # conf=0.8, acc=1 → (0.8-1)^2 = 0.04
        conf = np.full(10, 0.8)
        acc = np.ones(10)
        assert math.isclose(M.compute_brier(conf, acc), 0.04, abs_tol=1e-9)

    def test_brier_excludes_negative_sentinels(self):
        # DCPO masks out malformed confidences (negative sentinels).
        conf = np.array([-0.1, -0.2, 0.8, 0.8])
        acc = np.array([1.0, 1.0, 1.0, 1.0])
        # Only the two 0.8s contribute: mean((0.8-1)^2) = 0.04
        assert math.isclose(M.compute_brier(conf, acc), 0.04, abs_tol=1e-9)

    def test_mce_bounded_between_0_and_1(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0, 1, 500)
        acc = (rng.uniform(0, 1, 500) > 0.5).astype(float)
        mce = M.compute_mce(conf, acc, n_bins=15)
        assert 0.0 <= mce <= 1.0

    def test_auroc_perfect_ranking(self):
        # Higher confidence for correct samples → AUROC = 1
        conf = np.array([0.9, 0.8, 0.1, 0.2])
        acc = np.array([1.0, 1.0, 0.0, 0.0])
        assert math.isclose(M.compute_auroc(conf, acc), 1.0, abs_tol=1e-9)

    def test_auroc_random(self):
        # All-same confidence → AUROC = 0.5 (or defined tied-value behavior)
        conf = np.full(100, 0.5)
        acc = (np.arange(100) % 2).astype(float)
        auroc = M.compute_auroc(conf, acc)
        assert math.isclose(auroc, 0.5, abs_tol=1e-9)

    def test_auroc_nan_when_single_class(self):
        conf = np.array([0.1, 0.9])
        acc = np.array([1.0, 1.0])   # only one class
        assert math.isnan(M.compute_auroc(conf, acc))

    def test_logits_confidence_from_token_logprobs(self):
        # log(0.9), log(0.9) → mean log = log(0.9), exp → 0.9
        import math as _m
        lps = [_m.log(0.9), _m.log(0.9), _m.log(0.9)]
        got = M.logits_confidence_from_token_logprobs(lps)
        assert math.isclose(got, 0.9, abs_tol=1e-9)

    def test_logits_confidence_empty(self):
        assert M.logits_confidence_from_token_logprobs([]) == 0.0

    def test_length_normed_seq_prob(self):
        # cum_lp=-3, n_tokens=3 → exp(-1) ≈ 0.3679
        got = M.length_normed_seq_prob(-3.0, 3)
        assert math.isclose(got, math.exp(-1), abs_tol=1e-9)

    def test_length_normed_zero_tokens(self):
        assert M.length_normed_seq_prob(0.0, 0) == 0.0


# ---------------------------------------------------------------------------
# CalibratedMathTask reward variants
#
# We test the reward math only; we do NOT instantiate MathTask's dataset loader
# (which would require network + datasets lib). We subclass-ish the reward math
# via a synthetic object and call get_fitness with hand-crafted generations.
# ---------------------------------------------------------------------------

def _make_task(variant: str, lambda_cal: float = 0.5, instance_weight: float = 0.3):
    """Build a CalibratedMathTask bypassing MathTask.__init__ (no network)."""
    # Stub `egg_img` — `tasks.py` does `from egg_img import EGG_IMG, CHICK_IMG`
    # at module top level for a banner; it's not installed in test envs.
    import types as _types
    if "egg_img" not in sys.modules:
        _stub = _types.ModuleType("egg_img")
        _stub.EGG_IMG = ""
        _stub.CHICK_IMG = ""
        sys.modules["egg_img"] = _stub
    # Stub `gem.utils.math_grader` — tasks.py imports `extract_answer` and
    # `grade` at module level. CalibratedMathTask uses DCPO's grader instead,
    # so these stubs are never exercised in this test.
    if "gem" not in sys.modules:
        gem_pkg = _types.ModuleType("gem")
        gem_utils = _types.ModuleType("gem.utils")
        gem_math = _types.ModuleType("gem.utils.math_grader")
        gem_math.extract_answer = lambda *a, **k: None
        gem_math.grade = lambda *a, **k: False
        sys.modules["gem"] = gem_pkg
        sys.modules["gem.utils"] = gem_utils
        sys.modules["gem.utils.math_grader"] = gem_math
    from tasks import CalibratedMathTask
    task = CalibratedMathTask.__new__(CalibratedMathTask)
    task.reward_variant = variant
    task.lambda_cal = lambda_cal
    task.instance_weight = instance_weight
    task.enable_thinking = False
    task.tokenizer = None
    task.apply_chat_template = False
    task.ans_format = "boxed"
    return task


def _gen(boxed_ans: str, conf: str = None) -> str:
    s = f"Reasoning. The answer is \\boxed{{{boxed_ans}}}."
    if conf is not None:
        s += f"\nCONFIDENCE: {conf}"
    return s


class TestCalibratedMathTaskRewards:
    # Ground truth for all tests.
    GT = "42"

    def test_instance_correct_perfect_confidence(self):
        task = _make_task("instance")
        gens = [_gen("42", "1.0")]
        trunc = [False]
        fitness, model_ans, fits, meta = task.get_fitness(gens, trunc, self.GT)
        # C=1, q=1 → R = 1 - 0.3 * (1-1)^2 = 1
        assert math.isclose(fits[0], 1.0, abs_tol=1e-9)
        assert meta["correct_rate"] == 1.0

    def test_instance_wrong_confident(self):
        task = _make_task("instance")
        gens = [_gen("7", "1.0")]
        trunc = [False]
        _, _, fits, _ = task.get_fitness(gens, trunc, self.GT)
        # C=0, q=1 → R = 0 - 0.3 * (0-1)^2 = -0.3
        assert math.isclose(fits[0], -0.3, abs_tol=1e-9)

    def test_instance_wrong_unconfident(self):
        task = _make_task("instance")
        gens = [_gen("7", "0.0")]
        trunc = [False]
        _, _, fits, _ = task.get_fitness(gens, trunc, self.GT)
        # C=0, q=0 → R = 0 - 0.3 * (0-0)^2 = 0
        assert math.isclose(fits[0], 0.0, abs_tol=1e-9)

    def test_rlcr_correct_half_confidence(self):
        task = _make_task("rlcr", lambda_cal=0.5)
        gens = [_gen("42", "0.5")]
        trunc = [False]
        _, _, fits, _ = task.get_fitness(gens, trunc, self.GT)
        # C=1, q=0.5 → R = 1 - 0.5 * (0.5-1)^2 = 1 - 0.5 * 0.25 = 0.875
        assert math.isclose(fits[0], 0.875, abs_tol=1e-9)

    def test_rlcr_missing_confidence_uses_zero(self):
        task = _make_task("rlcr", lambda_cal=0.5)
        gens = [_gen("42")]    # no CONFIDENCE line
        trunc = [False]
        _, _, fits, meta = task.get_fitness(gens, trunc, self.GT)
        # C=1, q=0 → R = 1 - 0.5 * (0-1)^2 = 0.5
        assert math.isclose(fits[0], 0.5, abs_tol=1e-9)
        assert meta["parse_success_rate"] == 0.0

    def test_hybrid_group_mean_applies(self):
        task = _make_task("hybrid", lambda_cal=0.5)
        # 4 rollouts: 2 correct, 2 wrong → group mean C = 0.5
        gens = [
            _gen("42", "1.0"),   # correct, confident
            _gen("42", "0.5"),   # correct, medium
            _gen("7",  "0.9"),   # wrong, confident
            _gen("7",  "0.1"),   # wrong, unsure
        ]
        trunc = [False] * 4
        _, _, fits, meta = task.get_fitness(gens, trunc, self.GT)

        mean_C = 0.5  # (1+1+0+0)/4
        # Per DCPO Hybrid: target_i = 0.5*C_i + 0.5*mean_C
        #   i=0: target = 0.5*1+0.25=0.75;  R = 1 - 0.5*(0.75-1.0)^2 = 1 - 0.03125 = 0.96875
        #   i=1: target = 0.75;             R = 1 - 0.5*(0.75-0.5)^2 = 1 - 0.03125 = 0.96875
        #   i=2: target = 0.5*0+0.25=0.25;  R = 0 - 0.5*(0.25-0.9)^2 = -0.5*0.4225 = -0.21125
        #   i=3: target = 0.25;             R = 0 - 0.5*(0.25-0.1)^2 = -0.5*0.0225 = -0.01125
        expected = [0.96875, 0.96875, -0.21125, -0.01125]
        for got, want in zip(fits.tolist(), expected):
            assert math.isclose(got, want, abs_tol=1e-9), f"got {got}, want {want}"
        assert math.isclose(meta["correct_rate"], mean_C, abs_tol=1e-9)

    def test_truncated_samples_zero_reward(self):
        task = _make_task("hybrid")
        gens = [_gen("42", "1.0"), "partial cut off..."]
        trunc = [False, True]
        _, _, fits, meta = task.get_fitness(gens, trunc, self.GT)
        assert fits[1] == 0.0
        assert meta["truncation_rate"] == 0.5

    def test_pass_at_k_max_aggregation(self):
        task = _make_task("instance")
        gens = [_gen("7", "1.0"), _gen("42", "0.5")]
        trunc = [False, False]
        fitness, _, fits, _ = task.get_fitness(gens, trunc, self.GT, pass_at_k=True)
        # With pass_at_k, we want max(fits). Correct one has higher reward.
        assert fitness == float(np.max(fits))

    def test_dcpo_is_correct_strips_conf_before_grading(self):
        task = _make_task("instance")
        # Confidence number after the boxed answer should NOT break grading.
        s = "The answer is \\boxed{42}. CONFIDENCE: 0.9"
        is_correct, model_ans = task._dcpo_is_correct(s, "42")
        assert is_correct is True


# ---------------------------------------------------------------------------
# Property: rewards should always land in a bounded range
# ---------------------------------------------------------------------------

class TestRewardBounds:
    def test_instance_bounded(self):
        task = _make_task("instance", instance_weight=0.3)
        # Worst case: C=1, q=0 → R = 1 - 0.3 = 0.7. Or C=0, q=1 → R = -0.3.
        for C_gt, q_str, expected_sign in [("42", "0.0", "pos"), ("7", "1.0", "neg")]:
            gens = [_gen(C_gt, q_str)]
            _, _, fits, _ = task.get_fitness(gens, [False], "42")
            if expected_sign == "pos":
                assert fits[0] >= 0.0
            else:
                assert fits[0] <= 0.0
            assert -0.3 <= fits[0] <= 1.0

    def test_rlcr_bounded(self):
        task = _make_task("rlcr", lambda_cal=0.5)
        for C_gt, q_str in [("42", "0.0"), ("42", "1.0"), ("7", "0.0"), ("7", "1.0")]:
            gens = [_gen(C_gt, q_str)]
            _, _, fits, _ = task.get_fitness(gens, [False], "42")
            # RLCR: R = C - 0.5*(q-C)^2. C,q ∈ [0,1] → R ∈ [-0.5, 1].
            assert -0.5 <= fits[0] <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
