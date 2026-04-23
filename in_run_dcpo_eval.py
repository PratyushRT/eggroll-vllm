"""
In-run DCPO-style calibration eval for the ES training loop.

Runs AIME24 (always), AMC24 (if data/amc24.parquet exists), and MATH-500
(optional) against the LIVE vLLM engine used for training — no new process,
no new model load. Designed to be called every `steps_per_eval` steps so we
can see calibration trajectory (Acc, ECE, PCE, Brier, MCE, AUROC, parse
success, mean conf, conf entropy, pass@k) during training.

Integration (from es_lora_multinode.py):

    from in_run_dcpo_eval import run_in_run_eval
    metrics = run_in_run_eval(
        llm=engines[0],
        tokenizer=tokenizer,
        step=es_step,
        aime24_repeats=4,
        amc24_repeats=2,
        math500_n=100,
        math500_repeats=2,
        prompt_template="dcpo_verbose",   # or "conf_tags" — MUST match training
        enable_thinking=False,
        out_dir=os.path.join(run_dir, "in_run_evals"),
    )
    wandb.log(metrics)

Sampling params match `eval_dcpo_benchmarks.py` EXACTLY (temperature=0.7,
top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=3000, logprobs=1).

Dual confidence:
  * verbal: regex match on CONFIDENCE: / <conf>...</conf>, fallback to 0.0
  * logits: exp(mean token logprob) over the response (length-normalized)

Per-benchmark metrics (both verbal & logits):
    acc, ece, pce, brier, mce, auroc, parse_success, mean_conf,
    conf_entropy, pass_at_k, n_examples, n_generations, gen_seconds

Returns a flat dict keyed for wandb (prefix `eval/`), including `eval/step`.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dcpo_grader import compute_score as dcpo_compute_score
from dcpo_metrics import (
    compute_auroc,
    compute_brier,
    compute_ece,
    compute_mce,
    compute_overconf_ece,
    logits_confidence_from_token_logprobs,
)
from dcpo_prompt import (
    CONF_RE,
    DCPO_CONFIDENCE_DIRECTIVE,
    strip_conf,
)

# `tasks.py` exposes the <conf>...</conf> variant (the parallel template).
# Import defensively: the module should still be import-able if tasks.py can't
# be loaded in some environments.
try:
    from tasks import CONF_TAGS_DIRECTIVE, CONF_TAGS_RE
    _HAVE_CONF_TAGS = True
except Exception:
    CONF_TAGS_DIRECTIVE = None
    CONF_TAGS_RE = None
    _HAVE_CONF_TAGS = False


# ---------------------------------------------------------------------------
# DCPO-exact sampling params (match eval_dcpo_benchmarks.py / calibration_main.py).
# ---------------------------------------------------------------------------
_DCPO_TEMPERATURE = 0.7
_DCPO_TOP_P = 0.8
_DCPO_TOP_K = 20
_DCPO_PRESENCE_PENALTY = 1.5
_DCPO_MAX_TOKENS = 3000
_ECE_BINS = 20               # spec: 20 uniform bins
_CONF_HIST_BINS = 10         # spec: 10-bin Shannon entropy
_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Prompt-template selection (MUST match training).
# ---------------------------------------------------------------------------
def _resolve_template(prompt_template: str):
    """Return (directive_str, compiled_regex) for the active template."""
    if prompt_template == "conf_tags":
        if not _HAVE_CONF_TAGS:
            raise RuntimeError(
                "prompt_template='conf_tags' requested but tasks.CONF_TAGS_* "
                "could not be imported."
            )
        return CONF_TAGS_DIRECTIVE, CONF_TAGS_RE
    if prompt_template == "dcpo_verbose":
        return DCPO_CONFIDENCE_DIRECTIVE, CONF_RE
    raise ValueError(
        f"prompt_template must be 'dcpo_verbose' or 'conf_tags', got {prompt_template!r}"
    )


def _parse_conf(text: str, regex) -> Tuple[float, bool]:
    """Training-style parse: regex match -> clamped [0,1] float, else 0.0.

    Returns (conf, parsed_ok). `parsed_ok` distinguishes a real match from the
    0.0 fallback (needed for `parse_success`, `mean_conf`, `conf_entropy`).
    """
    m = regex.search(text)
    if m:
        try:
            q = float(m.group(1))
        except ValueError:
            return 0.0, False
        return max(0.0, min(1.0, q)), True
    return 0.0, False


def _strip_conf_generic(text: str, prompt_template: str) -> str:
    if prompt_template == "conf_tags":
        import re as _re
        return _re.sub(r"<conf>.*?</conf>", "", text, flags=_re.DOTALL).strip()
    return strip_conf(text)


# ---------------------------------------------------------------------------
# Dataset loaders. Env vars override the default data/ paths for air-gapped use.
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path, n: Optional[int] = None,
                problem_field: str = "problem",
                answer_field: str = "answer") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            row = json.loads(line)
            prob = row.get(problem_field) or row.get("question") or ""
            ans = row.get(answer_field) or row.get("solution") or ""
            out.append({"problem": str(prob), "answer": str(ans)})
    return out


def _load_parquet(path: Path, n: Optional[int] = None) -> List[Dict[str, str]]:
    """VERL-format parquet: `prompt` list-of-chat, `reward_model.ground_truth`.
    Also accepts plain `problem`/`answer` columns.
    """
    import pandas as pd
    df = pd.read_parquet(path)
    out: List[Dict[str, str]] = []
    upto = len(df) if n is None else min(n, len(df))
    for i in range(upto):
        row = df.iloc[i]
        prob: Any = None
        ans: Any = None
        if "problem" in df.columns:
            prob = row["problem"]
        elif "Problem" in df.columns:
            prob = row["Problem"]
        if "answer" in df.columns:
            ans = row["answer"]
        elif "Answer" in df.columns:
            ans = row["Answer"]
        # VERL-style fallback.
        if (prob is None or (isinstance(prob, float) and math.isnan(prob))) and "prompt" in df.columns:
            prompt_col = row["prompt"]
            try:
                prob = prompt_col[0]["content"] if prompt_col is not None else ""
            except Exception:
                prob = ""
        if (ans is None or (isinstance(ans, float) and math.isnan(ans))) and "reward_model" in df.columns:
            rm = row["reward_model"]
            try:
                ans = rm.get("ground_truth") if isinstance(rm, dict) else ""
            except Exception:
                ans = ""
        out.append({"problem": str(prob or ""), "answer": str(ans or "")})
    return out


def _load_math500(n: int) -> List[Dict[str, str]]:
    local = os.environ.get("IN_RUN_MATH500_PATH")
    path = Path(local) if local else (_DATA_DIR / "MATH-500" / "test.jsonl")
    if path.exists():
        return _load_jsonl(path, n=n)
    # Fallback: HF hub.
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append({
            "problem": str(row.get("problem", row.get("question", ""))),
            "answer": str(row.get("answer", row.get("solution", ""))),
        })
    return out


def _load_aime24() -> List[Dict[str, str]]:
    local = os.environ.get("IN_RUN_AIME24_PATH")
    path = Path(local) if local else (_DATA_DIR / "aime24.parquet")
    if path.exists():
        return _load_parquet(path)
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    out = []
    for row in ds:
        prob = row.get("Problem") or row.get("problem") or row.get("question") or ""
        ans = row.get("Answer") or row.get("answer") or ""
        out.append({"problem": str(prob), "answer": str(ans)})
    return out


def _load_amc24() -> Optional[List[Dict[str, str]]]:
    """Return None if the AMC24 parquet isn't available (another agent is
    sourcing it concurrently — skip without crashing)."""
    local = os.environ.get("IN_RUN_AMC24_PATH")
    path = Path(local) if local else (_DATA_DIR / "amc24.parquet")
    if not path.exists():
        return None
    try:
        return _load_parquet(path)
    except Exception as e:
        warnings.warn(f"in_run_dcpo_eval: failed to load AMC24 from {path}: {e}")
        return None


def _load_aime25() -> Optional[List[Dict[str, str]]]:
    """AIME25: prefer local JSONL/parquet under data/, fall back to HF hub."""
    local = os.environ.get("IN_RUN_AIME25_PATH")
    candidates: List[Path] = []
    if local:
        candidates.append(Path(local))
    candidates.extend([
        _DATA_DIR / "aime25.jsonl",
        _DATA_DIR / "aime25.parquet",
    ])
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    return _load_parquet(p)
                return _load_jsonl(p)
            except Exception as e:
                warnings.warn(f"in_run_dcpo_eval: failed to load AIME25 from {p}: {e}")
    # Fallback: HF hub. Try a couple of common mirrors.
    for repo_id, split in [
        ("yentinglin/aime_2025", "train"),
        ("opencompass/AIME2025", "test"),
        ("math-ai/aime25", "test"),
    ]:
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, split=split)
            out: List[Dict[str, str]] = []
            for row in ds:
                prob = (row.get("problem") or row.get("Problem") or
                        row.get("question") or row.get("Question") or "")
                ans = (row.get("answer") or row.get("Answer") or
                       row.get("solution") or "")
                out.append({"problem": str(prob), "answer": str(ans)})
            if out:
                return out
        except Exception:
            continue
    warnings.warn("in_run_dcpo_eval: AIME25 unavailable locally and HF fallbacks failed; skipping.")
    return None


def _load_amc23() -> Optional[List[Dict[str, str]]]:
    """AMC23: prefer local parquet/JSONL under data/, fall back to HF hub."""
    local = os.environ.get("IN_RUN_AMC23_PATH")
    candidates: List[Path] = []
    if local:
        candidates.append(Path(local))
    candidates.extend([
        _DATA_DIR / "amc23.parquet",
        _DATA_DIR / "amc23.jsonl",
    ])
    for p in candidates:
        if p.exists():
            try:
                if p.suffix == ".parquet":
                    return _load_parquet(p)
                return _load_jsonl(p)
            except Exception as e:
                warnings.warn(f"in_run_dcpo_eval: failed to load AMC23 from {p}: {e}")
    for repo_id, split in [
        ("math-ai/amc23", "test"),
        ("AI-MO/aimo-validation-amc", "train"),
    ]:
        try:
            from datasets import load_dataset
            ds = load_dataset(repo_id, split=split)
            out: List[Dict[str, str]] = []
            for row in ds:
                prob = (row.get("problem") or row.get("Problem") or
                        row.get("question") or "")
                ans = (row.get("answer") or row.get("Answer") or "")
                out.append({"problem": str(prob), "answer": str(ans)})
            if out:
                return out
        except Exception:
            continue
    warnings.warn("in_run_dcpo_eval: AMC23 unavailable locally and HF fallbacks failed; skipping.")
    return None


# ---------------------------------------------------------------------------
# Prompt construction.
# ---------------------------------------------------------------------------
def _format_chat(tokenizer, user_content: str, enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# vLLM generation. Handles Ray-actor engines and raw vllm.LLM alike.
# ---------------------------------------------------------------------------
def _generate(llm, prompts: List[str], sampling_params) -> list:
    gen = getattr(llm, "generate", None)
    if gen is None:
        raise AttributeError("`llm` has no `.generate` method")
    remote = getattr(gen, "remote", None)
    if remote is not None:
        import ray
        return ray.get(remote(prompts, sampling_params))
    return gen(prompts, sampling_params)


def _make_sampling_params(n: int, max_tokens: int, eos_token: Optional[str], seed: int):
    from vllm import SamplingParams
    stop = [eos_token] if eos_token else None
    return SamplingParams(
        n=int(n),
        temperature=_DCPO_TEMPERATURE,
        top_p=_DCPO_TOP_P,
        top_k=_DCPO_TOP_K,
        presence_penalty=_DCPO_PRESENCE_PENALTY,
        max_tokens=int(max_tokens),
        logprobs=1,
        seed=seed,
        stop=stop,
    )


# ---------------------------------------------------------------------------
# Logits confidence extraction.
# ---------------------------------------------------------------------------
def _logits_conf(comp) -> float:
    """DCPO logits confidence: exp(mean token logprob) over the response."""
    cum_lp = getattr(comp, "cumulative_logprob", None)
    n_tok = len(comp.token_ids) if getattr(comp, "token_ids", None) else 0
    if cum_lp is not None and n_tok > 0:
        return math.exp(cum_lp / n_tok)
    # Fallback: walk `logprobs` list-of-dicts.
    step_lp = getattr(comp, "logprobs", None) or []
    token_ids = comp.token_ids or []
    tlps: List[float] = []
    for step_idx, step_map in enumerate(step_lp):
        if step_map is None or step_idx >= len(token_ids):
            continue
        chosen = token_ids[step_idx]
        lp_obj = step_map.get(chosen) if isinstance(step_map, dict) else None
        if lp_obj is None:
            try:
                lp_obj = max(step_map.values(),
                             key=lambda x: getattr(x, "logprob", float("-inf")))
            except Exception:
                continue
        lp = getattr(lp_obj, "logprob", None)
        if lp is not None:
            tlps.append(float(lp))
    return logits_confidence_from_token_logprobs(tlps) if tlps else 0.0


# ---------------------------------------------------------------------------
# Entropy (over 10 uniform bins of the parsed-confidence distribution).
# ---------------------------------------------------------------------------
def _conf_entropy(parsed_confs: np.ndarray, n_bins: int = _CONF_HIST_BINS) -> float:
    if parsed_confs.size == 0:
        return 0.0
    hist, _ = np.histogram(parsed_confs, bins=n_bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


# ---------------------------------------------------------------------------
# Pass@k.
#
# Unbiased estimator (Chen et al. 2021, Codex):
#     pass@k = 1 - C(n-c, k) / C(n, k)
# averaged over prompts, where n = samples per prompt, c = correct count.
# When k > n the estimator is undefined — return NaN. When k == n this
# degenerates to the "any-correct" version.
# ---------------------------------------------------------------------------
def _pass_at_k_unbiased(correct_per_prompt: List[List[int]], k: int) -> float:
    if not correct_per_prompt:
        return float("nan")
    vals: List[float] = []
    for xs in correct_per_prompt:
        n = len(xs)
        if n == 0 or k > n:
            continue
        c = int(sum(xs))
        if n - c < k:
            vals.append(1.0)
        else:
            # 1 - C(n-c, k) / C(n, k) — compute stable via product form.
            #   C(n-c, k) / C(n, k) = prod_{i=0..k-1} (n-c-i) / (n-i)
            prob_all_wrong = 1.0
            for i in range(k):
                prob_all_wrong *= float(n - c - i) / float(n - i)
            vals.append(1.0 - prob_all_wrong)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# Backward-compat wrapper (legacy callers — any-correct pass@k).
def _pass_at_k(correct_per_prompt: List[List[int]]) -> float:
    if not correct_per_prompt:
        return float("nan")
    hits = [1.0 if any(xs) else 0.0 for xs in correct_per_prompt]
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# Per-benchmark scoring + metric aggregation.
# ---------------------------------------------------------------------------
def _score_outputs(
    examples: List[Dict[str, str]],
    outputs,
    prompt_template: str,
    regex,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (metrics_dict, raw_samples_list)."""
    conf_verbal: List[float] = []
    conf_logits: List[float] = []
    correct: List[int] = []
    parse_ok: List[int] = []
    per_prompt_correct: List[List[int]] = []
    raw_samples: List[Dict[str, Any]] = []

    for ex_idx, (ex, req_out) in enumerate(zip(examples, outputs)):
        gt = ex["answer"]
        prompt_corrects: List[int] = []
        for comp in req_out.outputs:
            text = comp.text
            q, ok = _parse_conf(text, regex)
            conf_verbal.append(q)
            parse_ok.append(1 if ok else 0)
            cl = _logits_conf(comp)
            conf_logits.append(cl)

            cleaned = _strip_conf_generic(text, prompt_template)
            try:
                is_correct = dcpo_compute_score(cleaned, str(gt)) > 0.5
            except Exception:
                is_correct = False
            c_int = 1 if is_correct else 0
            correct.append(c_int)
            prompt_corrects.append(c_int)

            raw_samples.append({
                "prompt_idx": ex_idx,
                "answer_gt": str(gt),
                "text": text,
                "verbal_conf": q,
                "verbal_parsed": bool(ok),
                "logits_conf": cl,
                "correct": c_int,
            })
        per_prompt_correct.append(prompt_corrects)

    cv = np.asarray(conf_verbal, dtype=np.float64)
    cl = np.asarray(conf_logits, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    ok = np.asarray(parse_ok, dtype=np.float64)

    n_gen = int(acc.size)
    parsed_mask = ok.astype(bool)
    parsed_confs = cv[parsed_mask]

    m: Dict[str, Any] = {
        "n_examples": len(examples),
        "n_generations": n_gen,
        "acc": float(acc.mean()) if n_gen else float("nan"),
        "parse_success": float(ok.mean()) if n_gen else float("nan"),
        "mean_conf": float(parsed_confs.mean()) if parsed_confs.size else 0.0,
        "conf_entropy": _conf_entropy(parsed_confs, _CONF_HIST_BINS),
        # Verbal confidence metrics (20-bin ECE per spec).
        "ece_verbal": compute_ece(cv, acc, n_bins=_ECE_BINS),
        "pce_verbal": compute_overconf_ece(cv, acc, n_bins=_ECE_BINS),
        "brier_verbal": compute_brier(cv, acc),
        "mce_verbal": compute_mce(cv, acc, n_bins=_ECE_BINS),
        "auroc_verbal": compute_auroc(cv, acc),
        # Logits confidence metrics.
        "ece_logits": compute_ece(cl, acc, n_bins=_ECE_BINS),
        "pce_logits": compute_overconf_ece(cl, acc, n_bins=_ECE_BINS),
        "brier_logits": compute_brier(cl, acc),
        "mce_logits": compute_mce(cl, acc, n_bins=_ECE_BINS),
        "auroc_logits": compute_auroc(cl, acc),
        "pass_at_k": _pass_at_k(per_prompt_correct),
        # Unbiased pass@k (Chen et al. 2021). pass@1 is the per-sample accuracy;
        # pass@4 requires >=4 samples per prompt, else NaN.
        "pass_at_1": _pass_at_k_unbiased(per_prompt_correct, 1),
        "pass_at_4": _pass_at_k_unbiased(per_prompt_correct, 4),
    }
    return m, raw_samples


# ---------------------------------------------------------------------------
# Flat wandb dict helpers.
# ---------------------------------------------------------------------------
_BENCH_METRIC_KEYS = [
    "acc", "ece_verbal", "pce_verbal", "brier_verbal", "mce_verbal", "auroc_verbal",
    "ece_logits", "pce_logits", "brier_logits", "mce_logits", "auroc_logits",
    "parse_success", "mean_conf", "conf_entropy",
    "pass_at_k", "pass_at_1", "pass_at_4",
    "n_examples", "n_generations", "gen_seconds",
]


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _flatten(prefix: str, m: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Flatten a benchmark metric dict into `eval/<prefix>_<key>`.
    Emits NaN for every key if `m` is None (benchmark skipped)."""
    out: Dict[str, float] = {}
    for k in _BENCH_METRIC_KEYS:
        # spec names pass_at_k as "pass_at_{k}" depending on repeats — use _4, _2 etc.
        out[f"eval/{prefix}_{k}"] = _f(m.get(k)) if m is not None else float("nan")
    return out


# ---------------------------------------------------------------------------
# Public entrypoint.
# ---------------------------------------------------------------------------
def run_in_run_eval(
    llm,
    tokenizer,
    step: int,
    aime24_repeats: int = 4,
    amc24_repeats: int = 2,
    aime25_repeats: int = 0,
    amc23_repeats: int = 0,
    math500_n: int = 100,
    math500_repeats: int = 2,
    enable_thinking: bool = False,
    prompt_template: str = "dcpo_verbose",
    max_tokens: int = _DCPO_MAX_TOKENS,
    out_dir: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a DCPO-exact calibration eval against the LIVE vLLM engine.

    Args:
        llm: Live vLLM engine (raw `vllm.LLM` or Ray actor exposing `.generate`).
        tokenizer: HF tokenizer (must match the engine's model) for chat templating.
        step: Training step (embedded as `eval/step`).
        aime24_repeats: Samples per AIME24 problem (30 × this = rollouts).
        amc24_repeats: Samples per AMC24 problem. Skipped (returns NaN metrics)
            if data/amc24.parquet isn't available.
        math500_n: First-N subsample of MATH-500.
        math500_repeats: Samples per MATH-500 problem. Set to 0 to skip.
        enable_thinking: Qwen3 chat-template flag. DCPO uses False.
        prompt_template: "dcpo_verbose" or "conf_tags" — MUST match training.
        max_tokens: Generation cap (default 3000, matches DCPO).
        out_dir: If given, dump full eval + raw per-sample data to
            `{out_dir}/step_{step}_eval.json`.
        seed: Sampling seed.

    Returns:
        Flat dict keyed `eval/<bench>_<metric>` (wandb-ready). See module
        docstring for the full key list.
    """
    total_start = time.time()
    directive, regex = _resolve_template(prompt_template)
    # Defensive: tokenizer.eos_token may be None for some tokenizers.
    eos_token = getattr(tokenizer, "eos_token", None)

    # Per-benchmark raw sample storage for JSON dump.
    raw: Dict[str, Any] = {}
    metrics: Dict[str, Optional[Dict[str, Any]]] = {
        "aime24": None, "amc24": None, "math500": None,
        "aime25": None, "amc23": None,
    }

    # --- AIME24 (always run) ----------------------------------------------
    aime24_examples = _load_aime24()
    aime24_prompts = [
        _format_chat(tokenizer, ex["problem"] + directive, enable_thinking)
        for ex in aime24_examples
    ]
    sp = _make_sampling_params(aime24_repeats, max_tokens, eos_token, seed)
    t0 = time.time()
    aime24_outputs = _generate(llm, aime24_prompts, sp)
    aime24_dt = time.time() - t0
    aime24_m, aime24_raw = _score_outputs(aime24_examples, aime24_outputs,
                                          prompt_template, regex)
    aime24_m["gen_seconds"] = round(aime24_dt, 1)
    metrics["aime24"] = aime24_m
    raw["aime24"] = aime24_raw

    # --- AMC24 (skip if parquet missing) ----------------------------------
    amc24_examples = _load_amc24()
    if amc24_examples is None:
        warnings.warn(
            "in_run_dcpo_eval: AMC24 parquet not found at data/amc24.parquet "
            "(and $IN_RUN_AMC24_PATH unset); skipping AMC24."
        )
    else:
        amc24_prompts = [
            _format_chat(tokenizer, ex["problem"] + directive, enable_thinking)
            for ex in amc24_examples
        ]
        sp = _make_sampling_params(amc24_repeats, max_tokens, eos_token, seed)
        t0 = time.time()
        amc24_outputs = _generate(llm, amc24_prompts, sp)
        amc24_dt = time.time() - t0
        amc24_m, amc24_raw = _score_outputs(amc24_examples, amc24_outputs,
                                            prompt_template, regex)
        amc24_m["gen_seconds"] = round(amc24_dt, 1)
        metrics["amc24"] = amc24_m
        raw["amc24"] = amc24_raw

    # --- AIME25 (optional: aime25_repeats == 0 skips; HF fallback) --------
    if aime25_repeats and aime25_repeats > 0:
        aime25_examples = _load_aime25()
        if aime25_examples is None:
            warnings.warn("in_run_dcpo_eval: AIME25 unavailable; skipping.")
        else:
            aime25_prompts = [
                _format_chat(tokenizer, ex["problem"] + directive, enable_thinking)
                for ex in aime25_examples
            ]
            sp = _make_sampling_params(aime25_repeats, max_tokens, eos_token, seed)
            t0 = time.time()
            aime25_outputs = _generate(llm, aime25_prompts, sp)
            aime25_dt = time.time() - t0
            aime25_m, aime25_raw = _score_outputs(aime25_examples, aime25_outputs,
                                                  prompt_template, regex)
            aime25_m["gen_seconds"] = round(aime25_dt, 1)
            metrics["aime25"] = aime25_m
            raw["aime25"] = aime25_raw

    # --- AMC23 (optional: amc23_repeats == 0 skips; HF fallback) ----------
    if amc23_repeats and amc23_repeats > 0:
        amc23_examples = _load_amc23()
        if amc23_examples is None:
            warnings.warn("in_run_dcpo_eval: AMC23 unavailable; skipping.")
        else:
            amc23_prompts = [
                _format_chat(tokenizer, ex["problem"] + directive, enable_thinking)
                for ex in amc23_examples
            ]
            sp = _make_sampling_params(amc23_repeats, max_tokens, eos_token, seed)
            t0 = time.time()
            amc23_outputs = _generate(llm, amc23_prompts, sp)
            amc23_dt = time.time() - t0
            amc23_m, amc23_raw = _score_outputs(amc23_examples, amc23_outputs,
                                                prompt_template, regex)
            amc23_m["gen_seconds"] = round(amc23_dt, 1)
            metrics["amc23"] = amc23_m
            raw["amc23"] = amc23_raw

    # --- MATH-500 (optional: math500_repeats == 0 skips) ------------------
    if math500_repeats and math500_repeats > 0:
        math500_examples = _load_math500(math500_n)
        math500_prompts = [
            _format_chat(tokenizer, ex["problem"] + directive, enable_thinking)
            for ex in math500_examples
        ]
        sp = _make_sampling_params(math500_repeats, max_tokens, eos_token, seed)
        t0 = time.time()
        math500_outputs = _generate(llm, math500_prompts, sp)
        math500_dt = time.time() - t0
        math500_m, math500_raw = _score_outputs(math500_examples, math500_outputs,
                                                prompt_template, regex)
        math500_m["gen_seconds"] = round(math500_dt, 1)
        metrics["math500"] = math500_m
        raw["math500"] = math500_raw

    total_dt = time.time() - total_start

    # --- Flatten for wandb ------------------------------------------------
    flat: Dict[str, float] = {"eval/step": int(step),
                              "eval/total_seconds": _f(round(total_dt, 1))}
    flat.update(_flatten("aime24", metrics["aime24"]))
    flat.update(_flatten("amc24", metrics["amc24"]))
    if aime25_repeats and aime25_repeats > 0:
        flat.update(_flatten("aime25", metrics["aime25"]))
    if amc23_repeats and amc23_repeats > 0:
        flat.update(_flatten("amc23", metrics["amc23"]))
    if math500_repeats and math500_repeats > 0:
        flat.update(_flatten("math500", metrics["math500"]))

    # --- Optional JSON dump (metrics + raw per-sample data) ---------------
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "prompt_template": prompt_template,
            "enable_thinking": bool(enable_thinking),
            "max_tokens": int(max_tokens),
            "sampling": {
                "temperature": _DCPO_TEMPERATURE, "top_p": _DCPO_TOP_P,
                "top_k": _DCPO_TOP_K, "presence_penalty": _DCPO_PRESENCE_PENALTY,
                "aime24_repeats": aime24_repeats, "amc24_repeats": amc24_repeats,
                "aime25_repeats": aime25_repeats, "amc23_repeats": amc23_repeats,
                "math500_repeats": math500_repeats, "math500_n": math500_n,
            },
            "metrics_flat": flat,
            "metrics_per_benchmark": metrics,
            "raw_samples": raw,
            "total_seconds": round(total_dt, 1),
        }
        fp = out_path / f"step_{int(step)}_eval.json"
        with open(fp, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    return flat


__all__ = ["run_in_run_eval"]
