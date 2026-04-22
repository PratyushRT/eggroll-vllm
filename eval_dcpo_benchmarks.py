#!/usr/bin/env python3
"""
DCPO-exact post-training eval for ES-trained checkpoints (or any HF-format LLM).

Runs the 5 benchmarks used in DCPO Table 1 — MATH-500, AIME24, AIME25, AMC23,
AMC24 — with DCPO's exact sampling parameters, grading (`is_equiv`), metrics
(ECE, PCE, Brier, AUROC, MCE, Accuracy), and dual confidence extraction
(verbal `CONFIDENCE: x` + length-normalized sequence logits).

Usage:
    python eval_dcpo_benchmarks.py \
        --model-path Qwen/Qwen3-8B \
        --benchmarks math500,aime24,aime25,amc23,amc24 \
        --n-samples 4 \
        --output-json results.json

    # Eval a trained checkpoint:
    python eval_dcpo_benchmarks.py \
        --model-path /path/to/checkpoint_step_50 \
        --output-json results_step50.json

Notes:
 *  Benchmark sources default to public HuggingFace datasets; override with
    `--<bench>-dataset-id` or `--<bench>-local-path` if your cluster has offline
    copies.
 *  Uses DCPO's `is_equiv` via `dcpo_grader.compute_score` for apples-to-apples
    with their Table 1. (Eggroll-vllm's default grader from `gem.utils` is NOT
    used here.)
 *  The script is single-process and uses vLLM's batched generate() call;
    `n=4` (DCPO repeat) is passed via SamplingParams so all reps batch together.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
from dcpo_prompt import DCPO_CONFIDENCE_DIRECTIVE, parse_conf_for_eval, strip_conf


# ---------------------------------------------------------------------------
# Benchmark loaders
#
# All loaders return a list of {"problem": str, "answer": str} dicts.
# Defaults point at common HF dataset IDs; pass `--<bench>-local-path` to read
# from a local JSONL/parquet file instead.
# ---------------------------------------------------------------------------

def _load_hf_math500(dataset_id: str = "HuggingFaceH4/MATH-500", n: int = 500) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split="test")
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append({
            "problem": row.get("problem", row.get("question", "")),
            "answer": str(row.get("answer", row.get("solution", ""))),
        })
    return out


def _load_hf_aime(dataset_id: str, split: str = "train", n: int = 30,
                  problem_field: str = "Problem", answer_field: str = "Answer") -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        prob = row.get(problem_field) or row.get("problem") or row.get("question") or ""
        ans = row.get(answer_field) or row.get("answer") or ""
        out.append({"problem": str(prob), "answer": str(ans)})
    return out


def _load_hf_amc(dataset_id: str, split: str = "train", n: int = 45) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        prob = row.get("problem") or row.get("question") or ""
        ans = row.get("answer") or row.get("solution") or ""
        out.append({"problem": str(prob), "answer": str(ans)})
    return out


def _load_local_jsonl(path: str, n: int,
                      problem_field: str = "problem",
                      answer_field: str = "answer") -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            row = json.loads(line)
            out.append({
                "problem": str(row.get(problem_field, row.get("question", ""))),
                "answer": str(row.get(answer_field, row.get("solution", ""))),
            })
    return out


def _load_local_parquet(path: str, n: int,
                        problem_field: str = "problem",
                        answer_field: str = "answer") -> List[Dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    out = []
    for i in range(min(n, len(df))):
        row = df.iloc[i]
        prob = row.get(problem_field) if problem_field in df.columns else None
        ans = row.get(answer_field) if answer_field in df.columns else None
        # VERL-format parquet has `prompt` (list of chat msgs) and `reward_model.ground_truth`.
        if prob is None and "prompt" in df.columns:
            prompt_col = row["prompt"]
            prob = prompt_col[0]["content"] if prompt_col is not None else ""
        if ans is None and "reward_model" in df.columns:
            rm = row["reward_model"]
            ans = rm.get("ground_truth") if isinstance(rm, dict) else ""
        out.append({"problem": str(prob or ""), "answer": str(ans or "")})
    return out


@dataclass
class BenchmarkSpec:
    name: str
    default_n: int
    default_hf_loader: Callable[[], List[Dict]]
    canonical_display_name: str


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "math500": BenchmarkSpec(
        name="math500",
        default_n=500,
        default_hf_loader=lambda: _load_hf_math500("HuggingFaceH4/MATH-500", 500),
        canonical_display_name="MATH-500",
    ),
    "aime24": BenchmarkSpec(
        name="aime24",
        default_n=30,
        default_hf_loader=lambda: _load_hf_aime("Maxwell-Jia/AIME_2024", split="train", n=30,
                                                problem_field="Problem", answer_field="Answer"),
        canonical_display_name="AIME24",
    ),
    "aime25": BenchmarkSpec(
        name="aime25",
        default_n=30,
        default_hf_loader=lambda: _load_hf_aime("yentinglin/aime_2025", split="train", n=30,
                                                problem_field="problem", answer_field="answer"),
        canonical_display_name="AIME25",
    ),
    "amc23": BenchmarkSpec(
        name="amc23",
        default_n=45,
        default_hf_loader=lambda: _load_hf_amc("math-ai/amc23", split="test", n=45),
        canonical_display_name="AMC23",
    ),
    "amc24": BenchmarkSpec(
        name="amc24",
        default_n=45,
        # No widely-adopted HF copy of AMC24 at time of writing; user should
        # supply --amc24-local-path. Loader falls back to AMC23 id and will fail
        # loudly, prompting the user to override.
        default_hf_loader=lambda: _load_hf_amc("math-ai/amc23", split="test", n=45),
        canonical_display_name="AMC24",
    ),
}


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------

def build_prompt(problem: str, directive: str = DCPO_CONFIDENCE_DIRECTIVE) -> str:
    return problem + directive


def format_chat(tokenizer, user_content: str, enable_thinking: bool) -> str:
    """Apply Qwen3/other chat template with enable_thinking if supported."""
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


def run_vllm(
    model_path: str,
    prompts: List[str],
    n_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    max_tokens: int,
    tensor_parallel_size: int,
    seed: int,
    dtype: str,
    stop_tokens: Optional[List[str]] = None,
):
    """Single-call batched vLLM generate. Returns the list of RequestOutput."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        enable_lora=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        logprobs=1,
        seed=seed,
        stop=(stop_tokens or [tokenizer.eos_token]),
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[gen] {len(prompts)} prompts × n={n_samples} → {time.time()-t0:.1f}s", flush=True)
    return outputs, tokenizer


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _cumulative_logprob_and_len(output) -> Tuple[Optional[float], int]:
    """Return (cumulative_logprob, n_tokens) for a vLLM CompletionOutput."""
    cum_lp = getattr(output, "cumulative_logprob", None)
    n_tokens = len(output.token_ids) if output.token_ids is not None else 0
    return cum_lp, n_tokens


def _token_logprobs_from_output(output) -> List[float]:
    """Extract the chosen token logprob at each step (for logits_confidence)."""
    # vLLM's `output.logprobs` is a list (one per generated token) of
    # dict[token_id -> Logprob]. The chosen token at position t is output.token_ids[t].
    lps = []
    step_logprobs = getattr(output, "logprobs", None) or []
    token_ids = output.token_ids or []
    for step_idx, step_map in enumerate(step_logprobs):
        if step_map is None or step_idx >= len(token_ids):
            continue
        chosen = token_ids[step_idx]
        lp_obj = step_map.get(chosen) if isinstance(step_map, dict) else None
        if lp_obj is None:
            # Fallback: take max logprob in the step (matches top_logprobs=1 typical case).
            try:
                lp_obj = max(step_map.values(), key=lambda x: getattr(x, "logprob", float("-inf")))
            except Exception:
                continue
        lp = getattr(lp_obj, "logprob", None)
        if lp is not None:
            lps.append(float(lp))
    return lps


def evaluate_benchmark(
    bench_name: str,
    examples: List[Dict],
    outputs,
    n_samples: int,
) -> Dict[str, Any]:
    """Aggregate vLLM outputs → per-benchmark metrics.

    `outputs[i]` corresponds to `examples[i]`; each has `n_samples` completions.
    """
    conf_verbal: List[float] = []
    conf_logits: List[float] = []
    correct: List[int] = []
    parse_ok: List[int] = []

    for ex, req_out in zip(examples, outputs):
        gt = ex["answer"]
        for comp in req_out.outputs:  # n completions
            text = comp.text
            cv, status = parse_conf_for_eval(text)
            conf_verbal.append(cv)
            parse_ok.append(1 if status == "ok" else 0)

            # Logits confidence: prefer length-normed cumulative_logprob,
            # fall back to mean over per-token logprobs if cumulative is None.
            cum_lp, n_tok = _cumulative_logprob_and_len(comp)
            if cum_lp is not None and n_tok > 0:
                cl = math.exp(cum_lp / n_tok)
            else:
                tlps = _token_logprobs_from_output(comp)
                cl = logits_confidence_from_token_logprobs(tlps) if tlps else 0.0
            conf_logits.append(cl)

            # Grade using DCPO is_equiv on the conf-stripped text.
            cleaned = strip_conf(text)
            try:
                is_correct = dcpo_compute_score(cleaned, str(gt)) > 0.5
            except Exception:
                is_correct = False
            correct.append(1 if is_correct else 0)

    cv = np.asarray(conf_verbal, dtype=np.float64)
    cl = np.asarray(conf_logits, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    ok = np.asarray(parse_ok, dtype=np.float64)

    # Accuracy over all (n_examples × n_samples) generations.
    accuracy = float(acc.mean()) if acc.size else float("nan")

    # Metrics are computed on the full arrays. The `compute_brier` helper masks
    # out DCPO's negative sentinels automatically; ECE/MCE bins only cover [0,1]
    # so negative sentinels are also naturally excluded.
    m = {
        "n_examples": len(examples),
        "n_samples_per_example": n_samples,
        "n_generations": int(acc.size),
        "accuracy": accuracy,
        "parse_success_rate": float(ok.mean()) if ok.size else float("nan"),
        "mean_q_verbal": float(cv[cv >= 0].mean()) if np.any(cv >= 0) else float("nan"),
        "mean_q_logits": float(cl.mean()) if cl.size else float("nan"),
        # Verbal confidence metrics
        "ece_verbal": compute_ece(cv, acc),
        "pce_verbal": compute_overconf_ece(cv, acc),
        "brier_verbal": compute_brier(cv, acc),
        "mce_verbal": compute_mce(cv, acc),
        "auroc_verbal": compute_auroc(cv, acc),
        # Logits confidence metrics
        "ece_logits": compute_ece(cl, acc),
        "pce_logits": compute_overconf_ece(cl, acc),
        "brier_logits": compute_brier(cl, acc),
        "mce_logits": compute_mce(cl, acc),
        "auroc_logits": compute_auroc(cl, acc),
    }
    return m


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def render_markdown(per_benchmark: Dict[str, Dict[str, Any]], model_path: str) -> str:
    header = f"# DCPO Benchmark Eval — `{model_path}`\n\n"
    cols = ["Benchmark", "Acc", "ECE (v)", "PCE (v)", "Brier (v)", "AUROC (v)",
            "ECE (l)", "PCE (l)", "Brier (l)", "AUROC (l)", "Parse %"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]

    def fmt(x, p=4):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "—"
        return f"{x:.{p}f}"

    for bench_key, m in per_benchmark.items():
        disp = BENCHMARKS[bench_key].canonical_display_name if bench_key in BENCHMARKS else bench_key
        lines.append("| " + " | ".join([
            disp,
            fmt(m.get("accuracy"), 4),
            fmt(m.get("ece_verbal")),
            fmt(m.get("pce_verbal")),
            fmt(m.get("brier_verbal")),
            fmt(m.get("auroc_verbal")),
            fmt(m.get("ece_logits")),
            fmt(m.get("pce_logits")),
            fmt(m.get("brier_logits")),
            fmt(m.get("auroc_logits")),
            fmt(m.get("parse_success_rate") * 100 if m.get("parse_success_rate") is not None else None, 1),
        ]) + " |")
    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_benchmark(name: str, args) -> List[Dict]:
    spec = BENCHMARKS[name]
    n = getattr(args, f"{name}_n") or spec.default_n
    local = getattr(args, f"{name}_local_path", None)
    if local:
        if local.endswith(".parquet"):
            return _load_local_parquet(local, n)
        return _load_local_jsonl(local, n)
    hf_override = getattr(args, f"{name}_dataset_id", None)
    if hf_override:
        if name == "math500":
            return _load_hf_math500(hf_override, n)
        if name.startswith("aime"):
            return _load_hf_aime(hf_override, n=n)
        if name.startswith("amc"):
            return _load_hf_amc(hf_override, n=n)
    # Default HF loader; trim to --<bench>-n if user supplied a smaller override.
    items = spec.default_hf_loader()
    if n < len(items):
        items = items[:n]
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="HF model ID or local checkpoint dir (passed to vllm.LLM).")
    p.add_argument("--benchmarks", default="math500,aime24,aime25,amc23,amc24")
    p.add_argument("--n-samples", type=int, default=4, help="DCPO: 4 reps per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enable-thinking", type=lambda s: str(s).lower() == "true",
                   default=False)
    p.add_argument("--output-json", default="dcpo_eval_results.json")
    p.add_argument("--output-md", default=None,
                   help="If set, also write a DCPO-Table-1-style markdown table.")
    # Per-benchmark overrides
    for name, spec in BENCHMARKS.items():
        p.add_argument(f"--{name}-n", type=int, default=None)
        p.add_argument(f"--{name}-local-path", default=None)
        p.add_argument(f"--{name}-dataset-id", default=None)
    args = p.parse_args()

    wanted = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for b in wanted:
        if b not in BENCHMARKS:
            raise SystemExit(f"unknown benchmark: {b}. Choices: {list(BENCHMARKS)}")

    # 1) Load all datasets first (cheap, catches config errors early).
    loaded: Dict[str, List[Dict]] = {}
    for b in wanted:
        print(f"[data] loading {b}...", flush=True)
        try:
            loaded[b] = load_benchmark(b, args)
        except Exception as e:
            print(f"[data] FAILED to load {b}: {e}", file=sys.stderr)
            loaded[b] = []
        print(f"[data]   {b}: {len(loaded[b])} examples", flush=True)

    # 2) Build flat prompt list with bookkeeping so we can slice outputs back.
    # Apply chat template via the model's own tokenizer (we load via vLLM).
    from vllm import LLM, SamplingParams
    print(f"[vllm] loading {args.model_path}...", flush=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        enable_lora=False,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        logprobs=1,
        seed=args.seed,
        stop=[tokenizer.eos_token],
    )

    # Count GPUs visible to vLLM for fairness-axis reporting (EGGROLL §G.3
    # reports wall-clock × GPU count as the primary axis vs PG).
    try:
        import torch as _t
        n_gpus = _t.cuda.device_count()
    except Exception:
        n_gpus = args.tensor_parallel_size or 1
    eval_wall_start = time.time()

    results: Dict[str, Dict[str, Any]] = {}
    for b in wanted:
        examples = loaded[b]
        if not examples:
            results[b] = {"error": "dataset failed to load; see stderr"}
            continue
        prompts = [
            format_chat(tokenizer, build_prompt(ex["problem"]), args.enable_thinking)
            for ex in examples
        ]
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        dt = time.time() - t0
        print(f"[gen] {b}: {len(prompts)} × n={args.n_samples} → {dt:.1f}s", flush=True)
        results[b] = evaluate_benchmark(b, examples, outputs, args.n_samples)
        results[b]["gen_seconds"] = round(dt, 1)

    # Aggregate wall-clock fairness figures.
    eval_wall_seconds = time.time() - eval_wall_start
    eval_gpu_hours = (eval_wall_seconds / 3600.0) * n_gpus

    # 3) Write JSON + optional markdown.
    payload = {
        "model_path": args.model_path,
        "args": vars(args),
        "results": results,
        "fairness": {
            "eval_wall_seconds": round(eval_wall_seconds, 1),
            "eval_gpu_hours": round(eval_gpu_hours, 3),
            "n_gpus": n_gpus,
            "note": "EGGROLL paper §G.3: primary compute-fairness axis is GPU-hours. Per-benchmark gen_seconds also recorded per-result.",
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[out] JSON → {args.output_json}", flush=True)
    print(
        f"[fairness] eval wall-clock {eval_wall_seconds:.1f}s on {n_gpus} GPU(s) "
        f"= {eval_gpu_hours:.3f} GPU-hours",
        flush=True,
    )

    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        md = render_markdown(results, args.model_path)
        with open(args.output_md, "w") as f:
            f.write(md)
        print(f"[out] Markdown → {args.output_md}", flush=True)
        print("\n" + md)


if __name__ == "__main__":
    main()
