"""Precompute the anchor set of DeepScaler-40K training prompts.

Anchors are prompts where the base model (Qwen3-8B by default) achieves at least
the threshold accuracy (default 0.75) over K=4 sampled generations at temp=0.7.
These prompts form the "retention anchor" subset used by the A1/A2 triage runs:
we penalize the ES loss for regressing on them (retention penalty) and for
assigning high confidence when wrong on them (wrong-conf-anchor penalty).

The DeepScaler loader in `tasks.py` shuffles the dataset with the training seed
before use. We replicate that shuffle here so the indices in the output JSON
line up with `dataset[idx]` inside training when the same seed is passed.

Grading uses DCPO's is_equiv (dcpo_grader.compute_score > 0.5) to match what
the reward function will use during training.

Usage:
    python precompute_anchor_set.py \\
        --model-name Qwen/Qwen3-8B \\
        --seed 42 \\
        --num-prompts 4000 \\
        --samples-per-prompt 4 \\
        --temperature 0.7 \\
        --max-tokens 2048 \\
        --prompt-template conf_tags \\
        --out data/anchor_set_deepscaler40k_seed42.json

Output schema:
    {
      "model_name": "Qwen/Qwen3-8B",
      "dataset": "axon-rl/DeepScaleR-40K",
      "dataset_split": "train",
      "shuffle_seed": 42,
      "num_prompts_scanned": 4000,
      "samples_per_prompt": 4,
      "temperature": 0.7,
      "max_tokens": 2048,
      "prompt_template": "conf_tags",
      "enable_thinking": false,
      "threshold": 0.75,
      "anchor_indices": [0, 3, 5, 12, ...],          # indices into shuffled dataset
      "per_prompt_accuracy": [0.0, 0.5, 1.0, ...],   # length = num_prompts_scanned
      "n_anchors": 1234,
      "anchor_fraction": 0.3085,
      "wall_seconds": 1234.5,
      "timestamp": "2026-04-21T00:00:00Z"
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Local imports — DCPO grader + prompt helpers live at the repo root.
from dcpo_grader import compute_score as dcpo_compute_score, last_boxed_only_string, remove_boxed
from dcpo_prompt import (
    CONF_RE,
    DCPO_CONFIDENCE_DIRECTIVE,
    strip_conf,
)

# conf_tags template (matches the one used during training).
CONF_TAGS_DIRECTIVE = (
    "\n\nPlease reason step-by-step, put your final answer within \\boxed{}, "
    "and report your confidence on a separate line using the tags "
    "<conf>...</conf> where the value is a float in [0, 1]. "
    "Example: <conf>0.72</conf>"
)


def build_prompt(problem: str, directive: str, tokenizer, enable_thinking: bool) -> str:
    content = f"{problem}{directive}"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )


def grade_generation(text: str, gt_answer) -> bool:
    """DCPO-grader match, with both CONFIDENCE line and <conf> tags stripped."""
    cleaned = strip_conf(text)
    cleaned = re.sub(r"<conf>\s*[0-9.]+\s*</conf>", "", cleaned).strip()
    if isinstance(gt_answer, (list, tuple)):
        return any(dcpo_compute_score(cleaned, str(gt)) > 0.5 for gt in gt_answer)
    gt_str = str(gt_answer) if not isinstance(gt_answer, str) else gt_answer
    return dcpo_compute_score(cleaned, gt_str) > 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3-8B")
    ap.add_argument("--dataset", default="axon-rl/DeepScaleR-40K")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed — MUST match the training seed so indices align.")
    ap.add_argument("--num-prompts", type=int, default=4000,
                    help="How many shuffled prompts to scan for anchors.")
    ap.add_argument("--samples-per-prompt", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="A prompt is an anchor iff per-prompt acc >= threshold.")
    ap.add_argument("--prompt-template", choices=["dcpo_verbose", "conf_tags"], default="conf_tags")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Pass enable_thinking=True to the Qwen3 chat template (default False).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out", required=True, type=str, help="Output JSON path.")
    ap.add_argument("--resume", action="store_true",
                    help="If output file exists with partial data, resume from there.")
    args = ap.parse_args()

    start = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load + shuffle dataset (matches tasks.py MathTask.__init__) ---
    print(f"[anchor] loading {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split).shuffle(seed=args.seed)
    n_scan = min(args.num_prompts, len(ds))
    print(f"[anchor] scanning first {n_scan}/{len(ds)} prompts (shuffle seed={args.seed})",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    directive = CONF_TAGS_DIRECTIVE if args.prompt_template == "conf_tags" else DCPO_CONFIDENCE_DIRECTIVE

    # --- build all prompts + answers ---
    prompts, answers = [], []
    for i in range(n_scan):
        row = ds[i]
        problem = row["problem"]
        answer = row["answer"]
        prompts.append(build_prompt(problem, directive, tokenizer, args.enable_thinking))
        answers.append(answer)

    # --- load vLLM ---
    from vllm import LLM, SamplingParams

    print(f"[anchor] loading vLLM engine for {args.model_name}", flush=True)
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_lora=False,
        max_model_len=max(args.max_tokens + 512, 4096),
    )

    sp = SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
    )

    # --- generate ---
    print(f"[anchor] generating {n_scan} prompts x {args.samples_per_prompt} samples", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_wall = time.time() - t0
    print(f"[anchor] generation wall={gen_wall:.1f}s ({gen_wall/n_scan*1000:.1f} ms/prompt)",
          flush=True)

    # --- grade ---
    per_prompt_acc = np.zeros(n_scan, dtype=np.float64)
    per_prompt_correct = np.zeros(n_scan, dtype=np.int32)
    for i, out in enumerate(outputs):
        n_correct = 0
        for comp in out.outputs:
            text = comp.text
            if grade_generation(text, answers[i]):
                n_correct += 1
        per_prompt_correct[i] = n_correct
        per_prompt_acc[i] = n_correct / max(len(out.outputs), 1)

    anchor_mask = per_prompt_acc >= args.threshold
    anchor_indices = np.nonzero(anchor_mask)[0].tolist()

    print(f"[anchor] anchors: {len(anchor_indices)}/{n_scan} "
          f"({100.0 * len(anchor_indices) / n_scan:.1f}%) at threshold={args.threshold}",
          flush=True)

    payload = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "dataset_split": args.split,
        "shuffle_seed": args.seed,
        "num_prompts_scanned": n_scan,
        "samples_per_prompt": args.samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "prompt_template": args.prompt_template,
        "enable_thinking": bool(args.enable_thinking),
        "threshold": args.threshold,
        "anchor_indices": anchor_indices,
        "per_prompt_accuracy": per_prompt_acc.tolist(),
        "per_prompt_correct_count": per_prompt_correct.tolist(),
        "n_anchors": len(anchor_indices),
        "anchor_fraction": float(len(anchor_indices)) / max(n_scan, 1),
        "wall_seconds": time.time() - start,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[anchor] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
