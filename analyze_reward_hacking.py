"""Post-hoc reward-hacking forensics on saved in_run_dcpo_eval JSONs.

Usage:
    python analyze_reward_hacking.py --eval-dir checkpoints/in_run_dcpo_eval \
                                     --out-dir analysis/rewardhack_stage1

Inputs:
    step_{N}_eval.json files produced by in_run_dcpo_eval.py. Each contains:
      {
        "step": int,
        "raw_samples": {
          "aime24": [{prompt_idx, answer_gt, text, verbal_conf, verbal_parsed,
                      logits_conf, correct}, ...],
          "amc24":  [...],
          "math500": [...],
        },
        ...
      }

Outputs (all markdown + CSV in --out-dir):
    summary.md       — headline findings, one page
    t1_conf_vs_correct.csv    — mean/median conf on correct vs wrong by step/benchmark
    t1_conf_hist.csv          — full conf histogram (0.0-1.0, 20 bins) split by correct
    t3_length_by_step.csv     — response length (chars + word count) by step/benchmark/correct
    t4_conf_by_difficulty.csv — confidence trajectory on easy vs hard problems
    t2_diff_peak_vs_collapse.md  — side-by-side responses for problems the peak step got
                                   right and the collapse step got wrong (first 20 each bench)

Pure pandas/numpy. No wandb or GPU deps. Runs locally.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


BENCHMARKS = ["aime24", "amc24", "math500"]


def _word_count(s: str) -> int:
    return len(re.findall(r"\S+", s)) if s else 0


def load_evals(eval_dir: Path) -> pd.DataFrame:
    """Load all step_*_eval.json files into a long-format DataFrame."""
    rows: List[Dict[str, Any]] = []
    paths = sorted(glob.glob(str(eval_dir / "step_*_eval.json")),
                   key=lambda p: int(re.search(r"step_(\d+)_eval", p).group(1)))
    if not paths:
        # Fallback: some older runs used step_{N}.json (no _eval suffix).
        paths = sorted(glob.glob(str(eval_dir / "step_*.json")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    if not paths:
        raise FileNotFoundError(f"No step_*_eval.json found in {eval_dir}")

    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        step = int(payload.get("step", re.search(r"step_(\d+)", p).group(1)))
        raw = payload.get("raw_samples", {})
        for bench in BENCHMARKS:
            samples = raw.get(bench)
            if not samples:
                continue
            for s in samples:
                text = s.get("text", "") or ""
                rows.append({
                    "step": step,
                    "benchmark": bench,
                    "prompt_idx": int(s.get("prompt_idx", -1)),
                    "answer_gt": str(s.get("answer_gt", "")),
                    "text": text,
                    "n_chars": len(text),
                    "n_words": _word_count(text),
                    "verbal_conf": float(s.get("verbal_conf", 0.0)),
                    "verbal_parsed": bool(s.get("verbal_parsed", False)),
                    "logits_conf": float(s.get("logits_conf", 0.0)),
                    "correct": int(s.get("correct", 0)),
                })
    df = pd.DataFrame(rows)
    print(f"[load] {len(df)} rows across {df['step'].nunique()} eval steps, "
          f"benchmarks: {sorted(df['benchmark'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Test 1 — confidence vs correctness decoupling
# ---------------------------------------------------------------------------
def test1_conf_vs_correct(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    g = (df.groupby(["step", "benchmark", "correct"])
           .agg(n=("verbal_conf", "size"),
                mean_verbal=("verbal_conf", "mean"),
                median_verbal=("verbal_conf", "median"),
                p25_verbal=("verbal_conf", lambda x: np.percentile(x, 25)),
                p75_verbal=("verbal_conf", lambda x: np.percentile(x, 75)),
                mean_logits=("logits_conf", "mean"),
                parse_rate=("verbal_parsed", "mean"))
           .reset_index())
    g.to_csv(out_dir / "t1_conf_vs_correct.csv", index=False)

    # Histogram: bins of 0.05 on verbal_conf, split by correct.
    bins = np.linspace(0.0, 1.0, 21)
    hist_rows = []
    for (step, bench, corr), sub in df.groupby(["step", "benchmark", "correct"]):
        h, _ = np.histogram(sub["verbal_conf"], bins=bins)
        for i, count in enumerate(h):
            hist_rows.append({"step": step, "benchmark": bench, "correct": corr,
                              "bin_lo": bins[i], "bin_hi": bins[i+1],
                              "count": int(count), "frac": count / max(1, len(sub))})
    hist = pd.DataFrame(hist_rows)
    hist.to_csv(out_dir / "t1_conf_hist.csv", index=False)
    return g


# ---------------------------------------------------------------------------
# Test 3 — response length trajectory
# ---------------------------------------------------------------------------
def test3_length(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    g = (df.groupby(["step", "benchmark", "correct"])
           .agg(n=("n_words", "size"),
                mean_words=("n_words", "mean"),
                median_words=("n_words", "median"),
                p25_words=("n_words", lambda x: np.percentile(x, 25)),
                p75_words=("n_words", lambda x: np.percentile(x, 75)),
                mean_chars=("n_chars", "mean"))
           .reset_index())
    g.to_csv(out_dir / "t3_length_by_step.csv", index=False)
    return g


# ---------------------------------------------------------------------------
# Test 4 — confidence trajectory on easy vs hard problems
# ---------------------------------------------------------------------------
def test4_conf_by_difficulty(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Define "easy" = step 0 avg correct >= 0.5 on that (benchmark, prompt_idx);
    # "hard" = step 0 avg correct < 0.5.
    step0 = df[df["step"] == df["step"].min()]
    diff = (step0.groupby(["benchmark", "prompt_idx"])["correct"].mean()
                 .reset_index()
                 .rename(columns={"correct": "step0_acc"}))
    diff["difficulty"] = np.where(diff["step0_acc"] >= 0.5, "easy", "hard")
    merged = df.merge(diff[["benchmark", "prompt_idx", "difficulty"]],
                      on=["benchmark", "prompt_idx"], how="left")

    g = (merged.groupby(["step", "benchmark", "difficulty"])
               .agg(n=("verbal_conf", "size"),
                    acc=("correct", "mean"),
                    mean_verbal=("verbal_conf", "mean"),
                    mean_logits=("logits_conf", "mean"),
                    mean_words=("n_words", "mean"))
               .reset_index())
    g.to_csv(out_dir / "t4_conf_by_difficulty.csv", index=False)
    return g


# ---------------------------------------------------------------------------
# Test 2 — side-by-side responses for peak-right / collapse-wrong problems
# ---------------------------------------------------------------------------
def test2_peak_vs_collapse(df: pd.DataFrame, peak_step: int, collapse_step: int,
                           out_dir: Path, max_per_bench: int = 10) -> None:
    lines: List[str] = []
    lines.append(f"# Peak-right vs collapse-wrong responses\n")
    lines.append(f"Peak step: {peak_step}  |  Collapse step: {collapse_step}\n")

    for bench in BENCHMARKS:
        sub = df[df["benchmark"] == bench]
        if sub.empty:
            continue
        peak = (sub[sub["step"] == peak_step]
                  .groupby("prompt_idx")
                  .agg(peak_acc=("correct", "mean"),
                       peak_conf=("verbal_conf", "mean"),
                       peak_text=("text", "first"),
                       answer_gt=("answer_gt", "first")))
        coll = (sub[sub["step"] == collapse_step]
                  .groupby("prompt_idx")
                  .agg(coll_acc=("correct", "mean"),
                       coll_conf=("verbal_conf", "mean"),
                       coll_text=("text", "first")))
        merged = peak.join(coll, how="inner")
        regressed = merged[(merged["peak_acc"] >= 0.5) & (merged["coll_acc"] < 0.5)]
        regressed = regressed.sort_values("coll_conf", ascending=False).head(max_per_bench)

        lines.append(f"\n## {bench}: {len(regressed)} regressed prompts shown\n")
        for idx, row in regressed.iterrows():
            lines.append(f"\n### prompt_idx={idx}  gt=`{row['answer_gt']}`\n")
            lines.append(f"- Peak (step {peak_step}): acc={row['peak_acc']:.2f}, "
                         f"conf={row['peak_conf']:.3f}")
            lines.append(f"- Collapse (step {collapse_step}): acc={row['coll_acc']:.2f}, "
                         f"conf={row['coll_conf']:.3f}\n")
            lines.append(f"<details><summary>Peak response</summary>\n\n```\n"
                         f"{row['peak_text'][:3000]}\n```\n</details>\n")
            lines.append(f"<details><summary>Collapse response</summary>\n\n```\n"
                         f"{row['coll_text'][:3000]}\n```\n</details>\n")

    (out_dir / "t2_diff_peak_vs_collapse.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def write_summary(df: pd.DataFrame, t1: pd.DataFrame, t3: pd.DataFrame,
                  t4: pd.DataFrame, out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# Reward-hacking forensics — headline\n")
    lines.append(f"Eval steps analyzed: {sorted(df['step'].unique())}")
    lines.append(f"Benchmarks: {sorted(df['benchmark'].unique())}")
    lines.append(f"Total rollouts: {len(df)}\n")

    step_min = int(df["step"].min())
    step_max = int(df["step"].max())

    lines.append("## T1 — Confidence on wrong answers (verbal, mean)\n")
    lines.append("| benchmark | step | conf(correct) | conf(wrong) | decoupling gap |")
    lines.append("|---|---|---|---|---|")
    for bench in BENCHMARKS:
        for step in sorted(df["step"].unique()):
            rows = t1[(t1["benchmark"] == bench) & (t1["step"] == step)]
            c = rows[rows["correct"] == 1]["mean_verbal"]
            w = rows[rows["correct"] == 0]["mean_verbal"]
            if c.empty or w.empty:
                continue
            gap = float(c.iloc[0]) - float(w.iloc[0])
            lines.append(f"| {bench} | {step} | {float(c.iloc[0]):.3f} | "
                         f"{float(w.iloc[0]):.3f} | {gap:+.3f} |")

    lines.append("\n**Interpretation:** a shrinking (or negative) decoupling gap "
                 "between steps means the model is getting equally confident on "
                 "wrong and right answers — the reward-hacking signature.\n")

    lines.append("## T3 — Response length on correct vs wrong (mean words)\n")
    lines.append("| benchmark | step | words(correct) | words(wrong) |")
    lines.append("|---|---|---|---|")
    for bench in BENCHMARKS:
        for step in sorted(df["step"].unique()):
            rows = t3[(t3["benchmark"] == bench) & (t3["step"] == step)]
            c = rows[rows["correct"] == 1]["mean_words"]
            w = rows[rows["correct"] == 0]["mean_words"]
            if c.empty or w.empty:
                continue
            lines.append(f"| {bench} | {step} | {float(c.iloc[0]):.0f} | "
                         f"{float(w.iloc[0]):.0f} |")
    lines.append("\n**Interpretation:** if both correct and wrong response lengths "
                 "collapse in lockstep, the population found a short-output exploit.\n")

    lines.append("## T4 — Confidence by difficulty (step-0 easy vs hard)\n")
    lines.append("| benchmark | step | easy conf | easy acc | hard conf | hard acc |")
    lines.append("|---|---|---|---|---|---|")
    for bench in BENCHMARKS:
        for step in sorted(df["step"].unique()):
            rows = t4[(t4["benchmark"] == bench) & (t4["step"] == step)]
            e = rows[rows["difficulty"] == "easy"]
            h = rows[rows["difficulty"] == "hard"]
            if e.empty or h.empty:
                continue
            lines.append(f"| {bench} | {step} | {float(e['mean_verbal'].iloc[0]):.3f} "
                         f"| {float(e['acc'].iloc[0]):.3f} "
                         f"| {float(h['mean_verbal'].iloc[0]):.3f} "
                         f"| {float(h['acc'].iloc[0]):.3f} |")
    lines.append("\n**Interpretation:** confidence on hard problems should go "
                 "*down* with real calibration learning; if it stays high or rises "
                 "while hard-accuracy doesn't improve, it's reward-hacking.\n")

    lines.append(f"\nSee `t2_diff_peak_vs_collapse.md` for side-by-side responses "
                 f"on problems the peak step got right and the final step got wrong.\n")

    (out_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=str, required=True,
                    help="Directory containing step_*_eval.json files")
    ap.add_argument("--out-dir", type=str, default="analysis/rewardhack",
                    help="Where to write CSVs and markdown")
    ap.add_argument("--peak-step", type=int, default=60,
                    help="Step to treat as 'peak' for T2 diff")
    ap.add_argument("--collapse-step", type=int, default=70,
                    help="Step to treat as 'collapse' for T2 diff")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_evals(eval_dir)

    t1 = test1_conf_vs_correct(df, out_dir)
    t3 = test3_length(df, out_dir)
    t4 = test4_conf_by_difficulty(df, out_dir)

    # Only run T2 if both peak and collapse steps are present.
    steps = set(df["step"].unique().tolist())
    if args.peak_step in steps and args.collapse_step in steps:
        test2_peak_vs_collapse(df, args.peak_step, args.collapse_step, out_dir)
    else:
        print(f"[T2] skipping — need steps {args.peak_step} and "
              f"{args.collapse_step}, have {sorted(steps)}")

    write_summary(df, t1, t3, t4, out_dir)
    print(f"[done] wrote analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
