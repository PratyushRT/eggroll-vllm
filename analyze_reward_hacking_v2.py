"""Rigorous reward-hacking forensics v2.

Three checks on the saved in_run_dcpo_eval JSONs:

  1. Prompt-level bootstrap CIs for {acc, ECE, Brier, AUROC, conf_gap}
     (cluster bootstrap: resample problems, not samples)
  2. Paired retention/gain 2x2 matrix between step 0 and step t
  3. Calibration conditioned on retention status:
     {retained, regressed, gained, still_wrong} x mean confidence

A problem is "correct at step t" iff its mean correctness across repeats >= 0.5.

Usage:
  python analyze_reward_hacking_v2.py --eval-dir /tmp/esvpg_rescue/stage1_only \
                                      --out-dir analysis/stage1_rewardhack_v2 \
                                      --baseline-step 0 --bootstrap-B 2000
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


BENCHMARKS = ["aime24", "amc24", "math500"]
N_ECE_BINS = 20
CORRECT_THRESHOLD = 0.5  # problem-level correctness threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_evals(eval_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    paths = sorted(glob.glob(str(eval_dir / "step_*_eval.json")),
                   key=lambda p: int(re.search(r"step_(\d+)_eval", p).group(1)))
    if not paths:
        paths = sorted(glob.glob(str(eval_dir / "step_*.json")),
                       key=lambda p: int(re.search(r"step_(\d+)", p).group(1)))
    for p in paths:
        with open(p) as f:
            payload = json.load(f)
        step = int(payload.get("step", re.search(r"step_(\d+)", p).group(1)))
        raw = payload.get("raw_samples", {})
        for bench in BENCHMARKS:
            for s in (raw.get(bench) or []):
                rows.append({
                    "step": step, "benchmark": bench,
                    "prompt_idx": int(s.get("prompt_idx", -1)),
                    "answer_gt": str(s.get("answer_gt", "")),
                    "verbal_conf": float(s.get("verbal_conf", 0.0)),
                    "correct": int(s.get("correct", 0)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric functions (operate on flat sample arrays)
# ---------------------------------------------------------------------------
def _ece_uniform(conf: np.ndarray, acc: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def _brier(conf: np.ndarray, acc: np.ndarray) -> float:
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - acc) ** 2))


def _auroc(conf: np.ndarray, acc: np.ndarray) -> float:
    # Only defined with both classes present
    if len(conf) == 0 or acc.min() == acc.max():
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(acc.astype(int), conf))
    except Exception:
        return float("nan")


def _conf_gap(conf: np.ndarray, acc: np.ndarray) -> float:
    m_c = conf[acc == 1]
    m_w = conf[acc == 0]
    if len(m_c) == 0 or len(m_w) == 0:
        return float("nan")
    return float(m_c.mean() - m_w.mean())


def _compute_metrics(conf: np.ndarray, acc: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(acc.mean()) if len(acc) else float("nan"),
        "ece": _ece_uniform(conf, acc),
        "brier": _brier(conf, acc),
        "auroc": _auroc(conf, acc),
        "conf_gap": _conf_gap(conf, acc),
    }


# ---------------------------------------------------------------------------
# Check 1 — cluster bootstrap over problems
# ---------------------------------------------------------------------------
def cluster_bootstrap_cis(df: pd.DataFrame, B: int, seed: int,
                          out_dir: Path) -> pd.DataFrame:
    """For each (benchmark, step) resample problem_idx with replacement B times.

    Each resample keeps ALL repeats for the chosen problems (cluster bootstrap).
    Report point estimate + 95% percentile CI for each metric.
    """
    rng = np.random.default_rng(seed)
    out_rows: List[Dict[str, Any]] = []

    for (bench, step), g in df.groupby(["benchmark", "step"]):
        problems = sorted(g["prompt_idx"].unique())
        P = len(problems)
        # Precompute per-problem (conf, correct) lists for speed.
        by_p = {p: g[g["prompt_idx"] == p][["verbal_conf", "correct"]].to_numpy()
                for p in problems}

        # Point estimate on all data
        point = _compute_metrics(g["verbal_conf"].to_numpy(),
                                 g["correct"].to_numpy().astype(float))

        boot: Dict[str, List[float]] = {k: [] for k in point}
        for _ in range(B):
            sel = rng.integers(0, P, size=P)
            chunks = [by_p[problems[i]] for i in sel]
            arr = np.vstack(chunks)
            m = _compute_metrics(arr[:, 0], arr[:, 1].astype(float))
            for k, v in m.items():
                boot[k].append(v)

        row = {"benchmark": bench, "step": int(step), "n_problems": P}
        for k, v in point.items():
            row[f"{k}_point"] = v
            arr = np.array([x for x in boot[k] if np.isfinite(x)])
            if len(arr):
                lo, hi = np.percentile(arr, [2.5, 97.5])
                row[f"{k}_lo"] = float(lo)
                row[f"{k}_hi"] = float(hi)
                row[f"{k}_ci_width"] = float(hi - lo)
            else:
                row[f"{k}_lo"] = row[f"{k}_hi"] = row[f"{k}_ci_width"] = float("nan")
        out_rows.append(row)

    res = pd.DataFrame(out_rows).sort_values(["benchmark", "step"])
    res.to_csv(out_dir / "c1_bootstrap_cis.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Check 2 — paired retention / gain 2x2
# ---------------------------------------------------------------------------
def per_problem_correct(df: pd.DataFrame) -> pd.DataFrame:
    """Problem-level boolean: mean correctness across repeats >= CORRECT_THRESHOLD."""
    g = (df.groupby(["benchmark", "step", "prompt_idx"])
           .agg(acc=("correct", "mean"),
                mean_conf=("verbal_conf", "mean"),
                n_repeats=("correct", "size"),
                mean_conf_wrong=("verbal_conf", lambda x: x[df.loc[x.index, "correct"] == 0].mean()
                                 if (df.loc[x.index, "correct"] == 0).any() else float("nan")),
                mean_conf_right=("verbal_conf", lambda x: x[df.loc[x.index, "correct"] == 1].mean()
                                 if (df.loc[x.index, "correct"] == 1).any() else float("nan")))
           .reset_index())
    g["correct_bool"] = (g["acc"] >= CORRECT_THRESHOLD).astype(int)
    return g


def retention_matrices(pp: pd.DataFrame, baseline_step: int,
                       out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = pp[pp["step"] == baseline_step].set_index(["benchmark", "prompt_idx"])
    for (bench, step), g in pp.groupby(["benchmark", "step"]):
        if step == baseline_step:
            continue
        g_i = g.set_index(["benchmark", "prompt_idx"])
        joined = base[["correct_bool"]].rename(columns={"correct_bool": "base"}) \
                     .join(g_i[["correct_bool"]].rename(columns={"correct_bool": "now"}),
                           how="inner")
        n = len(joined)
        if n == 0:
            continue
        retained = int(((joined["base"] == 1) & (joined["now"] == 1)).sum())
        regressed = int(((joined["base"] == 1) & (joined["now"] == 0)).sum())
        gained = int(((joined["base"] == 0) & (joined["now"] == 1)).sum())
        still_wrong = int(((joined["base"] == 0) & (joined["now"] == 0)).sum())
        row = {
            "benchmark": bench,
            "step": int(step),
            "n_problems": n,
            "retained": retained,
            "regressed": regressed,
            "gained": gained,
            "still_wrong": still_wrong,
            "retention_rate": retained / max(1, retained + regressed),
            "gain_rate": gained / max(1, gained + still_wrong),
            "net_delta": gained - regressed,
            "acc_base": (retained + regressed) / n,
            "acc_now": (retained + gained) / n,
        }
        rows.append(row)
    res = pd.DataFrame(rows).sort_values(["benchmark", "step"])
    res.to_csv(out_dir / "c2_retention_matrix.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Check 3 — calibration conditioned on retention status
# ---------------------------------------------------------------------------
def calibration_by_status(df: pd.DataFrame, pp: pd.DataFrame,
                          baseline_step: int, out_dir: Path) -> pd.DataFrame:
    """For each step t, partition problems into 4 retention buckets based on
    step-0 vs step-t problem-level correctness. Then report, at step t:
      - mean confidence on the problem's samples (all repeats)
      - mean confidence on the WRONG samples at step t (the key signal for
        local reward-hacking — "did conf drop on newly-wrong answers?")
    """
    base = pp[pp["step"] == baseline_step].set_index(["benchmark", "prompt_idx"])
    rows: List[Dict[str, Any]] = []
    for (bench, step), g in pp.groupby(["benchmark", "step"]):
        if step == baseline_step:
            continue
        g_i = g.set_index(["benchmark", "prompt_idx"])
        joined = base[["correct_bool"]].rename(columns={"correct_bool": "base"}) \
                     .join(g_i[["correct_bool"]].rename(columns={"correct_bool": "now"}),
                           how="inner").reset_index()
        joined["status"] = np.select(
            [(joined["base"] == 1) & (joined["now"] == 1),
             (joined["base"] == 1) & (joined["now"] == 0),
             (joined["base"] == 0) & (joined["now"] == 1),
             (joined["base"] == 0) & (joined["now"] == 0)],
            ["retained", "regressed", "gained", "still_wrong"])
        # pull step-t samples
        samples = df[(df["benchmark"] == bench) & (df["step"] == step)]
        # Also step-0 samples for delta view on the regressed group
        samples0 = df[(df["benchmark"] == bench) & (df["step"] == baseline_step)]

        merged = samples.merge(
            joined[["benchmark", "prompt_idx", "status"]],
            on=["benchmark", "prompt_idx"], how="left")
        merged0 = samples0.merge(
            joined[["benchmark", "prompt_idx", "status"]],
            on=["benchmark", "prompt_idx"], how="left")

        for status in ["retained", "regressed", "gained", "still_wrong"]:
            sub = merged[merged["status"] == status]
            sub0 = merged0[merged0["status"] == status]
            if len(sub) == 0:
                continue
            n_problems = sub["prompt_idx"].nunique()
            mean_conf_all = float(sub["verbal_conf"].mean())
            # Key signal: conf on WRONG samples at step t within this bucket
            sub_wrong = sub[sub["correct"] == 0]
            mean_conf_wrong_t = float(sub_wrong["verbal_conf"].mean()) if len(sub_wrong) else float("nan")
            sub0_wrong = sub0[sub0["correct"] == 0]
            mean_conf_wrong_0 = float(sub0_wrong["verbal_conf"].mean()) if len(sub0_wrong) else float("nan")
            # And step-t acc within bucket
            acc_t = float(sub["correct"].mean())
            acc_0 = float(sub0["correct"].mean())
            rows.append({
                "benchmark": bench, "step": int(step), "status": status,
                "n_problems": n_problems, "n_samples": len(sub),
                "acc_base": acc_0, "acc_now": acc_t,
                "mean_conf_all": mean_conf_all,
                "mean_conf_wrong_now": mean_conf_wrong_t,
                "mean_conf_wrong_base": mean_conf_wrong_0,
                "conf_wrong_delta": mean_conf_wrong_t - mean_conf_wrong_0
                    if np.isfinite(mean_conf_wrong_t) and np.isfinite(mean_conf_wrong_0)
                    else float("nan"),
            })

    res = pd.DataFrame(rows).sort_values(["benchmark", "step", "status"])
    res.to_csv(out_dir / "c3_calibration_by_status.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def fmt_ci(p: float, lo: float, hi: float, nd: int = 3) -> str:
    if not np.isfinite(p):
        return "—"
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_summary(c1: pd.DataFrame, c2: pd.DataFrame, c3: pd.DataFrame,
                  baseline_step: int, out_dir: Path) -> None:
    lines: List[str] = ["# Rigorous reward-hacking analysis (v2)\n"]

    # ------------------------------------------------------------------
    lines.append("## Check 1 — Bootstrap CIs (cluster resample over problems)\n")
    lines.append("Point estimate with 95% bootstrap CI. `n_problems` shows eval size "
                 "(tells you how noisy to expect).\n")
    for bench in BENCHMARKS:
        sub = c1[c1["benchmark"] == bench].sort_values("step")
        if sub.empty:
            continue
        lines.append(f"\n### {bench} (n={int(sub['n_problems'].iloc[0])} problems)\n")
        lines.append("| step | acc | ECE | Brier | AUROC | conf_gap |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r['step'])} | "
                f"{fmt_ci(r['acc_point'], r['acc_lo'], r['acc_hi'])} | "
                f"{fmt_ci(r['ece_point'], r['ece_lo'], r['ece_hi'])} | "
                f"{fmt_ci(r['brier_point'], r['brier_lo'], r['brier_hi'])} | "
                f"{fmt_ci(r['auroc_point'], r['auroc_lo'], r['auroc_hi'])} | "
                f"{fmt_ci(r['conf_gap_point'], r['conf_gap_lo'], r['conf_gap_hi'])} |")

    # ------------------------------------------------------------------
    lines.append("\n## Check 2 — Paired retention/gain matrix vs step 0\n")
    lines.append("A problem is 'correct at step t' iff its mean correctness across "
                 f"repeats ≥ {CORRECT_THRESHOLD}. "
                 "`retention_rate = retained/(retained+regressed)`. "
                 "`gain_rate = gained/(gained+still_wrong)`.\n")
    for bench in BENCHMARKS:
        sub = c2[c2["benchmark"] == bench].sort_values("step")
        if sub.empty:
            continue
        lines.append(f"\n### {bench}\n")
        lines.append("| step | retained | regressed | gained | still_wrong "
                     "| retention | gain | net Δ | acc base→now |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {int(r['step'])} | {int(r['retained'])} | {int(r['regressed'])} "
                f"| {int(r['gained'])} | {int(r['still_wrong'])} "
                f"| {r['retention_rate']:.2f} | {r['gain_rate']:.2f} "
                f"| {int(r['net_delta']):+d} "
                f"| {r['acc_base']:.3f}→{r['acc_now']:.3f} |")

    # ------------------------------------------------------------------
    lines.append("\n## Check 3 — Calibration conditioned on retention status\n")
    lines.append("Key signal is the `regressed` row (problems step-0 got right, "
                 "step t gets wrong). `conf_wrong_delta` = conf on wrong samples at "
                 "step t minus conf on wrong samples at step 0, within that group.\n")
    lines.append("- **Negative delta** → calibrated forgetting (model became humble about new misses)")
    lines.append("- **Positive delta** → local reward-hacking (still confident despite newly wrong)\n")
    for bench in BENCHMARKS:
        sub = c3[c3["benchmark"] == bench].sort_values(["step", "status"])
        if sub.empty:
            continue
        lines.append(f"\n### {bench}\n")
        lines.append("| step | status | n | acc base→now | conf on wrong base→now | Δconf on wrong |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            cwb = r["mean_conf_wrong_base"]
            cwn = r["mean_conf_wrong_now"]
            dc = r["conf_wrong_delta"]
            cwb_s = f"{cwb:.3f}" if np.isfinite(cwb) else "—"
            cwn_s = f"{cwn:.3f}" if np.isfinite(cwn) else "—"
            dc_s = f"{dc:+.3f}" if np.isfinite(dc) else "—"
            lines.append(f"| {int(r['step'])} | {r['status']} | {int(r['n_problems'])} "
                         f"| {r['acc_base']:.3f}→{r['acc_now']:.3f} "
                         f"| {cwb_s}→{cwn_s} | {dc_s} |")

    (out_dir / "summary.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="analysis/stage1_rewardhack_v2")
    ap.add_argument("--baseline-step", type=int, default=0)
    ap.add_argument("--bootstrap-B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_evals(eval_dir)
    print(f"[load] {len(df)} rows; steps={sorted(df['step'].unique())}; "
          f"benchmarks={sorted(df['benchmark'].unique())}")

    pp = per_problem_correct(df)

    c1 = cluster_bootstrap_cis(df, B=args.bootstrap_B, seed=args.seed, out_dir=out_dir)
    print(f"[c1] bootstrap done, B={args.bootstrap_B}")
    c2 = retention_matrices(pp, baseline_step=args.baseline_step, out_dir=out_dir)
    print(f"[c2] retention matrices done")
    c3 = calibration_by_status(df, pp, baseline_step=args.baseline_step, out_dir=out_dir)
    print(f"[c3] calibration-by-status done")

    write_summary(c1, c2, c3, args.baseline_step, out_dir)
    print(f"[done] wrote to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
