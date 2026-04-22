"""
DCPO calibration metrics — verbatim port of `calibration_main.py` metric
functions from https://github.com/icip-cas/DCPO. Kept identical (uniform
15-bin default, same masking, same Brier formulation) so that our numbers
are directly comparable to DCPO's Table 1.

Functions:
    compute_ece(conf, acc, n_bins=15)        -> float   (uniform binning)
    compute_overconf_ece(conf, acc, n_bins=15) -> float (a.k.a. PCE)
    compute_brier(conf, acc)                 -> float
    compute_mce(conf, acc, n_bins=15)        -> float
    compute_auroc(conf, acc)                 -> float   (sklearn)
    logits_confidence_from_token_logprobs(logprobs) -> float
    length_normed_seq_prob(cumulative_logprob, n_tokens) -> float
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _as_arrays(confidences, accuracies):
    """Coerce inputs to 1-D numpy arrays of the same shape (float)."""
    c = np.asarray(confidences, dtype=np.float64).ravel()
    a = np.asarray(accuracies, dtype=np.float64).ravel()
    if c.shape != a.shape:
        raise ValueError(f"shape mismatch: conf {c.shape} vs acc {a.shape}")
    return c, a


def compute_ece(confidences, accuracies, n_bins: int = 15) -> float:
    """Uniform-bin ECE: Σ P(bin) · |acc − conf|.

    Matches DCPO `calibration_main.py::compute_ece` verbatim.
    First bin is closed on both ends; subsequent bins are (lo, hi].
    """
    c, a = _as_arrays(confidences, accuracies)
    if c.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c <= hi) if i == 0 else (c > lo) & (c <= hi)
        if np.any(mask):
            acc_i = a[mask].mean()
            conf_i = c[mask].mean()
            ece += np.abs(acc_i - conf_i) * mask.mean()
    return float(ece)


def compute_overconf_ece(confidences, accuracies, n_bins: int = 15) -> float:
    """Over-confidence ECE (a.k.a. PCE): only accumulate bins where conf > acc.

    Matches DCPO `calibration_main.py::compute_overconf_ece` verbatim.
    """
    c, a = _as_arrays(confidences, accuracies)
    if c.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c <= hi) if i == 0 else (c > lo) & (c <= hi)
        if np.any(mask):
            acc_i = a[mask].mean()
            conf_i = c[mask].mean()
            if conf_i > acc_i:
                ece += (conf_i - acc_i) * mask.mean()
    return float(ece)


def compute_brier(confidences, accuracies) -> float:
    """Brier score over valid (q ∈ [0,1]) samples only: mean((q − acc)²).

    Matches DCPO `calibration_main.py::compute_BS` verbatim (DCPO masks out
    malformed negative confidences from this computation).
    """
    c, a = _as_arrays(confidences, accuracies)
    mask = (c >= 0) & (c <= 1)
    if not np.any(mask):
        return float("nan")
    return float(np.mean((c[mask] - a[mask]) ** 2))


def compute_mce(confidences, accuracies, n_bins: int = 15) -> float:
    """Maximum Calibration Error: max over bins of |acc − conf|.

    Matches DCPO `calibration_main.py::compute_MCE` verbatim.
    """
    c, a = _as_arrays(confidences, accuracies)
    if c.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (c >= lo) & (c <= hi) if i == 0 else (c > lo) & (c <= hi)
        if np.any(mask):
            mce = max(mce, float(np.abs(a[mask].mean() - c[mask].mean())))
    return float(mce)


def compute_auroc(confidences, accuracies) -> float:
    """sklearn AUROC, with `accuracies` as labels and `confidences` as scores.

    Matches DCPO `calibration_main.py::compute_auroc` verbatim.
    Returns NaN if only one class is present (undefined).
    """
    from sklearn.metrics import roc_auc_score  # local import; optional dep

    c, a = _as_arrays(confidences, accuracies)
    if len(np.unique(a)) < 2:
        return float("nan")
    return float(roc_auc_score(a, c))


def logits_confidence_from_token_logprobs(token_logprobs: Iterable[float]) -> float:
    """DCPO logits confidence: exp(mean over all response-token logprobs).

    Matches DCPO `get_answer_and_logit_confidence`:
        avg_logprob = sum(token_logprobs) / len(token_logprobs)
        confidence = exp(avg_logprob)
    """
    lps = [float(x) for x in token_logprobs if x is not None]
    if not lps:
        return 0.0
    return math.exp(sum(lps) / len(lps))


def length_normed_seq_prob(cumulative_logprob: float, n_tokens: int) -> float:
    """Fast path when we have vLLM's `cumulative_logprob` and token count only.

    Equivalent to `logits_confidence_from_token_logprobs` when the cumulative
    logprob is the sum of the same token logprobs.
    """
    if n_tokens <= 0:
        return 0.0
    return math.exp(cumulative_logprob / n_tokens)


# --- Aliases matching the paper/plan nomenclature --------------------------

# PCE (Positive Calibration Error) in the DCPO paper = compute_overconf_ece.
pce = compute_overconf_ece
ece = compute_ece
brier = compute_brier
mce = compute_mce
auroc = compute_auroc
