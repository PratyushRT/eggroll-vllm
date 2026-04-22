"""
DCPO prompt directive and confidence parsing — kept verbatim from
`calibration_main.py::get_confidence_and_answer` in the DCPO reference
implementation (https://github.com/icip-cas/DCPO).

Typos ("singal") and full-width colons ("：") are intentional: they are
the literal strings the DCPO paper trained and evaluated against. Changing
them would introduce a prompt confound. An ablation with a cleaned prompt
can be run separately.
"""
from __future__ import annotations

import re

# Verbatim from DCPO calibration_main.py:120-125 (the `directive` string).
DCPO_CONFIDENCE_DIRECTIVE = (
    "\n\nPlease put your final answer within \\boxed{}.\n"
    "\nAlso output a singal line at the end of the answer："
    "CONFIDENCE: <float number between 0 and 1>\n"
    "e.g. ：CONFIDENCE: 0.83\n"
    "Please make sure CONFIDENCE part is in a singal line with the exact same format。\n"
)

# Regex verbatim from DCPO (confidence.py, hybrid.py, calibration_main.py).
CONF_RE = re.compile(r"CONFIDENCE:\s*([01]?\.\d+|\d+)")

# DCPO eval defaults (calibration_main.py). Training reward managers use 0.0
# on parse failure instead; see `parse_conf_for_training`.
EVAL_DEFAULT_NO_MATCH = -0.1     # conf is None → -0.1
EVAL_DEFAULT_PARSE_FAIL = -0.2   # regex matched but float() raised → -0.2 (dead path in practice)


def parse_conf_for_eval(text: str):
    """Verbatim DCPO eval-side parse. Returns (conf, status).

    Negative sentinels propagate through ECE/Brier masks (they are excluded
    from binned calibration metrics by the `(c >= 0) & (c <= 1)` guard), while
    AUROC still ranks malformed responses as low-confidence — matching DCPO.
    """
    match = CONF_RE.search(text)
    if match:
        try:
            conf = float(match.group(1))
            return conf, "ok"
        except ValueError:
            return EVAL_DEFAULT_PARSE_FAIL, "parse_fail"
    return EVAL_DEFAULT_NO_MATCH, "no_match"


def parse_conf_for_training(text: str) -> tuple[float, str]:
    """Training-side parse. Matches DCPO reward-manager behavior:
    unparseable or missing confidence → 0.0 (never negative). Clamps to [0,1].
    """
    match = CONF_RE.search(text)
    if match:
        try:
            conf = float(match.group(1))
        except ValueError:
            return 0.0, "parse_fail"
        return max(0.0, min(1.0, conf)), "ok"
    return 0.0, "no_match"


def strip_conf(text: str) -> str:
    """Remove `CONFIDENCE: <number>` line-ish content before answer grading.

    Matches DCPO's `answer = re.sub(r"CONFIDENCE:\\s*[0-9.]+", "", gen_text).strip()`
    in `get_confidence_and_answer`.
    """
    return re.sub(r"CONFIDENCE:\s*[0-9.]+", "", text).strip()


__all__ = [
    "DCPO_CONFIDENCE_DIRECTIVE",
    "CONF_RE",
    "EVAL_DEFAULT_NO_MATCH",
    "EVAL_DEFAULT_PARSE_FAIL",
    "parse_conf_for_eval",
    "parse_conf_for_training",
    "strip_conf",
]
