# Reward-hacking forensics — headline

Eval steps analyzed: [0, 10, 20, 30, 40, 50, 60, 70]
Benchmarks: ['aime24', 'amc24', 'math500']
Total rollouts: 3280

## T1 — Confidence on wrong answers (verbal, mean)

| benchmark | step | conf(correct) | conf(wrong) | decoupling gap |
|---|---|---|---|---|
| aime24 | 0 | 0.944 | 0.537 | +0.407 |
| aime24 | 10 | 0.977 | 0.452 | +0.525 |
| aime24 | 20 | 0.974 | 0.485 | +0.489 |
| aime24 | 30 | 0.972 | 0.574 | +0.397 |
| aime24 | 40 | 0.977 | 0.562 | +0.415 |
| aime24 | 50 | 0.976 | 0.468 | +0.508 |
| aime24 | 60 | 0.980 | 0.513 | +0.467 |
| aime24 | 70 | 0.971 | 0.526 | +0.444 |
| amc24 | 0 | 0.989 | 0.705 | +0.283 |
| amc24 | 10 | 0.991 | 0.706 | +0.285 |
| amc24 | 20 | 0.992 | 0.728 | +0.264 |
| amc24 | 30 | 0.991 | 0.665 | +0.326 |
| amc24 | 40 | 0.991 | 0.698 | +0.293 |
| amc24 | 50 | 0.992 | 0.668 | +0.324 |
| amc24 | 60 | 0.986 | 0.721 | +0.265 |
| amc24 | 70 | 0.987 | 0.707 | +0.281 |
| math500 | 0 | 0.994 | 0.752 | +0.242 |
| math500 | 10 | 0.996 | 0.805 | +0.190 |
| math500 | 20 | 0.989 | 0.797 | +0.192 |
| math500 | 30 | 0.996 | 0.788 | +0.208 |
| math500 | 40 | 0.996 | 0.767 | +0.228 |
| math500 | 50 | 0.996 | 0.855 | +0.141 |
| math500 | 60 | 0.993 | 0.812 | +0.181 |
| math500 | 70 | 0.995 | 0.851 | +0.143 |

**Interpretation:** a shrinking (or negative) decoupling gap between steps means the model is getting equally confident on wrong and right answers — the reward-hacking signature.

## T3 — Response length on correct vs wrong (mean words)

| benchmark | step | words(correct) | words(wrong) |
|---|---|---|---|
| aime24 | 0 | 608 | 1087 |
| aime24 | 10 | 682 | 1088 |
| aime24 | 20 | 653 | 1129 |
| aime24 | 30 | 652 | 1036 |
| aime24 | 40 | 603 | 1074 |
| aime24 | 50 | 660 | 1077 |
| aime24 | 60 | 666 | 1083 |
| aime24 | 70 | 533 | 1063 |
| amc24 | 0 | 495 | 804 |
| amc24 | 10 | 515 | 778 |
| amc24 | 20 | 501 | 771 |
| amc24 | 30 | 493 | 829 |
| amc24 | 40 | 498 | 825 |
| amc24 | 50 | 489 | 790 |
| amc24 | 60 | 536 | 780 |
| amc24 | 70 | 449 | 759 |
| math500 | 0 | 265 | 545 |
| math500 | 10 | 255 | 544 |
| math500 | 20 | 249 | 571 |
| math500 | 30 | 260 | 597 |
| math500 | 40 | 247 | 569 |
| math500 | 50 | 265 | 520 |
| math500 | 60 | 270 | 562 |
| math500 | 70 | 263 | 566 |

**Interpretation:** if both correct and wrong response lengths collapse in lockstep, the population found a short-output exploit.

## T4 — Confidence by difficulty (step-0 easy vs hard)

| benchmark | step | easy conf | easy acc | hard conf | hard acc |
|---|---|---|---|---|---|
| aime24 | 0 | 0.877 | 0.786 | 0.544 | 0.033 |
| aime24 | 10 | 0.946 | 0.786 | 0.455 | 0.054 |
| aime24 | 20 | 0.973 | 0.714 | 0.464 | 0.043 |
| aime24 | 30 | 0.874 | 0.750 | 0.613 | 0.098 |
| aime24 | 40 | 0.945 | 0.821 | 0.580 | 0.076 |
| aime24 | 50 | 0.945 | 0.821 | 0.483 | 0.065 |
| aime24 | 60 | 0.946 | 0.857 | 0.538 | 0.076 |
| aime24 | 70 | 0.832 | 0.607 | 0.515 | 0.000 |
| amc24 | 0 | 0.912 | 0.759 | 0.717 | 0.000 |
| amc24 | 10 | 0.932 | 0.722 | 0.700 | 0.083 |
| amc24 | 20 | 0.916 | 0.722 | 0.747 | 0.056 |
| amc24 | 30 | 0.897 | 0.704 | 0.671 | 0.028 |
| amc24 | 40 | 0.935 | 0.778 | 0.700 | 0.056 |
| amc24 | 50 | 0.879 | 0.704 | 0.721 | 0.083 |
| amc24 | 60 | 0.930 | 0.778 | 0.723 | 0.028 |
| amc24 | 70 | 0.876 | 0.611 | 0.733 | 0.083 |
| math500 | 0 | 0.977 | 0.924 | 0.745 | 0.000 |
| math500 | 10 | 0.989 | 0.930 | 0.775 | 0.071 |
| math500 | 20 | 0.983 | 0.919 | 0.746 | 0.036 |
| math500 | 30 | 0.984 | 0.919 | 0.775 | 0.071 |
| math500 | 40 | 0.977 | 0.907 | 0.782 | 0.143 |
| math500 | 50 | 0.990 | 0.930 | 0.850 | 0.107 |
| math500 | 60 | 0.988 | 0.930 | 0.777 | 0.036 |
| math500 | 70 | 0.988 | 0.930 | 0.845 | 0.107 |

**Interpretation:** confidence on hard problems should go *down* with real calibration learning; if it stays high or rises while hard-accuracy doesn't improve, it's reward-hacking.


See `t2_diff_peak_vs_collapse.md` for side-by-side responses on problems the peak step got right and the final step got wrong.
