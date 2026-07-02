#!/usr/bin/env python3
"""
analyze_judge_scores.py

Reads final_judge_results.json and shows the full distribution of
final_score, with main percentiles and a textual histogram.

Use AFTER running run_judge_test_100.sh to choose the threshold
to set in Config.Refine.TARGET_ACCURACY.

Usage:
    python analyze_judge_scores.py <path/a/final_judge_results.json>
    python analyze_judge_scores.py /leonardo_work/.../test_100/final_judge_results.json
"""

import json
import sys
import os


def percentile(sorted_vals: list, p: float) -> float:
    '''Percentile calculation for a sorted list of values.'''
    n = len(sorted_vals)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def histogram(vals: list, bins: int = 20, width: int = 40) -> str:
    '''Normalized histogram of values as a string.'''
    lo, hi = min(vals), max(vals)
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        b = min(int((v - lo) / step), bins - 1)
        counts[b] += 1
    max_c = max(counts)
    lines = []
    for i, c in enumerate(counts):
        bar_lo = lo + i * step
        bar_hi = bar_lo + step
        bar = "█" * int(c / max_c * width)
        lines.append(f"  {bar_lo:.2f}-{bar_hi:.2f} │{bar:<{width}}│ {c:3d}")
    return "\n".join(lines)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None

    # Auto-detect if not specified
    if path is None:
        candidates = [
            "/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_100/final_judge_results.json",
            "final_judge_results.json",
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break

    if path is None or not os.path.exists(path):
        print("❌ File not found. Specify the path:")
        print("   python analyze_judge_scores.py <path/a/final_judge_results.json>")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f" Distribution of FINAL SCORE — {os.path.basename(path)}")
    print(f" N concepts: {len(data)}")
    print(f"{'='*60}\n")

    scores = sorted(d.get("final_score", 0.0) for d in data.values())
    n = len(scores)

    # ── Base statistics ──────────────────────────────────────
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5

    print(f"  Min:    {scores[0]:.3f}   ({scores[0]:.1%})")
    print(f"  Max:    {scores[-1]:.3f}   ({scores[-1]:.1%})")
    print(f"  Mean:  {mean:.3f}   ({mean:.1%})")
    print(f"  Std:    {std:.3f}")

    # ── Percentiles ───────────────────────────────────────────
    print(f"\n  {'Percentile':<12} {'Score':>8}   {'Pass rate if used as threshold':>10}")
    print(f"  {'-'*55}")
    for p in [10, 25, 50, 75, 90, 95]:
        v = percentile(scores, p)
        # If we use v as threshold, what percentage of concepts would pass?
        pass_rate = sum(1 for s in scores if s >= v) / n
        print(f"  P{p:02d}         {v:>8.3f}   {pass_rate:>6.1%} of concepts would pass")

    # ── Histogram ───────────────────────────────────────────
    print(f"\n  Distribution (histogram):\n")
    print(histogram(scores))

    # ── Breakdown per metric ─────────────────────────────────
    print(f"\n  Breakdown per metric (mean among all concepts):\n")
    metrics = ["clip_i", "clip_t", "dino_i", "tifa_score"]
    for m in metrics:
        vals = [d.get("metrics", {}).get(m, 0.0) for d in data.values()]
        vals = [v for v in vals if v > 0]
        if vals:
            print(f"  {m:<15} mean={sum(vals)/len(vals):.3f}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}  "
                  f"(out of {len(vals)}/{n} concepts)")

    # ── Suggested threshold ────────────────────────────────
    print(f"\n{'='*60}")
    print(f" SUGGESTED THRESHOLD")
    print(f"{'='*60}")

    p50 = percentile(scores, 50)
    p75 = percentile(scores, 75)
    p90 = percentile(scores, 90)

    print(f"""
  Choose based on the project's objective:

  ┌─────────────────────────────────────────────────────┐
  │ Objective             │ Threshold │ Expected Pass Rate │
  ├─────────────────────────────────────────────────────┤
  │ Permissive (baseline) │   {p50:.2f}    │     ~50%         │
  │ Balanced              │   {p75:.2f}    │     ~25%         │
  │ Aggressive (high QA)  │   {p90:.2f}    │     ~10%         │
  └─────────────────────────────────────────────────────┘

  In config.py:
      TARGET_ACCURACY = {p75:.2f}  # balanced — modify as needed
""")

    # ── List of failed concepts ────────────────────────────────
    chosen = p75
    print(f"  Concepts that would NOT pass with threshold={chosen:.2f}:\n")
    for cid, d in sorted(data.items(), key=lambda x: x[1].get("final_score", 0)):
        s = d.get("final_score", 0)
        if s < chosen:
            m = d.get("metrics", {})
            print(f"  ❌ {cid:<12} score={s:.3f} | "
                  f"CLIP-I={m.get('clip_i',0):.3f} | "
                  f"DINO={m.get('dino_i',0):.3f} | "
                  f"TIFA={m.get('tifa_score',0):.1%}")


if __name__ == "__main__":
    main()