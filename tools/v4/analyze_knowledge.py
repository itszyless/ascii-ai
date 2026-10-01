"""Analyze the latest OrcaAI v4 knowledge benchmark report.

Reads data/reports/v4_knowledge_benchmark.json and prints a concise decision:
- whether knowledge continuation improved held-out factual accuracy
- whether it hurt ranking quality
- which generation setting ranked best in the sweep
- whether the knowledge checkpoint is worth carrying forward

This script never trains or modifies model weights.
"""

from __future__ import annotations

import json
from pathlib import Path


REPORT_PATH = Path("data/reports/v4_knowledge_benchmark.json")


def pct(value):
    return f"{value * 100:.1f}%"


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {REPORT_PATH}. Run the knowledge benchmark first."
        )

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    checkpoints = report.get("checkpoints", {})

    grounded = checkpoints.get("grounded", {})
    knowledge = checkpoints.get("knowledge", {})

    if not grounded or not knowledge:
        raise ValueError(
            "Benchmark report is missing grounded or knowledge results."
        )

    g = grounded["facts"]
    k = knowledge["facts"]

    g_acc = float(g["accuracy"])
    k_acc = float(k["accuracy"])
    g_rank = float(g["mean_rank"])
    k_rank = float(k["mean_rank"])

    acc_delta = k_acc - g_acc
    rank_delta = k_rank - g_rank

    sweep = report.get("generation_sweep", [])
    best = sweep[0] if sweep else None

    print("=" * 76)
    print("OrcaAI v4 knowledge benchmark analysis")
    print("=" * 76)
    print()
    print(
        f"GROUNDED:  {g['correct']}/{g['total']} "
        f"({pct(g_acc)}) | mean rank {g_rank:.2f}"
    )
    print(
        f"KNOWLEDGE: {k['correct']}/{k['total']} "
        f"({pct(k_acc)}) | mean rank {k_rank:.2f}"
    )
    print()
    print(
        f"Accuracy change: {acc_delta * 100:+.1f} percentage points"
    )
    print(
        f"Mean-rank change: {rank_delta:+.2f} "
        f"(negative is better)"
    )

    if acc_delta >= 0.08 and rank_delta <= 0.05:
        verdict = "STRONG_IMPROVEMENT"
        message = (
            "Knowledge continuation materially improved held-out facts. "
            "Carry the knowledge checkpoint forward."
        )
    elif acc_delta > 0.02 and rank_delta <= 0.10:
        verdict = "IMPROVEMENT"
        message = (
            "Knowledge continuation improved held-out facts enough to keep. "
            "Carry it forward, but keep the grounded checkpoint as fallback."
        )
    elif acc_delta >= -0.02 and rank_delta <= 0.05:
        verdict = "MIXED"
        message = (
            "Knowledge continuation is roughly neutral on held-out facts. "
            "Do not assume it is better than grounded."
        )
    else:
        verdict = "REGRESSION"
        message = (
            "Knowledge continuation hurt held-out factual behavior. "
            "Keep the grounded checkpoint as the preferred base."
        )

    print()
    print(f"Decision: {verdict}")
    print(message)

    if best:
        print()
        print("Best generation sweep setting:")
        print(
            f"  temperature       = {best['temperature']:.2f}\n"
            f"  top_k             = {best['top_k']}\n"
            f"  top_p             = {best['top_p']:.2f}\n"
            f"  repetition_penalty= {best['repetition_penalty']:.2f}\n"
            f"  mean length       = {best['mean_length']:.1f}\n"
            f"  unique bigrams    = {best['unique_bigram_ratio']:.3f}\n"
            f"  sweep score       = {best['quality_score']:.3f}"
        )

    print()
    print("Knowledge misses:")
    for miss in k.get("misses", [])[:12]:
        print(
            f"  {miss['prompt']!r} -> {miss['predicted']!r} "
            f"(expected {miss['expected']!r}, rank {miss['rank']})"
        )

    print()
    print("=" * 76)
    print(
        "Send me this analysis output. "
        "I will use it to choose the next training stage."
    )


if __name__ == "__main__":
    main()
