"""Benchmark OrcaAI v4 knowledge checkpoint and sweep generation settings.

Designed as an unattended post-training job. It does not modify weights.
It compares grounded vs knowledge checkpoints on held-out facts, then runs a
larger generation sweep on the best knowledge checkpoint to identify settings
that reduce repetition without making outputs too short.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"
GROUNDED_PATH = CHECKPOINT_DIR / "model_v4_grounded.pt"
KNOWLEDGE_BEST_PATH = CHECKPOINT_DIR / "model_v4_knowledge_best.pt"
KNOWLEDGE_LATEST_PATH = CHECKPOINT_DIR / "model_v4_knowledge.pt"
REPORT_PATH = DATA_DIR / "reports" / "v4_knowledge_benchmark.json"

UNSEEN_TESTS = [
    ("Madrid is the capital of", " Spain", [" Portugal", " Italy", " France"]),
    ("Lisbon is the capital of", " Portugal", [" Spain", " France", " Italy"]),
    ("Canberra is the capital of", " Australia", [" New Zealand", " Canada", " India"]),
    ("Ottawa is the capital of", " Canada", [" Australia", " Ireland", " Mexico"]),
    ("Athens is the capital of", " Greece", [" Turkey", " Egypt", " Italy"]),
    ("Cairo is the capital of", " Egypt", [" Greece", " Kenya", " Morocco"]),
    ("Three plus four equals", " seven", [" six", " eight", " nine"]),
    ("Six plus three equals", " nine", [" eight", " ten", " seven"]),
    ("Ten minus four equals", " six", [" five", " seven", " eight"]),
    ("An octagon has", " eight", [" six", " seven", " nine"]),
    ("A square has", " four", [" three", " five", " six"]),
    ("One hour has", " sixty", [" thirty", " forty", " ninety"]),
    ("One minute has", " sixty", [" thirty", " fifty", " one hundred"]),
    ("The opposite of fast is", " slow", [" quick", " high", " bright"]),
    ("The opposite of dark is", " light", [" black", " night", " shadow"]),
    ("The opposite of old is", " young", [" ancient", " past", " slow"]),
    ("A horse is an", " animal", [" planet", " language", " machine"]),
    ("A banana is a", " fruit", [" vegetable", " metal", " country"]),
    ("A potato is a", " vegetable", [" fruit", " planet", " language"]),
    ("Milk is a", " drink", [" processor", " country", " planet"]),
    ("A mouse is an", " input device", [" output device", " operating system", " database"]),
    ("A printer is an", " output device", [" input device", " programming language", " CPU"]),
    ("A GPU processes", " graphics", [" paper", " water", " food"]),
    ("A hard drive stores", " data", [" sunlight", " oxygen", " sound only"]),
    ("Java is a programming", " language", [" battery", " monitor", " processor"]),
    ("CSS is used to style", " web pages", [" batteries", " CPUs", " hard drives"]),
    ("An operating system manages", " computer resources", [" planets", " blood", " weather"]),
    ("A file contains", " data", [" sunlight", " gravity", " fuel"]),
    ("The chemical symbol for iron is", " Fe", [" Ir", " In", " I"]),
    ("The chemical symbol for gold is", " Au", [" Ag", " Gd", " Go"]),
    ("Carbon dioxide has the formula", " CO2", [" H2O", " O2", " NaCl"]),
    ("Ice is", " solid water", [" liquid iron", " hot air", " dry sand"]),
    ("Steam is", " water vapor", [" frozen water", " solid carbon", " liquid metal"]),
    ("The Earth has one natural", " moon", [" sun", " ring", " ocean"]),
    ("Mars is a", " planet", [" star", " moon", " galaxy"]),
    ("Saturn is known for its", " rings", [" oceans", " forests", " mountains"]),
    ("The Sun is a", " star", [" planet", " moon", " comet"]),
    ("Bees make", " honey", [" plastic", " iron", " paper"]),
    ("Cows produce", " milk", [" gasoline", " electricity", " plastic"]),
    ("Trees have", " roots", [" wheels", " engines", " processors"]),
    ("A spider has", " eight legs", [" six legs", " four legs", " ten legs"]),
    ("A human has two", " lungs", [" hearts", " brains", " stomachs"]),
    ("The liver is an", " organ", [" planet", " language", " machine"]),
    ("Rain falls from", " clouds", [" roads", " engines", " computers"]),
    ("Fire is usually", " hot", [" frozen", " wet", " silent"]),
    ("The sky on a clear day is usually", " blue", [" green", " orange", " black"]),
    ("Chocolate is made from", " cocoa", [" iron", " plastic", " gasoline"]),
    ("Bread is commonly made from", " flour", [" steel", " glass", " silicon"]),
]

GENERATION_PROMPTS = [
    "RAM is used to",
    "The CPU executes",
    "A function in programming is",
    "Paris is the capital of",
    "Water freezes at",
    "Madrid is the capital of",
    "Three plus four equals",
    "A mouse is an",
    "A GPU processes",
    "An SSD is used for",
    "Python is a programming",
    "The chemical symbol for iron is",
    "Ice is",
    "The Sun is a",
    "Bees make",
    "A spider has",
    "The heart pumps",
    "A database stores",
    "The opposite of hot is",
    "The sky on a clear day is usually",
]

SETTINGS = [
    (0.25, 20, 0.85, 1.08),
    (0.30, 30, 0.88, 1.10),
    (0.35, 30, 0.90, 1.10),
    (0.35, 40, 0.90, 1.12),
    (0.40, 40, 0.90, 1.12),
    (0.40, 50, 0.92, 1.12),
    (0.45, 40, 0.92, 1.12),
    (0.45, 50, 0.94, 1.14),
    (0.50, 50, 0.92, 1.12),
    (0.50, 60, 0.95, 1.15),
    (0.55, 60, 0.95, 1.15),
    (0.60, 80, 0.96, 1.18),
]

SEEDS = [1337, 2404, 7]


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = OrcaAIv4(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


@torch.inference_mode()
def continuation_score(model, tokenizer, prompt, continuation, device):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(continuation)
    all_ids = prompt_ids + answer_ids

    x = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits[0], dim=-1)

    start = len(prompt_ids) - 1
    values = [
        float(log_probs[start + i, token].item())
        for i, token in enumerate(answer_ids)
        if start + i < log_probs.size(0)
    ]
    return sum(values) / len(values) if values else float("-inf")


@torch.inference_mode()
def fact_accuracy(model, tokenizer, device):
    correct = 0
    ranks = []
    misses = []

    for prompt, expected, distractors in UNSEEN_TESTS:
        candidates = [expected] + distractors
        scores = {
            c: continuation_score(model, tokenizer, prompt, c, device)
            for c in candidates
        }
        ranking = sorted(candidates, key=scores.get, reverse=True)
        rank = ranking.index(expected) + 1
        ranks.append(rank)
        if rank == 1:
            correct += 1
        else:
            misses.append({
                "prompt": prompt,
                "expected": expected.strip(),
                "predicted": ranking[0].strip(),
                "rank": rank,
            })

    return {
        "correct": correct,
        "total": len(UNSEEN_TESTS),
        "accuracy": correct / len(UNSEEN_TESTS),
        "mean_rank": mean(ranks),
        "misses": misses,
    }


def repetition_metrics(ids):
    if not ids:
        return 0.0, 0.0, 0.0

    token_ratio = len(set(ids)) / len(ids)
    bigrams = list(zip(ids, ids[1:]))
    bigram_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 1.0

    counts = {}
    for token in ids:
        counts[token] = counts.get(token, 0) + 1
    max_fraction = max(counts.values()) / len(ids)

    return token_ratio, bigram_ratio, max_fraction


@torch.inference_mode()
def run_sweep(model, tokenizer, device):
    results = []
    total = len(SETTINGS) * len(GENERATION_PROMPTS) * len(SEEDS)
    done = 0
    started = time.time()

    for setting_index, (temperature, top_k, top_p, penalty) in enumerate(SETTINGS, 1):
        rows = []

        for prompt in GENERATION_PROMPTS:
            prompt_ids = tokenizer.encode(prompt)
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            for seed in SEEDS:
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)

                generated = model.generate(
                    x,
                    max_new_tokens=100,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=penalty,
                    stop_token_ids={tokenizer.eos_id},
                )

                new_ids = generated[0, len(prompt_ids):].tolist()
                text = tokenizer.decode(new_ids).strip()
                token_ratio, bigram_ratio, max_fraction = repetition_metrics(new_ids)

                rows.append({
                    "prompt": prompt,
                    "seed": seed,
                    "text": text,
                    "length": len(new_ids),
                    "unique_token_ratio": token_ratio,
                    "unique_bigram_ratio": bigram_ratio,
                    "max_token_fraction": max_fraction,
                })

                done += 1
                if done % 30 == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    rate = done / elapsed
                    eta = (total - done) / max(rate, 1e-9) / 60
                    print(
                        f"  sweep {done}/{total} generations | "
                        f"ETA {eta:.1f} min"
                    )

        summary = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": penalty,
            "mean_length": mean(r["length"] for r in rows),
            "unique_token_ratio": mean(r["unique_token_ratio"] for r in rows),
            "unique_bigram_ratio": mean(r["unique_bigram_ratio"] for r in rows),
            "max_token_fraction": mean(r["max_token_fraction"] for r in rows),
            "samples": rows,
        }

        # Heuristic only: reward diversity, penalize one-token domination and
        # extremely short answers. This does not claim factual correctness.
        length_factor = min(summary["mean_length"] / 35.0, 1.0)
        summary["quality_score"] = (
            summary["unique_bigram_ratio"] * 2.0
            + summary["unique_token_ratio"]
            + length_factor * 0.5
            - summary["max_token_fraction"] * 2.0
        )
        results.append(summary)

        print(
            f"Setting {setting_index}/{len(SETTINGS)} | "
            f"T={temperature:.2f} K={top_k} P={top_p:.2f} RP={penalty:.2f} | "
            f"len={summary['mean_length']:.1f} | "
            f"bigram={summary['unique_bigram_ratio']:.3f} | "
            f"score={summary['quality_score']:.3f}"
        )

    return sorted(results, key=lambda x: x["quality_score"], reverse=True)


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Missing tokenizer_v4.json")

    knowledge_path = (
        KNOWLEDGE_BEST_PATH
        if KNOWLEDGE_BEST_PATH.exists()
        else KNOWLEDGE_LATEST_PATH
    )
    if not knowledge_path.exists():
        raise FileNotFoundError(
            "Knowledge checkpoint not found. Finish the knowledge run first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    print("=" * 76)
    print("OrcaAI v4 knowledge benchmark + generation sweep")
    print("=" * 76)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Knowledge checkpoint: {knowledge_path.name}")
    print(f"Generation sweep: {len(SETTINGS) * len(GENERATION_PROMPTS) * len(SEEDS)} runs")
    print()
    print("This does NOT train or modify model weights.")

    report = {"checkpoints": {}, "generation_sweep": []}

    for name, path in (("GROUNDED", GROUNDED_PATH), ("KNOWLEDGE", knowledge_path)):
        if not path.exists():
            continue

        print()
        print(f"Evaluating {name}: {path} ...")
        checkpoint, model = load_model(path, device)
        facts = fact_accuracy(model, tokenizer, device)

        print(
            f"{name} unseen facts: "
            f"{facts['correct']}/{facts['total']} "
            f"({facts['accuracy'] * 100:.1f}%) | "
            f"mean rank {facts['mean_rank']:.2f}"
        )

        report["checkpoints"][name.lower()] = {
            "path": str(path),
            "step": checkpoint.get("step"),
            "source_step": checkpoint.get("source_step"),
            "facts": facts,
        }

        if name == "KNOWLEDGE":
            print()
            print("Running unattended generation sweep...")
            sweep = run_sweep(model, tokenizer, device)
            report["generation_sweep"] = sweep

            print()
            print("Top generation settings:")
            for row in sweep[:3]:
                print(
                    f"  T={row['temperature']:.2f} "
                    f"K={row['top_k']} "
                    f"P={row['top_p']:.2f} "
                    f"RP={row['repetition_penalty']:.2f} | "
                    f"score={row['quality_score']:.3f} | "
                    f"len={row['mean_length']:.1f}"
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("Benchmark complete")
    print("=" * 76)
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
