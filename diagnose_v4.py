"""Diagnose OrcaAI v4 before chat fine-tuning.

This benchmark compares the best and latest v4 checkpoints using:
- multiple-choice continuation scoring (facts + basic concepts)
- raw generation prompts
- repetition measurements
- checkpoint metadata

It does not train or modify model weights.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"
BEST_PATH = CHECKPOINT_DIR / "model_v4_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "model_v4.pt"\nGROUNDED_PATH = CHECKPOINT_DIR / "model_v4_grounded.pt"
REPORT_PATH = DATA_DIR / "v4_diagnostic_report.json"

FACT_TESTS = [
    ("Paris is the capital of", " France", [" Germany", " Italy", " Spain"]),
    ("Berlin is the capital of", " Germany", [" France", " Austria", " Italy"]),
    ("Rome is the capital of", " Italy", [" Spain", " France", " Germany"]),
    ("Vienna is the capital of", " Austria", [" Germany", " Switzerland", " Hungary"]),
    ("Tokyo is the capital of", " Japan", [" China", " Korea", " Thailand"]),
    ("The Earth orbits the", " Sun", [" Moon", " Mars", " Jupiter"]),
    ("The Moon orbits the", " Earth", [" Sun", " Mars", " Venus"]),
    ("Water freezes at", " 0", [" 50", " 100", " 500"]),
    ("Water boils at", " 100", [" 0", " 20", " 50"]),
    ("Two plus two equals", " four", [" five", " three", " six"]),
    ("Five plus five equals", " ten", [" nine", " six", " twelve"]),
    ("A triangle has", " three", [" four", " five", " six"]),
    ("A week has", " seven", [" five", " six", " eight"]),
    ("A year has twelve", " months", [" weeks", " days", " hours"]),
    ("The opposite of hot is", " cold", [" warm", " heat", " fire"]),
    ("The opposite of up is", " down", [" left", " high", " above"]),
    ("A dog is an", " animal", [" planet", " machine", " language"]),
    ("A cat is an", " animal", [" country", " computer", " element"]),
    ("An apple is a", " fruit", [" vehicle", " planet", " language"]),
    ("A carrot is a", " vegetable", [" country", " machine", " metal"]),
    ("RAM is used to", " store data", [" cool the computer", " print paper", " connect electricity"]),
    ("The CPU executes", " instructions", [" photographs", " batteries", " speakers"]),
    ("A computer keyboard is an", " input device", [" operating system", " output file", " power supply"]),
    ("A monitor is an", " output device", [" input device", " programming language", " database"]),
    ("Python is a programming", " language", [" processor", " battery", " monitor"]),
    ("HTML is used to create", " web pages", [" batteries", " processors", " hard drives"]),
    ("A function in programming is", " reusable code", [" a physical cable", " a computer screen", " a battery"]),
    ("Binary uses the digits", " 0 and 1", [" 1 and 2", " 2 and 3", " 8 and 9"]),
    ("The brain is part of the", " nervous system", [" solar system", " road system", " file system"]),
    ("The heart pumps", " blood", [" electricity", " sunlight", " plastic"]),
    ("Plants use sunlight for", " photosynthesis", [" respiration only", " combustion", " encryption"]),
    ("Humans breathe", " oxygen", [" iron", " plastic", " silicon"]),
    ("The chemical symbol for water is", " H2O", [" CO2", " O2", " NaCl"]),
    ("The planet closest to the Sun is", " Mercury", [" Earth", " Saturn", " Neptune"]),
    ("The largest planet in the Solar System is", " Jupiter", [" Mars", " Venus", " Mercury"]),
    ("The ocean is made mostly of", " water", [" iron", " wood", " glass"]),
    ("English is a", " language", [" planet", " processor", " chemical"]),
    ("A book is made of", " pages", [" planets", " processors", " engines"]),
    ("A car usually has", " wheels", [" wings", " roots", " leaves"]),
    ("Birds have", " feathers", [" scales only", " wheels", " processors"]),
    ("Fish live in", " water", [" space", " fire", " concrete"]),
    ("The color of grass is usually", " green", [" purple", " black", " orange"]),
    ("Snow is usually", " white", [" green", " red", " purple"]),
    ("The Sun provides", " light", [" RAM", " software", " passwords"]),
    ("A database stores", " data", [" sunlight", " gasoline", " gravity"]),
    ("An SSD is used for", " storage", [" cooling", " printing", " networking only"]),
    ("A browser is used to view", " websites", [" batteries", " processors", " keyboards"]),
    ("The internet connects", " computers", [" planets", " organs", " chemicals"]),
]

GENERATION_PROMPTS = [
    "RAM is used to",
    "The CPU executes",
    "A function in programming is",
    "Paris is the capital of",
    "Water freezes at",
    "A dog is an",
    "Python is a programming",
    "The Earth orbits the",
    "The heart pumps",
    "An SSD is used for",
    "A database stores",
    "The opposite of hot is",
]


def load_checkpoint(path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model = OrcaAIv4(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


@torch.inference_mode()
def continuation_score(model, tokenizer, prompt, continuation, device):
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(continuation)

    if not prompt_ids or not answer_ids:
        return float("-inf")

    # Score answer tokens conditioned on the prompt. We concatenate token IDs
    # directly so every candidate is compared with the same prompt prefix.
    all_ids = prompt_ids + answer_ids
    if len(all_ids) > model.context_length:
        all_ids = all_ids[-model.context_length:]
        prompt_ids = prompt_ids[-max(1, model.context_length - len(answer_ids)):]

    tokens = torch.tensor(
        [all_ids[:-1]],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model(tokens)
    log_probs = F.log_softmax(logits[0], dim=-1)

    start = len(prompt_ids) - 1
    values = []
    for index, token_id in enumerate(answer_ids):
        position = start + index
        if position >= log_probs.size(0):
            break
        values.append(float(log_probs[position, token_id].item()))

    if not values:
        return float("-inf")

    # Length-normalized score avoids automatically preferring short answers.
    return sum(values) / len(values)


@torch.inference_mode()
def run_fact_tests(model, tokenizer, device):
    passed = 0
    rows = []

    for prompt, correct, distractors in FACT_TESTS:
        candidates = [correct] + distractors
        scores = {
            candidate: continuation_score(
                model,
                tokenizer,
                prompt,
                candidate,
                device,
            )
            for candidate in candidates
        }
        ranking = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )
        predicted = ranking[0]
        correct_rank = ranking.index(correct) + 1
        is_correct = predicted == correct
        passed += int(is_correct)

        rows.append({
            "prompt": prompt,
            "expected": correct.strip(),
            "predicted": predicted.strip(),
            "correct": is_correct,
            "expected_rank": correct_rank,
            "scores": {
                key.strip(): round(value, 4)
                for key, value in scores.items()
            },
        })

    return {
        "correct": passed,
        "total": len(rows),
        "accuracy": passed / len(rows),
        "mean_expected_rank": mean(
            row["expected_rank"] for row in rows
        ),
        "items": rows,
    }


def repetition_metrics(token_ids):
    if not token_ids:
        return {
            "unique_token_ratio": 0.0,
            "unique_bigram_ratio": 0.0,
            "max_token_fraction": 0.0,
        }

    unique_token_ratio = len(set(token_ids)) / len(token_ids)

    bigrams = list(zip(token_ids, token_ids[1:]))
    unique_bigram_ratio = (
        len(set(bigrams)) / len(bigrams)
        if bigrams
        else 1.0
    )

    counts = {}
    for token in token_ids:
        counts[token] = counts.get(token, 0) + 1

    max_token_fraction = max(counts.values()) / len(token_ids)

    return {
        "unique_token_ratio": unique_token_ratio,
        "unique_bigram_ratio": unique_bigram_ratio,
        "max_token_fraction": max_token_fraction,
    }


@torch.inference_mode()
def run_generations(
    model,
    tokenizer,
    device,
    seeds,
):
    rows = []

    for prompt in GENERATION_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        prompt_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        for seed in seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            generated = model.generate(
                prompt_tensor,
                max_new_tokens=90,
                temperature=0.40,
                top_k=40,
                top_p=0.90,
                repetition_penalty=1.12,
                stop_token_ids={tokenizer.eos_id},
            )
            new_ids = generated[0, len(prompt_ids):].tolist()
            text = tokenizer.decode(new_ids).strip()
            metrics = repetition_metrics(new_ids)

            rows.append({
                "prompt": prompt,
                "seed": seed,
                "continuation": text,
                **{
                    key: round(value, 4)
                    for key, value in metrics.items()
                },
            })

    return {
        "samples": rows,
        "mean_unique_token_ratio": mean(
            row["unique_token_ratio"] for row in rows
        ),
        "mean_unique_bigram_ratio": mean(
            row["unique_bigram_ratio"] for row in rows
        ),
        "mean_max_token_fraction": mean(
            row["max_token_fraction"] for row in rows
        ),
    }


def print_summary(name, checkpoint, facts, generations):
    print()
    print("=" * 76)
    print(f"{name}: step {checkpoint.get('step', 'unknown')}")
    print("=" * 76)
    print(
        f"Fact-choice accuracy:   "
        f"{facts['correct']}/{facts['total']} "
        f"({facts['accuracy'] * 100:.1f}%)"
    )
    print(
        f"Mean expected rank:     "
        f"{facts['mean_expected_rank']:.2f} "
        f"(1.00 is best)"
    )
    print(
        f"Unique-token ratio:     "
        f"{generations['mean_unique_token_ratio']:.3f}"
    )
    print(
        f"Unique-bigram ratio:    "
        f"{generations['mean_unique_bigram_ratio']:.3f}"
    )
    print(
        f"Max-token fraction:     "
        f"{generations['mean_max_token_fraction']:.3f} "
        f"(lower is better)"
    )

    failures = [
        row for row in facts["items"]
        if not row["correct"]
    ][:10]

    if failures:
        print()
        print("Example factual misses:")
        for row in failures:
            print(
                f"  {row['prompt']!r} -> "
                f"{row['predicted']!r} "
                f"(expected {row['expected']!r}, "
                f"rank {row['expected_rank']})"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extended",
        action="store_true",
        help=(
            "Use 5 generation seeds instead of 2. "
            "This takes longer but gives a more stable repetition estimate."
        ),
    )
    parser.add_argument(
        "--grounded",
        action="store_true",
        help="Also evaluate checkpoints/model_v4_grounded.pt.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Evaluate only model_v4.pt.",
    )
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoints/tokenizer_v4.json"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    seeds = [1337, 2404, 7, 42, 99] if args.extended else [1337, 2404]

    targets = []
    if not args.latest_only and BEST_PATH.exists():
        targets.append(("BEST", BEST_PATH))
    if LATEST_PATH.exists():
        targets.append(("LATEST", LATEST_PATH))
    if args.grounded:
        if not GROUNDED_PATH.exists():
            raise FileNotFoundError(
                "Missing checkpoints/model_v4_grounded.pt. "
                "Run: py -3.12 ground_v4.py"
            )
        targets.append(("GROUNDED", GROUNDED_PATH))

    if not targets:
        raise FileNotFoundError(
            "No OrcaAI v4 checkpoint found."
        )

    print("=" * 76)
    print("OrcaAI v4 diagnostic benchmark")
    print("=" * 76)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Fact tests: {len(FACT_TESTS)}")
    print(
        f"Generation runs/checkpoint: "
        f"{len(GENERATION_PROMPTS) * len(seeds)}"
    )
    print()
    print("This script DOES NOT train or modify weights.")

    report = {
        "device": str(device),
        "extended": args.extended,
        "checkpoints": {},
    }

    for name, path in targets:
        print()
        print(f"Loading {name}: {path} ...")
        checkpoint, model = load_checkpoint(
            path,
            device,
        )

        facts = run_fact_tests(
            model,
            tokenizer,
            device,
        )
        generations = run_generations(
            model,
            tokenizer,
            device,
            seeds,
        )

        print_summary(
            name,
            checkpoint,
            facts,
            generations,
        )

        report["checkpoints"][name.lower()] = {
            "path": str(path),
            "step": checkpoint.get("step"),
            "best_val": checkpoint.get("best_val"),
            "facts": facts,
            "generations": generations,
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    DATA_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("Diagnostic complete")
    print("=" * 76)
    print(f"Full report saved to: {REPORT_PATH}")
    print()
    print(
        "Send me the terminal summary (or the report file) "
        "before running finetune_v4.py."
    )


if __name__ == "__main__":
    main()
