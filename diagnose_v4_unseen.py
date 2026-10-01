"""Held-out OrcaAI v4 benchmark for testing grounding generalization.

None of these facts are intentionally included in generate_grounding_v4.py.
The goal is to compare the 120k pretrained base with the grounded checkpoint
on unseen factual associations, not on facts used during grounding.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
DATA_DIR = Path("data")

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"
LATEST_PATH = CHECKPOINT_DIR / "model_v4.pt"
GROUNDED_PATH = CHECKPOINT_DIR / "model_v4_grounded.pt"
REPORT_PATH = DATA_DIR / "v4_unseen_diagnostic_report.json"

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
    "Madrid is the capital of",
    "Three plus four equals",
    "A mouse is an",
    "A GPU processes",
    "The chemical symbol for iron is",
    "Ice is",
    "The Sun is a",
    "Bees make",
    "A spider has",
    "The sky on a clear day is usually",
]


def load_model(path, device):
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

    all_ids = prompt_ids + answer_ids
    if len(all_ids) > model.context_length:
        all_ids = all_ids[-model.context_length:]
        prompt_ids = prompt_ids[
            -max(1, model.context_length - len(answer_ids)):
        ]

    x = torch.tensor(
        [all_ids[:-1]],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model(x)
    log_probs = F.log_softmax(logits[0], dim=-1)

    start = len(prompt_ids) - 1
    values = []

    for offset, token_id in enumerate(answer_ids):
        position = start + offset
        if position >= log_probs.size(0):
            break
        values.append(
            float(log_probs[position, token_id].item())
        )

    return (
        sum(values) / len(values)
        if values
        else float("-inf")
    )


@torch.inference_mode()
def run_facts(model, tokenizer, device):
    rows = []
    correct = 0

    for prompt, expected, distractors in UNSEEN_TESTS:
        candidates = [expected] + distractors
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
            candidates,
            key=lambda candidate: scores[candidate],
            reverse=True,
        )

        predicted = ranking[0]
        rank = ranking.index(expected) + 1
        passed = predicted == expected
        correct += int(passed)

        rows.append({
            "prompt": prompt,
            "expected": expected.strip(),
            "predicted": predicted.strip(),
            "correct": passed,
            "expected_rank": rank,
            "scores": {
                key.strip(): round(value, 4)
                for key, value in scores.items()
            },
        })

    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "mean_expected_rank": mean(
            row["expected_rank"] for row in rows
        ),
        "items": rows,
    }


def repetition_metrics(token_ids):
    if not token_ids:
        return 0.0, 0.0, 0.0

    token_ratio = len(set(token_ids)) / len(token_ids)

    bigrams = list(zip(token_ids, token_ids[1:]))
    bigram_ratio = (
        len(set(bigrams)) / len(bigrams)
        if bigrams
        else 1.0
    )

    counts = {}
    for token in token_ids:
        counts[token] = counts.get(token, 0) + 1

    max_fraction = max(counts.values()) / len(token_ids)
    return token_ratio, bigram_ratio, max_fraction


@torch.inference_mode()
def run_generations(model, tokenizer, device):
    rows = []
    seeds = [1337, 2404, 7]

    for prompt in GENERATION_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        for seed in seeds:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            generated = model.generate(
                x,
                max_new_tokens=90,
                temperature=0.40,
                top_k=40,
                top_p=0.90,
                repetition_penalty=1.12,
                stop_token_ids={tokenizer.eos_id},
            )

            new_ids = generated[
                0,
                len(prompt_ids):,
            ].tolist()

            token_ratio, bigram_ratio, max_fraction = (
                repetition_metrics(new_ids)
            )

            rows.append({
                "prompt": prompt,
                "seed": seed,
                "continuation": tokenizer.decode(
                    new_ids
                ).strip(),
                "length": len(new_ids),
                "unique_token_ratio": token_ratio,
                "unique_bigram_ratio": bigram_ratio,
                "max_token_fraction": max_fraction,
            })

    return {
        "samples": rows,
        "mean_length": mean(
            row["length"] for row in rows
        ),
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


def print_result(name, checkpoint, facts, generations):
    print()
    print("=" * 76)
    print(
        f"{name}: source step "
        f"{checkpoint.get('source_step', checkpoint.get('step', 'unknown'))}"
    )
    print("=" * 76)
    print(
        f"UNSEEN fact accuracy:   "
        f"{facts['correct']}/{facts['total']} "
        f"({facts['accuracy'] * 100:.1f}%)"
    )
    print(
        f"Mean expected rank:     "
        f"{facts['mean_expected_rank']:.2f}"
    )
    print(
        f"Mean generation length: "
        f"{generations['mean_length']:.1f} tokens"
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
        f"{generations['mean_max_token_fraction']:.3f}"
    )

    misses = [
        item
        for item in facts["items"]
        if not item["correct"]
    ][:12]

    if misses:
        print()
        print("Example unseen misses:")
        for item in misses:
            print(
                f"  {item['prompt']!r} -> "
                f"{item['predicted']!r} "
                f"(expected {item['expected']!r}, "
                f"rank {item['expected_rank']})"
            )


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoints/tokenizer_v4.json"
        )

    if not LATEST_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoints/model_v4.pt"
        )

    if not GROUNDED_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoints/model_v4_grounded.pt"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = LocalTokenizer.load(
        TOKENIZER_PATH
    )

    print("=" * 76)
    print("OrcaAI v4 HELD-OUT grounding benchmark")
    print("=" * 76)
    print(f"Device:     {device}")
    if device.type == "cuda":
        print(
            f"GPU:        "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(f"Unseen facts: {len(UNSEEN_TESTS)}")
    print()
    print(
        "These facts were not intentionally included in the "
        "grounding dataset."
    )
    print(
        "This benchmark does not train or modify model weights."
    )

    report = {
        "tests": len(UNSEEN_TESTS),
        "checkpoints": {},
    }

    for name, path in (
        ("LATEST", LATEST_PATH),
        ("GROUNDED", GROUNDED_PATH),
    ):
        print()
        print(f"Loading {name}: {path} ...")

        checkpoint, model = load_model(
            path,
            device,
        )
        facts = run_facts(
            model,
            tokenizer,
            device,
        )
        generations = run_generations(
            model,
            tokenizer,
            device,
        )

        print_result(
            name,
            checkpoint,
            facts,
            generations,
        )

        report["checkpoints"][name.lower()] = {
            "path": str(path),
            "checkpoint_step": checkpoint.get("step"),
            "source_step": checkpoint.get("source_step"),
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
    print("Held-out diagnostic complete")
    print("=" * 76)
    print(f"Report: {REPORT_PATH}")
    print()
    print(
        "Send me the LATEST and GROUNDED summaries "
        "before chat fine-tuning."
    )


if __name__ == "__main__":
    main()
