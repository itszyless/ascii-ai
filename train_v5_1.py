"""OrcaAI v5.1 continued pretraining.

Salvage pass for the existing v5 weights: keep the same architecture/tokenizer,
start from the strongest general v5 checkpoint, and spend a few more hours on
a cleaner English-heavy multilingual mix. No synthetic arithmetic is used.

This deliberately reuses already-prepared v5 files, including the multilingual
Wikipedia file created by prepare_v5_boost.py, so there is no new large data
download before training.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")

FINEWEB_TRAIN = DATA_DIR / "fineweb_v5.bin"
FINEWEB_VAL = DATA_DIR / "fineweb_v5_val.bin"
WIKI_TRAIN = DATA_DIR / "wikipedia_v5.bin"
WIKI_VAL = DATA_DIR / "wikipedia_v5_val.bin"
MULTI_TRAIN = DATA_DIR / "multilingual_v5_boost.bin"
MULTI_VAL = DATA_DIR / "multilingual_v5_boost_val.bin"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"
BASE_BEST_PATH = CHECKPOINT_DIR / "model_v5_best.pt"
BASE_LATEST_PATH = CHECKPOINT_DIR / "model_v5.pt"

LATEST_PATH = CHECKPOINT_DIR / "model_v5_1.pt"
BEST_PATH = CHECKPOINT_DIR / "model_v5_1_best.pt"

MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
DEFAULT_STEPS = 15_000

LEARNING_RATE = 3.0e-5
MIN_LEARNING_RATE = 5.0e-6
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.08

# Keep English semantics dominant while teaching useful multilingual structure
# throughout the continuation instead of in a separate Wikipedia-only boost.
SOURCE_WEIGHTS = {
    "fineweb": 0.65,
    "wiki_en": 0.10,
    "de": 0.12,
    "fr": 0.04,
    "es": 0.04,
    "other": 0.05,
}

PRINT_EVERY = 25
EVAL_EVERY = 500
EVAL_BATCHES = 12
SAVE_EVERY = 500
SAMPLE_EVERY = 500

SEED = 5101

# prepare_v5_boost.py writes these languages into one file in this exact order.
MULTI_TRAIN_LENGTHS = {
    "de": 24_000_000,
    "fr": 12_000_000,
    "es": 12_000_000,
    "it": 12_000_000,
    "pt": 12_000_000,
    "nl": 10_000_000,
    "pl": 10_000_000,
    "cs": 8_000_000,
    "sv": 8_000_000,
}

MULTI_VAL_LENGTHS = {
    "de": 1_000_000,
    "fr": 500_000,
    "es": 500_000,
    "it": 500_000,
    "pt": 500_000,
    "nl": 416_666,
    "pl": 416_666,
    "cs": 333_333,
    "sv": 333_333,
}

SAMPLE_PROMPTS = [
    "RAM is a type of",
    "The CPU executes",
    "Paris is the capital of",
    "Ice melts when",
    "A database is",
    "Deutschland ist",
    "Die Hauptstadt von Österreich ist",
    "Wasser gefriert bei",
    "Paris est la capitale de",
    "Madrid es la capital de",
]


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def slice_languages(data, lengths):
    expected = sum(lengths.values())
    if len(data) < expected:
        raise ValueError(
            f"Multilingual token file is too small: {len(data):,} < "
            f"{expected:,}. Re-run: py -3.12 orca.py prepare-v5-boost"
        )

    result = {}
    start = 0
    for language, length in lengths.items():
        result[language] = data[start:start + length]
        start += length
    return result


def other_view(language_views):
    # Italian through Swedish are contiguous after de/fr/es in the prepared file.
    first = language_views["it"]
    last = language_views["sv"]
    # np.memmap slices preserve offsets; reconstruct a contiguous view from the
    # original backing data through offsets in the caller instead, so this helper
    # is not used directly.
    return first, last


def get_batch(data, context_length, device):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError(f"Dataset too small: {len(data):,} tokens")

    starts = torch.randint(
        0,
        max_start + 1,
        (MICRO_BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + context_length],
                dtype=np.int64,
            )
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[i + 1:i + context_length + 1],
                dtype=np.int64,
            )
        )
        for i in starts
    ])

    return x.to(device), y.to(device)


def choose_source(sources):
    roll = random.random()
    total = 0.0
    for name, weight in SOURCE_WEIGHTS.items():
        total += weight
        if roll <= total:
            return sources[name]
    return sources["fineweb"]


def learning_rate(step, total_steps):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(
        total_steps - WARMUP_STEPS,
        1,
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


@torch.no_grad()
def evaluate_source(model, data, context_length, device, use_bf16):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, context_length, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(model, val_sources, context_length, device, use_bf16):
    losses = {}
    for name, source in val_sources.items():
        losses[name] = evaluate_source(
            model,
            source,
            context_length,
            device,
            use_bf16,
        )

    combined = sum(
        SOURCE_WEIGHTS[name] * losses[name]
        for name in SOURCE_WEIGHTS
    )
    return combined, losses


@torch.no_grad()
def sample_text(model, tokenizer, device, prompt):
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        x,
        max_new_tokens=60,
        temperature=0.35,
        top_k=30,
        top_p=0.90,
        repetition_penalty=1.08,
        stop_token_ids={tokenizer.eos_id},
    )

    text = tokenizer.decode(
        generated[0, len(ids):].tolist()
    ).replace("\n", " ").strip()

    model.train()
    return text


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    total_steps,
    config,
    best_val,
    source_checkpoint,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "continuation_steps": total_steps,
            "config": config,
            "stage": "pretrain_v5_1",
            "best_val": best_val,
            "source_checkpoint": str(source_checkpoint),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Continuation optimizer steps (default: 15000)",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")

    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    fineweb_train = load_tokens(FINEWEB_TRAIN)
    fineweb_val = load_tokens(FINEWEB_VAL)
    wiki_train = load_tokens(WIKI_TRAIN)
    wiki_val = load_tokens(WIKI_VAL)
    multi_train = load_tokens(MULTI_TRAIN)
    multi_val = load_tokens(MULTI_VAL)

    train_languages = slice_languages(
        multi_train,
        MULTI_TRAIN_LENGTHS,
    )
    val_languages = slice_languages(
        multi_val,
        MULTI_VAL_LENGTHS,
    )

    other_train_start = (
        MULTI_TRAIN_LENGTHS["de"]
        + MULTI_TRAIN_LENGTHS["fr"]
        + MULTI_TRAIN_LENGTHS["es"]
    )
    other_val_start = (
        MULTI_VAL_LENGTHS["de"]
        + MULTI_VAL_LENGTHS["fr"]
        + MULTI_VAL_LENGTHS["es"]
    )

    sources = {
        "fineweb": fineweb_train,
        "wiki_en": wiki_train,
        "de": train_languages["de"],
        "fr": train_languages["fr"],
        "es": train_languages["es"],
        "other": multi_train[other_train_start:sum(MULTI_TRAIN_LENGTHS.values())],
    }
    val_sources = {
        "fineweb": fineweb_val,
        "wiki_en": wiki_val,
        "de": val_languages["de"],
        "fr": val_languages["fr"],
        "es": val_languages["es"],
        "other": multi_val[other_val_start:sum(MULTI_VAL_LENGTHS.values())],
    }

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    resume = LATEST_PATH.exists()
    if resume:
        source_path = LATEST_PATH
    elif BASE_BEST_PATH.exists():
        source_path = BASE_BEST_PATH
    elif BASE_LATEST_PATH.exists():
        source_path = BASE_LATEST_PATH
    else:
        raise FileNotFoundError(
            "No v5 base checkpoint found. Finish train-v5 first."
        )

    checkpoint = torch.load(
        source_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    context_length = int(config["context_length"])

    model = OrcaAIv5(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_gradient_checkpointing(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    start_step = 0
    best_val = float("inf")

    if checkpoint.get("stage") == "pretrain_v5_1":
        if checkpoint.get("continuation_steps") != args.steps:
            print(
                "Warning: resuming a v5.1 checkpoint created with a "
                "different --steps value."
            )
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val", float("inf")))
        print(f"Resuming OrcaAI v5.1 from step {start_step:,}")
    else:
        print(
            f"Starting OrcaAI v5.1 from {source_path.name} "
            f"(base step {checkpoint.get('step', 'unknown')})"
        )

    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = (
        MICRO_BATCH_SIZE
        * context_length
        * GRAD_ACCUM_STEPS
    )

    print("=" * 72)
    print("OrcaAI v5.1 semantic continuation")
    print("=" * 72)
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")
    print(f"Parameters:           {params:,}")
    print(f"Context:              {context_length}")
    print(f"Continuation steps:   {args.steps:,}")
    print(f"Positions/step:       {tokens_per_step:,}")
    print(f"Max LR:               {LEARNING_RATE:.2e}")
    print(
        "Data mix:             "
        + " / ".join(
            f"{SOURCE_WEIGHTS[name]:.0%} {name}"
            for name in SOURCE_WEIGHTS
        )
    )
    print("Synthetic math:       none")
    print()

    started = time.time()
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        lr = learning_rate(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            source = choose_source(sources)
            x, y = get_batch(
                source,
                context_length,
                device,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(x, y)
                scaled = loss / GRAD_ACCUM_STEPS

            scaled.backward()
            accumulated_loss += (
                loss.item() / GRAD_ACCUM_STEPS
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        if step == start_step + 1 or step % PRINT_EVERY == 0:
            elapsed = max(time.time() - started, 1e-6)
            completed = step - start_step
            rate = completed / elapsed
            tok_s = rate * tokens_per_step
            eta_hours = (
                (args.steps - step)
                / max(rate, 1e-9)
                / 3600
            )

            print(
                f"V5.1 {step:>5}/{args.steps} | "
                f"train {accumulated_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tok_s:,.0f} tok/s | "
                f"ETA {eta_hours:.2f}h"
            )

        if step % EVAL_EVERY == 0:
            val, losses = evaluate(
                model,
                val_sources,
                context_length,
                device,
                use_bf16,
            )

            details = " | ".join(
                f"{name} {losses[name]:.4f}"
                for name in (
                    "fineweb",
                    "wiki_en",
                    "de",
                    "fr",
                    "es",
                    "other",
                )
            )
            print(f"  validation: {val:.4f} | {details}")

            if val < best_val:
                best_val = val
                save_checkpoint(
                    BEST_PATH,
                    model,
                    optimizer,
                    step,
                    args.steps,
                    config,
                    best_val,
                    source_path,
                )
                print("  new best v5.1 checkpoint saved")

        if step % SAMPLE_EVERY == 0:
            print("  semantic samples:")
            for prompt in SAMPLE_PROMPTS:
                print(
                    f"    {prompt!r} -> "
                    f"{sample_text(model, tokenizer, device, prompt)}"
                )

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                LATEST_PATH,
                model,
                optimizer,
                step,
                args.steps,
                config,
                best_val,
                source_path,
            )
            print("  v5.1 resume checkpoint saved")

    save_checkpoint(
        LATEST_PATH,
        model,
        optimizer,
        args.steps,
        args.steps,
        config,
        best_val,
        source_path,
    )

    print()
    print("=" * 72)
    print("OrcaAI v5.1 continuation complete")
    print("=" * 72)
    print(f"Latest: {LATEST_PATH}")
    print(f"Best:   {BEST_PATH}")
    print()
    print("Next: py -3.12 orca.py test-v5-1")


if __name__ == "__main__":
    main()
