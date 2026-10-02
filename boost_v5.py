"""Continue OrcaAI v5 on multilingual Wikipedia + generated arithmetic.

Run this only after the normal 120k-step v5 pretraining finishes. It keeps the
same tokenizer/model architecture and starts with a fresh low-LR optimizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import random
import time

import numpy as np
import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")

MULTI_TRAIN = DATA_DIR / "multilingual_v5_boost.bin"
MULTI_VAL = DATA_DIR / "multilingual_v5_boost_val.bin"
MATH_TRAIN = DATA_DIR / "math_v5_boost.bin"
MATH_VAL = DATA_DIR / "math_v5_boost_val.bin"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"
BASE_BEST_PATH = CHECKPOINT_DIR / "model_v5_best.pt"
BASE_LATEST_PATH = CHECKPOINT_DIR / "model_v5.pt"
BOOST_PATH = CHECKPOINT_DIR / "model_v5_boost.pt"
BOOST_BEST_PATH = CHECKPOINT_DIR / "model_v5_boost_best.pt"

MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
DEFAULT_STEPS = 15_000

LEARNING_RATE = 6e-5
MIN_LEARNING_RATE = 8e-6
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.08

MULTILINGUAL_WEIGHT = 0.65
MATH_WEIGHT = 0.35

PRINT_EVERY = 25
EVAL_EVERY = 500
EVAL_BATCHES = 16
SAVE_EVERY = 500
SAMPLE_EVERY = 500

SEED = 2505

SAMPLE_PROMPTS = [
    "Deutschland ist",
    "Die Hauptstadt von Österreich ist",
    "7 × 8 =",
    "12 mal 9 ist",
    "Bonjour, je m'appelle",
    "Madrid es la capital de",
]


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: py -3.12 orca.py prepare-v5-boost"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, context_length, device):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError(f"Prepared dataset is too small: {len(data):,} tokens")

    starts = torch.randint(0, max_start + 1, (MICRO_BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(
            np.array(data[i:i + context_length], dtype=np.int64)
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            np.array(data[i + 1:i + context_length + 1], dtype=np.int64)
        )
        for i in starts
    ])
    return x.to(device), y.to(device)


def choose_source(multilingual, math_data):
    return (
        multilingual
        if random.random() < MULTILINGUAL_WEIGHT
        else math_data
    )


def learning_rate(local_step, total_steps):
    if local_step <= WARMUP_STEPS:
        return LEARNING_RATE * local_step / WARMUP_STEPS

    progress = (local_step - WARMUP_STEPS) / max(
        total_steps - WARMUP_STEPS, 1
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
def evaluate(
    model,
    multilingual_val,
    math_val,
    context_length,
    device,
    use_bf16,
):
    multi_loss = evaluate_source(
        model, multilingual_val, context_length, device, use_bf16
    )
    math_loss = evaluate_source(
        model, math_val, context_length, device, use_bf16
    )
    combined = (
        MULTILINGUAL_WEIGHT * multi_loss
        + MATH_WEIGHT * math_loss
    )
    return combined, multi_loss, math_loss


@torch.no_grad()
def sample_text(model, tokenizer, device, prompt):
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    generated = model.generate(
        tokens,
        max_new_tokens=70,
        temperature=0.35,
        top_k=30,
        top_p=0.90,
        repetition_penalty=1.10,
        stop_token_ids={tokenizer.eos_id},
    )

    continuation = tokenizer.decode(
        generated[0, len(ids):].tolist()
    ).replace("\n", " ")

    model.train()
    return continuation.strip()


def save_checkpoint(
    path,
    model,
    optimizer,
    local_step,
    total_steps,
    config,
    best_val,
    source_checkpoint,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": local_step,
            "boost_steps": total_steps,
            "config": config,
            "stage": "v5_multilingual_math_boost",
            "best_val": best_val,
            "source_checkpoint": str(source_checkpoint),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    multilingual_train = load_tokens(MULTI_TRAIN)
    multilingual_val = load_tokens(MULTI_VAL)
    math_train = load_tokens(MATH_TRAIN)
    math_val = load_tokens(MATH_VAL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    resume = BOOST_PATH.exists()
    if resume:
        source_path = BOOST_PATH
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

    if checkpoint.get("stage") == "v5_multilingual_math_boost":
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val", float("inf")))
        print(f"Resuming v5 boost from step {start_step:,}")
    else:
        base_step = checkpoint.get("step", "unknown")
        print(f"Starting v5 boost from {source_path.name} (base step {base_step})")

    context_length = int(config["context_length"])
    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = (
        MICRO_BATCH_SIZE * context_length * GRAD_ACCUM_STEPS
    )

    print("=" * 72)
    print("OrcaAI v5 multilingual + math continuation")
    print("=" * 72)
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(f"GPU:                  {torch.cuda.get_device_name(0)}")
    print(f"Parameters:           {params:,}")
    print(f"Context:              {context_length}")
    print(f"Steps:                {args.steps:,}")
    print(f"Positions/step:       {tokens_per_step:,}")
    print(
        f"Mix:                  {MULTILINGUAL_WEIGHT:.0%} multilingual / "
        f"{MATH_WEIGHT:.0%} math"
    )
    print()

    model.train()
    started = time.time()

    for step in range(start_step + 1, args.steps + 1):
        lr = learning_rate(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            source = choose_source(multilingual_train, math_train)
            x, y = get_batch(source, context_length, device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(x, y)
                scaled_loss = loss / GRAD_ACCUM_STEPS

            scaled_loss.backward()
            accumulated_loss += loss.item() / GRAD_ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % PRINT_EVERY == 0:
            elapsed = max(time.time() - started, 1e-6)
            completed = step - start_step
            steps_per_second = completed / elapsed
            tokens_per_second = steps_per_second * tokens_per_step
            remaining = args.steps - step
            eta_hours = remaining / max(steps_per_second, 1e-9) / 3600

            print(
                f"Boost {step:>5}/{args.steps} | "
                f"train {accumulated_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tokens_per_second:,.0f} tok/s | "
                f"ETA {eta_hours:.2f}h"
            )

        if step % EVAL_EVERY == 0:
            val, multi_loss, math_loss = evaluate(
                model,
                multilingual_val,
                math_val,
                context_length,
                device,
                use_bf16,
            )
            print(
                f"  validation: {val:.4f} | "
                f"multilingual {multi_loss:.4f} | "
                f"math {math_loss:.4f}"
            )

            if val < best_val:
                best_val = val
                save_checkpoint(
                    BOOST_BEST_PATH,
                    model,
                    optimizer,
                    step,
                    args.steps,
                    config,
                    best_val,
                    source_path,
                )
                print("  new best boost checkpoint saved")

        if step % SAMPLE_EVERY == 0:
            print("  boost samples:")
            for prompt in SAMPLE_PROMPTS:
                print(
                    f"    {prompt!r} -> "
                    f"{sample_text(model, tokenizer, device, prompt)}"
                )

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                BOOST_PATH,
                model,
                optimizer,
                step,
                args.steps,
                config,
                best_val,
                source_path,
            )
            print("  boost resume checkpoint saved")

    save_checkpoint(
        BOOST_PATH,
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
    print("v5 multilingual + math continuation complete")
    print("=" * 72)
    print(f"Latest: {BOOST_PATH}")
    print(f"Best:   {BOOST_BEST_PATH}")
    print("Next:   py -3.12 orca.py test-v5")


if __name__ == "__main__":
    main()
