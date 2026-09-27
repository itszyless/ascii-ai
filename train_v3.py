"""Pretrain LocalAI v3 from random weights.

v3 deliberately trains on hundreds of millions of token positions instead of
trying to turn a barely-trained base model into a chatbot.
"""

from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np
import torch

from model import LocalAI
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train_v3.bin"
VAL_PATH = DATA_DIR / "val_v3.bin"
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v3.json"
MODEL_PATH = CHECKPOINT_DIR / "model_v3.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_v3_best.pt"

CONTEXT_LENGTH = 512
EMBEDDING_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

BATCH_SIZE = 8
TRAINING_STEPS = 80_000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 1000

PRINT_EVERY = 25
EVAL_EVERY = 500
EVAL_BATCHES = 25
SAVE_EVERY = 500
SAMPLE_EVERY = 500

SAMPLE_PROMPTS = [
    "RAM is a type of",
    "The CPU is responsible for",
    "A computer uses memory to",
    "A function in programming is",
]


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: py -3.12 prepare_v3.py"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1
    if max_start < 0:
        raise ValueError("Prepared dataset is too small.")

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(
            np.array(data[i:i + CONTEXT_LENGTH], dtype=np.int64)
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            np.array(data[i + 1:i + CONTEXT_LENGTH + 1], dtype=np.int64)
        )
        for i in starts
    ])
    return x.to(device), y.to(device)


def learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(
        TRAINING_STEPS - WARMUP_STEPS, 1
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


@torch.no_grad()
def evaluate(model, data, device, use_bf16):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, device)
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
def sample_text(model, tokenizer, device, prompt):
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    generated = model.generate(
        tokens,
        max_new_tokens=60,
        temperature=0.55,
        top_k=30,
        repetition_penalty=1.05,
        stop_token_ids={tokenizer.eos_id},
    )

    continuation = tokenizer.decode(
        generated[0, len(ids):].tolist()
    ).replace("\n", " ")

    model.train()
    return continuation.strip()


def save_checkpoint(path, model, optimizer, step, config, best_val):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "stage": "pretrain_v3",
            "best_val": best_val,
        },
        path,
    )


def main():
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    train_data = load_tokens(TRAIN_PATH)
    val_data = load_tokens(VAL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }

    model = LocalAI(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    start_step = 0
    best_val = float("inf")

    if MODEL_PATH.exists():
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )
        if (
            checkpoint.get("stage") == "pretrain_v3"
            and checkpoint.get("config") == config
        ):
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = checkpoint.get("step", 0)
            best_val = checkpoint.get("best_val", float("inf"))
            print(f"Resuming v3 from step {start_step:,}")

    params = sum(p.numel() for p in model.parameters())
    tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH
    total_positions = TRAINING_STEPS * tokens_per_step

    print("=" * 70)
    print("LocalAI v3 pretraining")
    print("=" * 70)
    print(f"Device:             {device}")
    if device.type == "cuda":
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"BF16:               {'yes' if use_bf16 else 'no'}")
    print(f"Parameters:         {params:,}")
    print(f"Vocabulary:         {tokenizer.vocab_size:,}")
    print(f"Context:            {CONTEXT_LENGTH}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Train tokens file:  {len(train_data):,}")
    print(f"Val tokens file:    {len(val_data):,}")
    print(f"Training steps:     {TRAINING_STEPS:,}")
    print(f"Token positions:    {total_positions:,}")
    print()

    started = time.time()
    model.train()

    for step in range(start_step + 1, TRAINING_STEPS + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % PRINT_EVERY == 0:
            elapsed = max(time.time() - started, 1e-6)
            completed = step - start_step
            steps_per_second = completed / elapsed
            tokens_per_second = steps_per_second * tokens_per_step
            remaining = TRAINING_STEPS - step
            eta_hours = remaining / max(steps_per_second, 1e-9) / 3600

            print(
                f"Step {step:>6}/{TRAINING_STEPS} | "
                f"train {loss.item():.4f} | "
                f"lr {lr:.2e} | "
                f"{tokens_per_second:,.0f} tok/s | "
                f"ETA {eta_hours:.2f}h"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_data, device, use_bf16)
            print(f"  validation loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    BEST_MODEL_PATH,
                    model,
                    optimizer,
                    step,
                    config,
                    best_val,
                )
                print("  new best v3 checkpoint saved")

        if step % SAMPLE_EVERY == 0:
            print("  language samples:")
            for prompt in SAMPLE_PROMPTS:
                continuation = sample_text(
                    model,
                    tokenizer,
                    device,
                    prompt,
                )
                print(f"    {prompt!r} -> {continuation}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                MODEL_PATH,
                model,
                optimizer,
                step,
                config,
                best_val,
            )
            print("  resume checkpoint saved")

    save_checkpoint(
        MODEL_PATH,
        model,
        optimizer,
        TRAINING_STEPS,
        config,
        best_val,
    )

    print()
    print("=" * 70)
    print("v3 pretraining complete")
    print("=" * 70)
    print(f"Resume checkpoint: {MODEL_PATH}")
    print(f"Best checkpoint:   {BEST_MODEL_PATH}")
    print()
    print("IMPORTANT: test the base before chat fine-tuning:")
    print("  py -3.12 test_v3.py")
    print("Do not run finetune_v3.py until the base produces coherent English.")


if __name__ == "__main__":
    main()
