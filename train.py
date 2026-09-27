"""Pretrain LocalAI from random weights on prepared token data."""

from pathlib import Path
import math
import time

import numpy as np
import torch

from model import LocalAI
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"

# Practical starting point for an RTX 4060 Ti 8 GB.
CONTEXT_LENGTH = 256
EMBEDDING_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 24
TRAINING_STEPS = 10_000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
PRINT_EVERY = 50
EVAL_EVERY = 250
EVAL_BATCHES = 20
SAVE_EVERY = 500


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: py -3.12 prepare_data.py"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1
    if max_start < 0:
        raise ValueError("Prepared dataset is too small.")

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
    x = torch.stack(
        [
            torch.from_numpy(
                np.array(data[i : i + CONTEXT_LENGTH], dtype=np.int64)
            )
            for i in starts
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                np.array(data[i + 1 : i + CONTEXT_LENGTH + 1], dtype=np.int64)
            )
            for i in starts
        ]
    )
    return x.to(device), y.to(device)


def learning_rate(step):
    progress = min(step / TRAINING_STEPS, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, step, config):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "stage": "pretrain",
        },
        MODEL_PATH,
    )


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing tokenizer. Run: py -3.12 prepare_data.py"
        )

    train_data = load_tokens(TRAIN_PATH)
    val_data = load_tokens(VAL_PATH)
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    start_step = 0
    if MODEL_PATH.exists():
        old = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if old.get("config") == config and old.get("stage") == "pretrain":
            model.load_state_dict(old["model_state"])
            if "optimizer_state" in old:
                optimizer.load_state_dict(old["optimizer_state"])
            start_step = old.get("step", 0)
            print(f"Resuming checkpoint from step {start_step:,}")

    parameter_count = sum(p.numel() for p in model.parameters())

    print("=" * 62)
    print("LocalAI pretraining")
    print("=" * 62)
    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"Parameters:   {parameter_count:,}")
    print(f"Vocabulary:   {tokenizer.vocab_size:,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    print(f"Context:      {CONTEXT_LENGTH}")
    print()

    model.train()
    started = time.time()

    for step in range(start_step + 1, TRAINING_STEPS + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = (step - start_step) / elapsed
            print(
                f"Step {step:>6}/{TRAINING_STEPS} | "
                f"train {loss.item():.4f} | "
                f"lr {lr:.2e} | {rate:.2f} steps/s"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_data, device)
            print(f"  validation loss: {val_loss:.4f}")

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, config)
            print("  checkpoint saved")

    save_checkpoint(model, optimizer, TRAINING_STEPS, config)
    print()
    print(f"Pretraining complete: {MODEL_PATH}")
    print("Next: add conversations to data/chat.txt and run finetune.py")


if __name__ == "__main__":
    main()
