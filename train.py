"""Train LocalAI from random weights using data/training.txt."""

from pathlib import Path
import time

import torch

from model import LocalAI
from tokenizer import ByteTokenizer


DATA_PATH = Path("data/training.txt")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"

# Sized for an 8 GB RTX 4060 Ti.
CONTEXT_LENGTH = 256
EMBEDDING_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 24
TRAINING_STEPS = 5000
LEARNING_RATE = 3e-4
PRINT_EVERY = 50
SAVE_EVERY = 500


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1

    if max_start < 0:
        raise ValueError(
            f"Training data is too short. Add at least "
            f"{CONTEXT_LENGTH + 2} byte tokens to data/training.txt."
        )

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))

    x = torch.stack([data[i : i + CONTEXT_LENGTH] for i in starts])
    y = torch.stack([data[i + 1 : i + CONTEXT_LENGTH + 1] for i in starts])

    return x.to(device), y.to(device)


def save_checkpoint(model, optimizer, step, tokenizer, config):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        MODEL_PATH,
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/training.txt. Create it and put training text inside."
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("data/training.txt is empty.")

    tokenizer = ByteTokenizer()
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 58)
    print("LocalAI training")
    print("=" * 58)
    print(f"Device:          {device}")

    if device.type == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")

    print(f"Training bytes:  {len(encoded):,}")
    print(f"Context:         {CONTEXT_LENGTH}")
    print()

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
        weight_decay=0.01,
    )

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters:      {parameter_count:,}")
    print(f"Training steps:  {TRAINING_STEPS:,}")
    print()

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model.train()
    started = time.time()

    for step in range(1, TRAINING_STEPS + 1):
        x, y = get_batch(encoded, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            elapsed = time.time() - started
            steps_per_second = step / elapsed
            print(
                f"Step {step:>5}/{TRAINING_STEPS} | "
                f"Loss {loss.item():.4f} | "
                f"{steps_per_second:.2f} steps/s"
            )

        if step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, tokenizer, config)
            print(f"  checkpoint saved at step {step}")

    save_checkpoint(
        model,
        optimizer,
        TRAINING_STEPS,
        tokenizer,
        config,
    )

    print()
    print("Training complete.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
