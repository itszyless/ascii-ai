"""Fine-tune LocalAI on conversations without blindly memorizing chat.txt.

The file is split into train/validation data by complete conversations.  The
best validation checkpoint is kept, and training stops when validation no
longer improves.
"""

from pathlib import Path
import random
import time

import torch

from model import LocalAI
from tokenizer import LocalTokenizer


CHAT_PATH = Path("data/chat.txt")
MODEL_PATH = Path("checkpoints/model.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer.json")
OUTPUT_PATH = Path("checkpoints/chat_model.pt")

MAX_STEPS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
PRINT_EVERY = 25
EVAL_EVERY = 25
EVAL_BATCHES = 20
VALIDATION_FRACTION = 0.10
EARLY_STOP_PATIENCE = 4
SEED = 1337


def split_conversations(text):
    """Split on User: boundaries so validation examples are genuinely held out."""
    chunks = []
    current = []

    for line in text.splitlines():
        if line.startswith("User:") and current:
            chunk = "\n".join(current).strip()
            if "Assistant:" in chunk:
                chunks.append(chunk + "\n")
            current = []
        current.append(line)

    if current:
        chunk = "\n".join(current).strip()
        if "User:" in chunk and "Assistant:" in chunk:
            chunks.append(chunk + "\n")

    if len(chunks) < 20:
        raise ValueError("chat.txt needs at least 20 User:/Assistant: conversations.")

    rng = random.Random(SEED)
    rng.shuffle(chunks)
    val_count = max(1, int(len(chunks) * VALIDATION_FRACTION))
    return chunks[val_count:], chunks[:val_count]


def encode_conversations(conversations, tokenizer):
    ids = []
    for conversation in conversations:
        ids.extend(tokenizer.encode(conversation, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, context_length, device):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Dataset split needs more than {context_length + 1} tokens."
        )

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + context_length] for i in starts])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, context_length, device):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, context_length, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_best(model, config, step, train_loss, val_loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Pretrain first with train.py.")

    if not CHAT_PATH.exists():
        raise FileNotFoundError(
            "Create data/chat.txt with User:/Assistant: conversations first."
        )

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    text = CHAT_PATH.read_text(encoding="utf-8")
    train_conversations, val_conversations = split_conversations(text)
    train_data = encode_conversations(train_conversations, tokenizer)
    val_data = encode_conversations(val_conversations, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("=" * 62)
    print("LocalAI conversational fine-tuning")
    print("=" * 62)
    print(f"Device:              {device}")
    print(f"Base pretrain step:  {checkpoint.get('step', 'unknown')}")
    print(f"Train conversations: {len(train_conversations):,}")
    print(f"Val conversations:   {len(val_conversations):,}")
    print(f"Train tokens:        {len(train_data):,}")
    print(f"Val tokens:          {len(val_data):,}")
    print(f"Max steps:           {MAX_STEPS:,}")
    print(f"Learning rate:       {LEARNING_RATE:.1e}")
    print()

    best_val = float("inf")
    checks_without_improvement = 0
    started = time.time()

    model.train()
    for step in range(1, MAX_STEPS + 1):
        x, y = get_batch(train_data, config["context_length"], device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            rate = step / max(time.time() - started, 1e-6)
            print(
                f"Step {step:>4}/{MAX_STEPS} | "
                f"train {loss.item():.4f} | {rate:.2f} steps/s"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(
                model, val_data, config["context_length"], device
            )
            print(f"  validation loss: {val_loss:.4f}")

            if val_loss < best_val - 0.005:
                best_val = val_loss
                checks_without_improvement = 0
                save_best(model, config, step, loss.item(), val_loss)
                print("  new best chat checkpoint saved")
            else:
                checks_without_improvement += 1
                print(
                    f"  no meaningful improvement "
                    f"({checks_without_improvement}/{EARLY_STOP_PATIENCE})"
                )

                if checks_without_improvement >= EARLY_STOP_PATIENCE:
                    print("  early stopping: validation stopped improving")
                    break

    if not OUTPUT_PATH.exists():
        val_loss = evaluate(model, val_data, config["context_length"], device)
        save_best(model, config, step, loss.item(), val_loss)

    print()
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Best chat checkpoint: {OUTPUT_PATH}")
    print("Run: py -3.12 chat.py")


if __name__ == "__main__":
    main()
