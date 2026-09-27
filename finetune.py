"""Instruction-tune LocalAI v2 on chat.txt.

User text is context only. Loss is applied to Assistant response tokens, which
reduces the tendency to imitate the user's side of the dataset.
"""

from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F

from model import LocalAI
from tokenizer import LocalTokenizer


CHAT_PATH = Path("data/chat.txt")
MODEL_PATH = Path("checkpoints/model_v2_best.pt")
FALLBACK_MODEL_PATH = Path("checkpoints/model_v2.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer_v2.json")
OUTPUT_PATH = Path("checkpoints/chat_model_v2.pt")

MAX_STEPS = 1000
BATCH_SIZE = 12
LEARNING_RATE = 1e-5
PRINT_EVERY = 25
EVAL_EVERY = 25
EVAL_BATCHES = 30
VALIDATION_FRACTION = 0.10
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT = 0.003
SEED = 1337


def parse_pairs(text):
    pattern = re.compile(
        r"User:\s*(.*?)\r?\nAssistant:\s*(.*?)(?=\r?\n\s*\r?\nUser:|\Z)",
        re.DOTALL,
    )
    pairs = []
    for match in pattern.finditer(text):
        user = match.group(1).strip()
        assistant = match.group(2).strip()
        if user and assistant:
            pairs.append((user, assistant))
    if len(pairs) < 20:
        raise ValueError(
            "Could not find at least 20 'User:' / 'Assistant:' examples in chat.txt."
        )
    return pairs


def split_pairs(pairs):
    pairs = list(pairs)
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    val_count = max(1, int(len(pairs) * VALIDATION_FRACTION))
    return pairs[val_count:], pairs[:val_count]


def make_example(pair, tokenizer, context_length):
    user, assistant = pair
    prompt = f"User: {user}\nAssistant:"
    response = f" {assistant}\n"

    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response, add_eos=True)

    ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(ids) > context_length:
        # Keep the end: Assistant tokens matter most for instruction tuning.
        ids = ids[-context_length:]
        labels = labels[-context_length:]

    if all(label == -100 for label in labels):
        return None

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_examples(pairs, tokenizer, context_length):
    examples = []
    for pair in pairs:
        example = make_example(pair, tokenizer, context_length)
        if example is not None:
            examples.append(example)
    if not examples:
        raise ValueError("No usable chat examples after tokenization.")
    return examples


def get_batch(examples, tokenizer, context_length, device):
    chosen = random.choices(examples, k=BATCH_SIZE)
    x = torch.full(
        (BATCH_SIZE, context_length),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    y = torch.full(
        (BATCH_SIZE, context_length),
        -100,
        dtype=torch.long,
    )

    for row, (ids, labels) in enumerate(chosen):
        length = min(len(ids), context_length)
        x[row, :length] = ids[:length]
        y[row, :length] = labels[:length]

    return x.to(device), y.to(device)


def assistant_loss(model, x, labels):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, examples, tokenizer, context_length, device, use_bf16):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, labels = get_batch(
            examples, tokenizer, context_length, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = assistant_loss(model, x, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_best(model, config, step, train_loss, val_loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat_v2",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Run v2 prepare_data.py first.")
    if not CHAT_PATH.exists():
        raise FileNotFoundError("Missing data/chat.txt.")

    base_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not base_path.exists():
        raise FileNotFoundError("Run v2 train.py first.")

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    pairs = parse_pairs(CHAT_PATH.read_text(encoding="utf-8"))
    train_pairs, val_pairs = split_pairs(pairs)
    train_examples = build_examples(
        train_pairs, tokenizer, config["context_length"]
    )
    val_examples = build_examples(
        val_pairs, tokenizer, config["context_length"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    print("=" * 66)
    print("LocalAI v2 assistant-only fine-tuning")
    print("=" * 66)
    print(f"Base checkpoint:     {base_path.name}")
    print(f"Device:              {device}")
    print(f"BF16:                {'yes' if use_bf16 else 'no'}")
    print(f"Train conversations: {len(train_examples):,}")
    print(f"Val conversations:   {len(val_examples):,}")
    print(f"Max steps:           {MAX_STEPS:,}")
    print(f"Learning rate:       {LEARNING_RATE:.1e}")
    print()

    best_val = float("inf")
    stale_checks = 0
    started = time.time()

    model.train()
    for step in range(1, MAX_STEPS + 1):
        x, labels = get_batch(
            train_examples, tokenizer, config["context_length"], device
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = assistant_loss(model, x, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            rate = step / max(time.time() - started, 1e-6)
            print(
                f"Step {step:>4}/{MAX_STEPS} | "
                f"assistant train loss {loss.item():.4f} | {rate:.2f} steps/s"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(
                model,
                val_examples,
                tokenizer,
                config["context_length"],
                device,
                use_bf16,
            )
            print(f"  assistant validation loss: {val_loss:.4f}")

            if val_loss < best_val - MIN_IMPROVEMENT:
                best_val = val_loss
                stale_checks = 0
                save_best(model, config, step, loss.item(), val_loss)
                print("  new best chat checkpoint saved")
            else:
                stale_checks += 1
                print(
                    f"  no meaningful improvement "
                    f"({stale_checks}/{EARLY_STOP_PATIENCE})"
                )
                if stale_checks >= EARLY_STOP_PATIENCE:
                    print("  early stopping")
                    break

    print()
    print(f"Best assistant validation loss: {best_val:.4f}")
    print(f"Chat checkpoint: {OUTPUT_PATH}")
    print("Run: py -3.12 chat.py")


if __name__ == "__main__":
    main()
