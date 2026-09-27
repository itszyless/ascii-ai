"""Chat fine-tuning for a successfully pretrained LocalAI v3 base.

Do not run this until test_v3.py produces coherent language.
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
MODEL_PATH = Path("checkpoints/model_v3_best.pt")
FALLBACK_MODEL_PATH = Path("checkpoints/model_v3.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer_v3.json")
OUTPUT_PATH = Path("checkpoints/chat_model_v3.pt")

MAX_STEPS = 600
BATCH_SIZE = 10
LEARNING_RATE = 8e-6
PRINT_EVERY = 20
EVAL_EVERY = 20
EVAL_BATCHES = 30
VALIDATION_FRACTION = 0.10
EARLY_STOP_PATIENCE = 5
MIN_IMPROVEMENT = 0.003
TRAIN_LAST_BLOCKS = 3
SEED = 1337


def normalize(text):
    return " ".join(text.lower().split())


def parse_conversations(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    pattern = re.compile(
        r"User:\s*(.*?)\r?\nAssistant:\s*(.*?)(?=\r?\nUser:|\Z)",
        re.DOTALL,
    )

    conversations = []
    for block in blocks:
        turns = []
        for match in pattern.finditer(block):
            user = match.group(1).strip()
            assistant = match.group(2).strip()
            if user and assistant:
                turns.append((user, assistant))
        if turns:
            conversations.append(turns)

    if len(conversations) < 20:
        raise ValueError("Need at least 20 conversations in data/chat.txt.")

    return conversations


def rebalance(conversations):
    unique_pairs = []
    seen_pairs = set()

    for conversation in conversations:
        for user, assistant in conversation:
            key = (normalize(user), normalize(assistant))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            unique_pairs.append([(user, assistant)])

    multi = []
    seen_multi = set()

    for conversation in conversations:
        if len(conversation) < 2:
            continue

        key = tuple(
            (normalize(user), normalize(assistant))
            for user, assistant in conversation
        )

        if key in seen_multi:
            continue

        seen_multi.add(key)
        multi.append(conversation)

    rng = random.Random(SEED)
    rng.shuffle(multi)

    multi_limit = min(
        len(multi),
        max(40, len(unique_pairs) // 2),
    )

    examples = unique_pairs + multi[:multi_limit]
    rng.shuffle(examples)
    return examples


def split_examples(examples):
    examples = list(examples)
    rng = random.Random(SEED)
    rng.shuffle(examples)
    count = max(1, int(len(examples) * VALIDATION_FRACTION))
    return examples[count:], examples[:count]


def make_example(turns, tokenizer, context_length):
    token_ids = []
    target_mask = []

    for user, assistant in turns:
        prompt = tokenizer.encode(
            f"User: {user}\nAssistant:"
        )
        response = tokenizer.encode(
            f" {assistant}",
            add_eos=True,
        )

        token_ids.extend(prompt)
        target_mask.extend([False] * len(prompt))
        token_ids.extend(response)
        target_mask.extend([True] * len(response))

    if len(token_ids) < 2:
        return None

    if len(token_ids) > context_length + 1:
        token_ids = token_ids[-(context_length + 1):]
        target_mask = target_mask[-(context_length + 1):]

    x_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    next_mask = target_mask[1:]

    labels = [
        token if enabled else -100
        for token, enabled in zip(next_ids, next_mask)
    ]

    if all(label == -100 for label in labels):
        return None

    return (
        torch.tensor(x_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_examples(raw, tokenizer, context_length):
    result = []

    for turns in raw:
        item = make_example(turns, tokenizer, context_length)
        if item is not None:
            result.append(item)

    if not result:
        raise ValueError("No usable chat examples.")

    return result


def get_batch(examples, tokenizer, context_length, device):
    chosen = random.choices(examples, k=BATCH_SIZE)

    x = torch.full(
        (BATCH_SIZE, context_length),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (BATCH_SIZE, context_length),
        -100,
        dtype=torch.long,
    )

    for row, (ids, row_labels) in enumerate(chosen):
        length = min(len(ids), context_length)
        x[row, :length] = ids[:length]
        labels[row, :length] = row_labels[:length]

    return x.to(device), labels.to(device)


def loss_fn(model, x, labels):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.01,
    )


@torch.no_grad()
def evaluate(model, examples, tokenizer, context_length, device, use_bf16):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, labels = get_batch(
            examples,
            tokenizer,
            context_length,
            device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = loss_fn(model, x, labels)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in list(model.blocks)[-TRAIN_LAST_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.final_norm.parameters():
        parameter.requires_grad = True


def save_best(model, config, step, train_loss, val_loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat_v3",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    if not CHAT_PATH.exists():
        raise FileNotFoundError("Missing data/chat.txt.")

    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not model_path.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Finish v3 pretraining first.")

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]

    conversations = parse_conversations(
        CHAT_PATH.read_text(encoding="utf-8")
    )
    balanced = rebalance(conversations)
    train_raw, val_raw = split_examples(balanced)

    train_examples = build_examples(
        train_raw,
        tokenizer,
        config["context_length"],
    )
    val_examples = build_examples(
        val_raw,
        tokenizer,
        config["context_length"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    freeze_model(model)

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("=" * 70)
    print("LocalAI v3 chat fine-tuning")
    print("=" * 70)
    print(f"Base checkpoint:      {model_path.name}")
    print(f"Base step:            {checkpoint.get('step', 'unknown')}")
    print(f"Raw conversations:    {len(conversations):,}")
    print(f"Balanced examples:    {len(balanced):,}")
    print(f"Train examples:       {len(train_examples):,}")
    print(f"Validation examples:  {len(val_examples):,}")
    print(f"Context:              {config['context_length']}")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    print()

    best_val = float("inf")
    stale = 0
    started = time.time()

    for step in range(1, MAX_STEPS + 1):
        x, labels = get_batch(
            train_examples,
            tokenizer,
            config["context_length"],
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = loss_fn(model, x, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            rate = step / max(time.time() - started, 1e-6)
            print(
                f"Step {step:>4}/{MAX_STEPS} | "
                f"train {loss.item():.4f} | "
                f"{rate:.2f} steps/s"
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

            print(f"  validation loss: {val_loss:.4f}")

            if val_loss < best_val - MIN_IMPROVEMENT:
                best_val = val_loss
                stale = 0
                save_best(
                    model,
                    config,
                    step,
                    loss.item(),
                    val_loss,
                )
                print("  new best v3 chat checkpoint saved")
            else:
                stale += 1
                print(
                    f"  no meaningful improvement "
                    f"({stale}/{EARLY_STOP_PATIENCE})"
                )

                if stale >= EARLY_STOP_PATIENCE:
                    print("  early stopping")
                    break

    print()
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Saved: {OUTPUT_PATH}")
    print("Next: py -3.12 web_app.py")


if __name__ == "__main__":
    main()
