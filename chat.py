"""Interactive terminal chat for LocalAI v2."""

from pathlib import Path
import argparse

import torch

from model import LocalAI
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v2_best.pt"
FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v2.pt"
CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v2.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"

MAX_NEW_TOKENS = 160
TEMPERATURE = 0.65
TOP_K = 40


def build_history_prompt(turns, current_user, tokenizer, context_length):
    """Keep as many recent COMPLETE turns as fit, then append current question."""
    current = f"User: {current_user}\nAssistant:"
    current_ids = tokenizer.encode(current)

    if len(current_ids) >= context_length:
        return current_ids[-context_length:]

    selected = []
    used = len(current_ids)

    for user, assistant in reversed(turns):
        text = f"User: {user}\nAssistant: {assistant}\n"
        ids = tokenizer.encode(text)

        if used + len(ids) > context_length:
            break

        selected.append(ids)
        used += len(ids)

    prompt_ids = []
    for ids in reversed(selected):
        prompt_ids.extend(ids)
    prompt_ids.extend(current_ids)

    return prompt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        action="store_true",
        help="Test the pretrained base model instead of the chat model.",
    )
    args = parser.parse_args()

    if args.base:
        model_path = (
            BASE_MODEL_PATH if BASE_MODEL_PATH.exists() else FALLBACK_BASE_PATH
        )
    else:
        model_path = (
            CHAT_MODEL_PATH
            if CHAT_MODEL_PATH.exists()
            else (
                BASE_MODEL_PATH
                if BASE_MODEL_PATH.exists()
                else FALLBACK_BASE_PATH
            )
        )

    if not model_path.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "No v2 model found. Run prepare_data.py and train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = LocalAI(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 66)
    print("LocalAI v2")
    print("=" * 66)
    print(f"Running on: {device}")
    print(f"Checkpoint: {model_path.name}")
    print(f"Context: {model.context_length} tokens")
    print("Commands: /clear, /quit")
    print()

    turns = []

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in {"/quit", "quit", "exit"}:
            break

        if user_input.strip().lower() == "/clear":
            turns = []
            print("Conversation cleared.\n")
            continue

        if args.base:
            prompt_ids = tokenizer.encode(user_input)
            prompt_ids = prompt_ids[-model.context_length:]
            tokens = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            generated = model.generate(
                tokens,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                stop_token_ids={tokenizer.eos_id},
            )

            response = tokenizer.decode(
                generated[0, len(prompt_ids):].tolist()
            ).strip()

            print(f"AI continuation: {response}\n")
            continue

        prompt_ids = build_history_prompt(
            turns,
            user_input,
            tokenizer,
            model.context_length,
        )

        tokens = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        generated = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            stop_token_ids={tokenizer.eos_id},
        )

        new_tokens = generated[0, len(prompt_ids):].tolist()
        response = tokenizer.decode(new_tokens)

        for marker in ("\nUser:", "\nAssistant:"):
            if marker in response:
                response = response.split(marker, 1)[0]

        response = response.strip() or "..."
        print(f"AI: {response}\n")

        turns.append((user_input, response))


if __name__ == "__main__":
    main()
