"""Interactive terminal chat for a trained LocalAI model."""

from pathlib import Path

import torch

from model import LocalAI
from tokenizer import ByteTokenizer


CHECKPOINT_DIR = Path("checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

MAX_NEW_TOKENS = 220
TEMPERATURE = 0.75
TOP_K = 50


def main():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer.load(TOKENIZER_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = LocalAI(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 58)
    print("LocalAI")
    print("=" * 58)
    print(f"Running on: {device}")
    if device.type == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print()
    print("Write anything you want.")
    print("Commands: /clear, /quit")
    print()

    conversation = ""

    while True:
        user_input = input("You: ")

        if user_input.strip().lower() in {"/quit", "quit", "exit"}:
            break

        if user_input.strip().lower() == "/clear":
            conversation = ""
            print("Conversation cleared.\n")
            continue

        # The history is included so the model can see recent turns.
        conversation += f"User: {user_input}\nAssistant:"

        prompt_ids = tokenizer.encode(conversation)
        prompt_ids = prompt_ids[-model.context_length :]

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
        )

        new_tokens = generated[0, len(prompt_ids) :].tolist()
        response = tokenizer.decode(new_tokens)

        # Stop displaying once the model starts inventing the next speaker.
        for marker in ("\nUser:", "\nAssistant:"):
            if marker in response:
                response = response.split(marker, 1)[0]

        response = response.strip()

        if not response:
            response = "..."

        print(f"AI: {response}\n")

        conversation += f" {response}\n"


if __name__ == "__main__":
    main()
