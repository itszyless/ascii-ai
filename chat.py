"""Chat with a trained LocalAI checkpoint in the terminal."""

from pathlib import Path

import torch

from model import LocalAI
from tokenizer import CharacterTokenizer


CHECKPOINT_DIR = Path("checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "model.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"


def main():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CharacterTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    config = checkpoint["config"]

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("LocalAI")
    print(f"Running on: {device}")
    print("Type 'quit' to exit.")
    print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        prompt = f"User: {user_input}\nAI:"

        try:
            prompt_ids = tokenizer.encode(prompt)
        except ValueError as error:
            print(f"AI: {error}")
            print()
            continue

        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated = model.generate(
            tokens,
            max_new_tokens=160,
            temperature=0.8,
            top_k=40,
        )

        new_tokens = generated[0, len(prompt_ids) :].tolist()
        response = tokenizer.decode(new_tokens)

        if "\nUser:" in response:
            response = response.split("\nUser:", 1)[0]

        print("AI:", response.strip())
        print()


if __name__ == "__main__":
    main()
