"""Interactive sanity test for the LocalAI v3 base model.

This is NOT assistant/chat mode. Type sentence beginnings and inspect whether
the model can continue them coherently.
"""

from pathlib import Path

import torch

from model import LocalAI
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
MODEL_PATH = CHECKPOINT_DIR / "model_v3_best.pt"
FALLBACK_MODEL_PATH = CHECKPOINT_DIR / "model_v3.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v3.json"


def main():
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH

    if not model_path.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Finish v3 pretraining first.")

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

    print("=" * 70)
    print("LocalAI v3 base sanity test")
    print("=" * 70)
    print(f"Checkpoint: {model_path.name}")
    print(f"Step:       {checkpoint.get('step', 'unknown')}")
    print("This is text continuation, not chat.")
    print("Try: RAM is a type of")
    print("Type /quit to leave.")
    print()

    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in {"/quit", "quit", "exit"}:
            break
        if not prompt:
            continue

        ids = tokenizer.encode(prompt)
        ids = ids[-model.context_length:]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)

        generated = model.generate(
            tokens,
            max_new_tokens=80,
            temperature=0.55,
            top_k=30,
            repetition_penalty=1.05,
            stop_token_ids={tokenizer.eos_id},
        )

        continuation = tokenizer.decode(
            generated[0, len(ids):].tolist()
        ).strip()

        print(f"Continuation: {continuation}\n")


if __name__ == "__main__":
    main()
