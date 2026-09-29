"""Interactive sanity test for the OrcaAI v4 base model."""

from pathlib import Path

import torch

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_v4_best.pt"
FALLBACK_MODEL_PATH = CHECKPOINT_DIR / "model_v4.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"


def main():
    model_path = (
        BEST_MODEL_PATH
        if BEST_MODEL_PATH.exists()
        else FALLBACK_MODEL_PATH
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "No v4 checkpoint found. Run train_v4.py first."
        )

    tokenizer = LocalTokenizer.load(
        TOKENIZER_PATH
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = OrcaAIv4(
        **checkpoint["config"]
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    print("=" * 72)
    print("OrcaAI v4 base sanity test")
    print("=" * 72)
    print(f"Checkpoint: {model_path.name}")
    print(f"Step:       {checkpoint.get('step', 'unknown')}")
    print("This is text continuation, not chat.")
    print()
    print("Recommended prompts:")
    print("  RAM is used to")
    print("  The CPU executes")
    print("  Paris is the capital of")
    print("  Water freezes at")
    print("Type /quit to leave.")
    print()

    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() == "/quit":
            break

        ids = tokenizer.encode(prompt)
        tokens = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            generated = model.generate(
                tokens,
                max_new_tokens=120,
                temperature=0.40,
                top_k=40,
                top_p=0.90,
                repetition_penalty=1.12,
                stop_token_ids={
                    tokenizer.eos_id
                },
            )

        continuation = tokenizer.decode(
            generated[
                0,
                len(ids):,
            ].tolist()
        )

        print(
            f"Continuation: "
            f"{continuation.strip()}"
        )
        print()


if __name__ == "__main__":
    main()
