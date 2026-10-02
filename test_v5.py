"""Interactive sanity test for the OrcaAI v5 base model."""

from pathlib import Path

import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
BOOST_BEST_PATH = CHECKPOINT_DIR / "model_v5_boost_best.pt"
BOOST_PATH = CHECKPOINT_DIR / "model_v5_boost.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_v5_best.pt"
FALLBACK_MODEL_PATH = CHECKPOINT_DIR / "model_v5.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"


def main():
    model_path = next(
        (
            path for path in (
                BOOST_BEST_PATH,
                BOOST_PATH,
                BEST_MODEL_PATH,
                FALLBACK_MODEL_PATH,
            )
            if path.exists()
        ),
        FALLBACK_MODEL_PATH,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "No v5 checkpoint found. Run: py -3.12 orca.py train-v5 first."
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

    model = OrcaAIv5(
        **checkpoint["config"]
    ).to(device)
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    print("=" * 72)
    print("OrcaAI v5 base sanity test")
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
    print("  Die Hauptstadt von Österreich ist")
    print("  12 mal 9 ist")
    print("  7 × 8 =")
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
