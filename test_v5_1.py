"""Standardized semantic test for OrcaAI v5.1."""

from __future__ import annotations

from pathlib import Path

import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


CHECKPOINT_DIR = Path("checkpoints")
BEST_PATH = CHECKPOINT_DIR / "model_v5_1_best.pt"
LATEST_PATH = CHECKPOINT_DIR / "model_v5_1.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"

PROMPTS = [
    "RAM is a type of",
    "The CPU executes",
    "A database is",
    "Paris is the capital of",
    "Berlin is the capital of",
    "Ice melts when",
    "Water freezes at",
    "A dog is an",
    "Deutschland ist",
    "Die Hauptstadt von Österreich ist",
    "Die Hauptstadt von Deutschland ist",
    "Wasser gefriert bei",
    "RAM ist ein",
    "Paris est la capitale de",
    "Un ordinateur est",
    "Madrid es la capital de",
    "Una CPU es",
]


def main():
    model_path = BEST_PATH if BEST_PATH.exists() else LATEST_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            "No v5.1 checkpoint found. Run: py -3.12 orca.py train-v5-1"
        )

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model = OrcaAIv5(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 76)
    print("OrcaAI v5.1 semantic sanity test")
    print("=" * 76)
    print(f"Checkpoint: {model_path.name}")
    print(f"Step:       {checkpoint.get('step', 'unknown')}")
    print(f"Best val:   {checkpoint.get('best_val', 'unknown')}")
    print(f"Source:     {checkpoint.get('source_checkpoint', 'unknown')}")
    print("This is raw text continuation, not chat.")
    print()

    for index, prompt in enumerate(PROMPTS, 1):
        ids = tokenizer.encode(prompt)
        x = torch.tensor(
            [ids],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            generated = model.generate(
                x,
                max_new_tokens=70,
                temperature=0.30,
                top_k=20,
                top_p=0.90,
                repetition_penalty=1.08,
                stop_token_ids={tokenizer.eos_id},
            )

        continuation = tokenizer.decode(
            generated[0, len(ids):].tolist()
        ).strip()

        print(f"[{index:02d}] {prompt}")
        print(f"     -> {continuation}")
        print()

    print("=" * 76)
    print("If these continuations are materially better, continue with:")
    print("  py -3.12 orca.py prepare-v5-chat")
    print("  py -3.12 orca.py finetune-v5")
    print("  py -3.12 orca.py test-chat-v5")
    print("=" * 76)


if __name__ == "__main__":
    main()
