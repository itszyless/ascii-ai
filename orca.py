"""Small command launcher for OrcaAI.

Keeps the project root clean while providing short commands for the active v4
pipeline. Run "py -3.12 orca.py help" to see available commands.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "prepare": "prepare_v4.py",
    "train": "train_v4.py",
    "test": "test_v4.py",
    "finetune": "finetune_v4.py",\n    "make-chat-data": "tools/v4/generate_chat_data.py",
    "test-chat": "tools/v4/test_chat.py",
    "ground-data": "tools/v4/generate_grounding.py",
    "ground": "tools/v4/ground.py",
    "diagnose": "tools/v4/diagnose.py",
    "diagnose-unseen": "tools/v4/diagnose_unseen.py",
    "knowledge": "tools/v4/continue_knowledge.py",
    "benchmark-knowledge": "tools/v4/benchmark_knowledge.py",
    "analyze-knowledge": "tools/v4/analyze_knowledge.py",
    "web": "web_app.py",
}


def print_help():
    print("OrcaAI command launcher")
    print()
    print("Usage:")
    print("  py -3.12 orca.py <command> [args]")
    print()
    print("Commands:")
    descriptions = {
        "prepare": "prepare the v4 tokenizer and pretraining data",
        "train": "start/resume v4 base pretraining",
        "test": "interactive v4 base-model test",
        "ground-data": "create the factual grounding dataset",
        "ground": "run factual grounding",
        "diagnose": "compare v4 checkpoints on the main diagnostic",
        "diagnose-unseen": "test grounding on held-out facts",
        "knowledge": "run/resume knowledge-focused continued pretraining",
        "benchmark-knowledge": "long held-out + generation sweep benchmark",
        "analyze-knowledge": "summarize the latest knowledge benchmark",
        "finetune": "run v4 chat fine-tuning when the base is ready",\n        "make-chat-data": "generate the curated v4 chat dataset",
        "test-chat": "sanity-test the v4 chat checkpoint",
        "web": "start the local OrcaAI web app",
    }
    width = max(len(name) for name in COMMANDS)
    for name in COMMANDS:
        print(f"  {name:<{width}}  {descriptions[name]}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"help", "-h", "--help"}:
        print_help()
        return

    command = sys.argv[1]
    relative_path = COMMANDS.get(command)

    if relative_path is None:
        print(f"Unknown command: {command}")
        print()
        print_help()
        raise SystemExit(2)

    script_path = ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(
            f"Command target is missing: {script_path}"
        )

    # Preserve any extra CLI arguments for the selected script.
    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
