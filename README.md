# LocalAI

LocalAI is an **AI-assisted learning project** for learning how language models work by building and training one locally from random weights.

This is not an Ollama wrapper and it does not download a pretrained language model. The model architecture, training loop, tokenizer, checkpointing and terminal chat are implemented in this repository with PyTorch.

## Current version

LocalAI now uses a custom **decoder-only Transformer** with:

- causal multi-head self-attention
- token and positional embeddings
- residual connections
- LayerNorm
- feed-forward networks
- GPU training with CUDA
- automatic checkpoints
- interactive multi-turn terminal chat
- a byte-level tokenizer that accepts arbitrary UTF-8 user input

The byte tokenizer means you can type text the model never saw during training without getting an "unknown character" error. Understanding that text still depends on what the model learned from its training data.

## AI-assisted learning

AI is used as a learning and development assistant to explain concepts, debug problems and guide implementation. The model itself starts from random weights and is trained locally.

## What this project does not use

- Ollama
- pretrained language models
- downloaded model weights
- hosted LLM APIs for inference

## Quick start

Create a local file called `data/training.txt`. Training data is intentionally ignored by Git.

A simple conversational format is:

```text
User: hello
Assistant: hey! how are you?

User: explain what ram is
Assistant: RAM is short-term working memory used by your computer.

User: bro i dont get it lol
Assistant: Think of RAM like a desk. The things you're using right now stay on the desk so they're quick to reach.
```

Train:

```powershell
py -3.12 train.py
```

Chat:

```powershell
py -3.12 chat.py
```

Then type anything directly into the terminal:

```text
You: yo whats up
AI: ...
```

Use `/clear` to reset conversation history and `/quit` to exit.

## Project structure

```text
LocalAI/
├── data/
│   ├── example.txt
│   └── training.txt       # local / ignored by Git
├── checkpoints/
│   ├── tokenizer.json     # generated locally
│   └── model.pt           # generated locally
├── tokenizer.py
├── model.py
├── train.py
├── chat.py
├── requirements.txt
└── README.md
```

## Training data matters

The Transformer does not automatically know English or facts. Training from scratch means its knowledge comes from the data used to train it.

A tiny conversation file is useful for testing the pipeline, but it will mostly be memorized. A useful small language model requires a much larger and more varied corpus. Future versions of this project will add a proper dataset/pretraining pipeline and a learned subword tokenizer.

## Hardware

Current development/training hardware:

- NVIDIA GeForce RTX 4060 Ti
- 8 GB VRAM
- CUDA-enabled PyTorch

The current configuration is intentionally kept small enough for consumer hardware.

## Goal

The goal is to understand the real components behind GPT-style language models on a practical scale: data, tokenization, attention, Transformer blocks, next-token prediction, optimization, checkpoints and inference.

A consumer-trained educational model will not match frontier systems such as ChatGPT, but it can implement and demonstrate the same fundamental language-modeling ideas.
