# LocalAI

LocalAI is an **AI-assisted learning project** focused on understanding how modern language models work by building and training one from scratch.

The goal is not to wrap an existing model or rely on tools such as Ollama. Instead, this project explores the full process of creating a local language model with PyTorch, including tokenization, embeddings, self-attention, Transformer architecture, training, checkpoints, and text generation.

## Goals

- Build a GPT-style Transformer from scratch
- Train the model locally on my own hardware
- Start from randomly initialized weights
- Use my own training data and datasets
- Learn how tokenization and next-token prediction work
- Learn self-attention and Transformer architecture
- Use GPU acceleration with PyTorch and CUDA
- Create a local chat interface
- Document the learning process as the project evolves

## What this project does not use

- Ollama
- Pretrained language models
- Downloaded model weights
- OpenAI API or other hosted LLM APIs for inference

The model itself is trained locally from scratch.

## AI-assisted learning

AI is used as a **learning and development assistant** throughout this project to explain concepts, help debug code, and guide implementation.

The purpose is to learn how the model works rather than hide the implementation behind an existing AI service. The model being built in this repository does not depend on another language model when it runs.

## Hardware

Development and training are currently being done on an NVIDIA GeForce RTX 4060 Ti with 8 GB VRAM.

## Current status

Early development.

The project started with a small recurrent neural-network experiment to understand tokenization, next-token prediction, loss, and training. The next stage is a custom GPT-style decoder-only Transformer trained from random weights.

## Planned structure

```text
LocalAI/
├── data/
│   └── training.txt
├── checkpoints/
├── tokenizer.py
├── model.py
├── train.py
├── chat.py
├── requirements.txt
└── README.md
```

## Disclaimer

This is an educational project. A model trained on a single consumer GPU will be far smaller than large commercial language models, but the aim is to understand and implement the same fundamental ideas on a practical scale.
