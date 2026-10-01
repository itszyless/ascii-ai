# OrcaAI

OrcaAI is a from-scratch GPT-style language model and local AI playground.

It starts from random weights. No Ollama, no downloaded pretrained LLM weights, and no hosted model is used for inference.

## Current generation: OrcaAI v4

v3 successfully learned English-like syntax, but remained weak on factual meaning and repetition. v4 is a new experiment designed around those failures instead of simply training v3 longer.

v4 changes:

- ~47M parameters
- 12 Transformer blocks
- 8 attention heads
- 512-dimensional embeddings
- 512-token context
- 12,000-token byte-level BPE vocabulary
- RoPE positional encoding
- RMSNorm
- SwiGLU feed-forward layers
- bias-free attention/MLP projections
- residual-branch initialization scaling
- tied input/output embeddings
- PyTorch scaled dot-product attention
- gradient accumulation
- mixed-source pretraining
- deterministic train/validation separation
- separate validation metrics for each source
- top-k + top-p generation
- stronger repetition control

## Why v4 exists

OrcaAI v3 reached 80,000 steps and produced grammatical English, but examples such as:

```text
RAM is a type of -> a virus...
```

showed that surface-level language fluency was improving faster than factual/semantic reliability.

Instead of chat-fine-tuning that weak base, v4 changes both the model architecture and the data mix.

## Data

v4 prepares a fresh tokenizer and a 500M-token training corpus:

```text
350M FineWeb-Edu tokens
150M English Wikipedia tokens
7M FineWeb-Edu validation tokens
3M Wikipedia validation tokens
```

The two public sources are streamed during preparation, so their full datasets do not need to be downloaded locally.

Prepare v4:

```powershell
py -3.12 prepare_v4.py
```

This creates:

```text
data/fineweb_v4.bin
data/fineweb_v4_val.bin
data/wikipedia_v4.bin
data/wikipedia_v4_val.bin
checkpoints/tokenizer_v4.json
```

## Pretraining

Start v4:

```powershell
py -3.12 train_v4.py
```

The default run is 120,000 optimizer steps.

Each step uses two micro-batches of 4 × 512 tokens, so the effective batch is 8 sequences and the full run processes about 491.5M token positions.

Training automatically saves:

```text
checkpoints/model_v4.pt
checkpoints/model_v4_best.pt
```

If training is interrupted, run the same command again and it resumes from the latest compatible checkpoint.

## Test the base model

Before any chat fine-tuning:

```powershell
py -3.12 test_v4.py
```

Useful tests:

```text
RAM is used to
The CPU executes
A function in programming is
Paris is the capital of
Water freezes at
A dog is an
```

Do not run chat fine-tuning merely because the output is grammatical. The base should also be relevant and reasonably factual.

## Diagnose v4 before fine-tuning

After pretraining, run the diagnostic benchmark before deciding whether the base is ready for chat tuning:

```powershell
py -3.12 diagnose_v4.py --extended
```

It compares `model_v4_best.pt` and `model_v4.pt`, scores factual continuations against distractors, generates multiple seeded continuations, measures repetition, and saves a full report to:

```text
data/v4_diagnostic_report.json
```

The diagnostic does not train or modify model weights.

## Chat fine-tuning

Once the v4 base passes the sanity test:

```powershell
py -3.12 finetune_v4.py
```

It uses `data/chat.txt`, deduplicates repeated response patterns, trains at a low learning rate, and periodically replays pretraining text to reduce catastrophic forgetting.

Output:

```text
checkpoints/chat_model_v4.pt
```

The pretrained base checkpoint is never overwritten.

## Web app

Run:

```powershell
py -3.12 web_app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web app automatically prefers v4 checkpoints when they exist, then falls back to v3/v2.

OrcaAI includes:

- local chat history
- response regeneration
- ratings
- copy/share/download/read-aloud actions
- smooth Thinking animation
- progressive response reveal
- local memory
- local knowledge retrieval
- response candidate ranking
- checkpoint/device status
- OrcaAI branding and favicon

## Model Lab

Open:

```text
http://127.0.0.1:5000/lab
```

The Model Lab exposes:

- architecture information
- parameter count
- GPU/VRAM information
- tokenizer visualization
- token IDs
- next-token probabilities
- entropy
- temperature/top-k controls
- latency and tokens/sec
- checkpoint explorer
- memory manager
- knowledge retrieval
- evaluation prompts
- chat/feedback statistics

## Hardware target

Current development target:

```text
NVIDIA GeForce RTX 4060 Ti
8 GB VRAM
Windows
Python 3.12
CUDA-enabled PyTorch
```

v4 uses a micro-batch of 4 with gradient accumulation specifically to keep the effective batch useful without assuming huge VRAM.

## Install

```powershell
py -3.12 -m pip install -r requirements.txt
```

For the current CUDA setup:

```powershell
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Project layout

The root is intentionally kept small. Active training utilities and diagnostics live under `tools/v4/`, while old v3 training scripts are archived under `legacy/v3/`.

Use the command launcher for normal work:

```powershell
py -3.12 orca.py help
```

Common commands:

```powershell
py -3.12 orca.py train
py -3.12 orca.py test
py -3.12 orca.py diagnose-unseen
py -3.12 orca.py knowledge
py -3.12 orca.py benchmark-knowledge
py -3.12 orca.py web
```

Main layout:

```text
orca.py               single command launcher
model.py              legacy-compatible v1-v3 Transformer
model_v4.py           active v4 Transformer
tokenizer.py          byte-level BPE tokenizer
web_app.py            local OrcaAI web app
lab_api.py            Model Lab backend
quality_layer.py      retrieval + response ranking

prepare_v4.py         one-time v4 corpus preparation
train_v4.py           base v4 pretraining
test_v4.py            base-model sanity test
finetune_v4.py        chat fine-tuning

tools/v4/             grounding, diagnostics, knowledge runs, benchmarks
legacy/v3/            archived v3 training scripts
templates/            web UI
static/               UI assets
data/                 local datasets/reports (gitignored)
checkpoints/          local model/tokenizer checkpoints (gitignored)
```

## What the model is actually learning

During pretraining, OrcaAI never receives an instruction such as "learn English."

It repeatedly sees token sequences and is asked to predict the next token. When its prediction is wrong, backpropagation slightly changes millions of parameters. Across hundreds of millions of token examples, useful statistical structure becomes encoded in those parameters.

That is enough for a model to gradually learn spelling, punctuation, grammar, common phrases, document structure, relationships between words, and eventually some facts and concepts.

Grammar can emerge before factual understanding because predicting local sentence structure is easier than reliably representing the many relationships needed for accurate world knowledge.

## Reality check

OrcaAI implements real next-token language-model training, but scale still matters enormously.

A ~47M-parameter model trained on one RTX 4060 Ti is a serious learning project, not a frontier model. v4 is intended to improve the quality ceiling while keeping the entire training process understandable and locally runnable.
