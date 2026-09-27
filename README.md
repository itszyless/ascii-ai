# OrcaAI

OrcaAI is a from-scratch GPT-style language model and local AI playground built to understand how modern language models actually work.

The model starts from random weights. No Ollama, no downloaded pretrained language-model weights, and no hosted LLM is used for inference.

> **Current status:** OrcaAI v3 is still experimental. The base model is being pretrained locally and is not expected to match frontier assistants.

## What makes OrcaAI different

OrcaAI is not a wrapper around another chatbot. The project contains its own tokenizer, decoder-only Transformer, pretraining loop, chat fine-tuning, local web UI, checkpoint system, evaluation tools, memory layer, knowledge retrieval, and developer lab.

The core pipeline is:

```text
raw text
   ↓
fresh byte-level BPE tokenizer
   ↓
decoder-only Transformer
   ↓
next-token pretraining
   ↓
base checkpoint
   ↓
chat fine-tuning
   ↓
OrcaAI chat model
```

## OrcaAI v3

The current v3 configuration uses:

- decoder-only Transformer
- 8 Transformer blocks
- 8 attention heads
- 512-dimensional embeddings
- 512-token context window
- 8,192-token byte-level BPE vocabulary
- tied token/input-output embeddings
- causal self-attention
- GELU feed-forward networks
- residual connections and LayerNorm
- AdamW optimization
- warmup + cosine learning-rate decay
- BF16 when supported
- validation checkpoints
- automatic resume checkpoints
- repetition-aware generation
- local chat fine-tuning

With the current vocabulary this is roughly a 30M-parameter model.

## Training data

The v3 pipeline prepares a fresh tokenizer and training corpus from streamed educational web text.

Default v3 preparation:

```text
250M training tokens
5M validation tokens
8,192-token vocabulary
```

Prepare it with:

```powershell
py -3.12 prepare_v3.py
```

No pretrained model weights are downloaded during data preparation.

## Pretraining

Start or resume v3 pretraining:

```powershell
py -3.12 train_v3.py
```

The run is configured for up to 80,000 steps. A checkpoint is saved regularly, so training can be stopped and resumed later.

Important files:

```text
checkpoints/model_v3.pt
checkpoints/model_v3_best.pt
checkpoints/tokenizer_v3.json
```

Before chat fine-tuning, test the raw base model:

```powershell
py -3.12 test_v3.py
```

Useful continuation prompts:

```text
RAM is a type of
The CPU is responsible for
A computer uses memory to
A function in programming is
Paris is the capital of
```

The base should produce coherent and relevant continuations before moving to chat fine-tuning.

## Chat fine-tuning

Once the base model is good enough:

```powershell
py -3.12 finetune_v3.py
```

The chat dataset lives at:

```text
data/chat.txt
```

The fine-tuned checkpoint is saved separately as:

```text
checkpoints/chat_model_v3.pt
```

This means fine-tuning never overwrites the pretrained base checkpoint.

## Web app

Start OrcaAI locally:

```powershell
py -3.12 web_app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The web UI includes chat history, regeneration, ratings, copy/share/download/read-aloud actions, a smooth Thinking state, progressive answer reveal, and local checkpoint status.

The OrcaAI whale mark is used as the favicon and assistant identity.

## Model Lab

Open:

```text
http://127.0.0.1:5000/lab
```

The Model Lab exposes model internals and experimental tools, including:

- live checkpoint and model metadata
- parameter count
- architecture viewer
- GPU / VRAM information
- tokenizer visualization
- token IDs
- next-token probabilities
- prediction entropy
- generation controls
- temperature
- top-k
- repetition penalty
- seed control
- latency
- tokens per second
- checkpoint explorer
- local memories
- local knowledge documents
- retrieval search
- chat statistics
- feedback statistics
- repeatable evaluation prompts

## Local memory and knowledge

OrcaAI can store optional local memories and knowledge documents outside the model weights.

These are retrieval layers, not hidden retraining. Relevant local context can be inserted into the model prompt when useful.

Files remain local under `data/`.

## Hardware target

Current development machine:

```text
NVIDIA GeForce RTX 4060 Ti
8 GB VRAM
Windows
Python 3.12
CUDA-enabled PyTorch
```

The architecture is deliberately small enough to experiment with on consumer hardware.

## Install

Install the Python dependencies:

```powershell
py -3.12 -m pip install -r requirements.txt
```

For the current Windows CUDA setup, PyTorch is installed separately:

```powershell
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Repository layout

```text
model.py              Transformer architecture
tokenizer.py          byte-level BPE tokenizer
prepare_v3.py         v3 corpus + tokenizer preparation
train_v3.py           v3 pretraining
test_v3.py            raw base-model sanity test
finetune_v3.py        chat fine-tuning
web_app.py            local Flask chat application
lab_api.py            Model Lab backend
quality_layer.py      retrieval + response candidate ranking
templates/            chat and Lab pages
static/               UI, OrcaAI icon and frontend code
data/                 local datasets, chats, memory and knowledge
checkpoints/          local tokenizer/model checkpoints
```

## Important reality check

OrcaAI implements the same broad learning idea used by GPT-style models: predict the next token across large amounts of text, then perform post-training for useful interaction.

Scale matters enormously. A roughly 30M-parameter model trained on one consumer GPU will not approach ChatGPT-class systems. The point of OrcaAI is to build and inspect the mechanism itself rather than disguise an API call as a custom AI.

## Privacy

Inference is performed by OrcaAI's own local weights. Chat history, memory and local knowledge are stored on the machine.

## Project status

OrcaAI is actively being developed. Model quality will change as v3 receives more pretraining and improved post-training.
