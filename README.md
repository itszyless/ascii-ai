# LocalAI

LocalAI is an **AI-assisted learning project** that builds and trains a GPT-style language model locally from random weights.

No Ollama, no downloaded pretrained model weights, and no hosted LLM is used for inference.

## What changed

LocalAI now has a real two-stage language-model pipeline:

**Pretraining:** general text -> tokenizer + Transformer -> language patterns

**Fine-tuning:** conversations -> pretrained Transformer -> chat behavior

The model is not a database of possible questions and answers. During pretraining it repeatedly predicts the next token. Gradient descent changes the model's parameters so it can learn statistical language patterns and generalize to text that was not literally present in the dataset.

## Architecture

- decoder-only Transformer
- causal multi-head self-attention
- learned token + positional embeddings
- residual connections and LayerNorm
- GELU feed-forward networks
- tied input/output embeddings
- byte-level BPE tokenizer trained from scratch
- CUDA training
- validation loss
- cosine learning-rate decay
- gradient clipping
- automatic checkpoints + resume
- multi-turn terminal chat

## 1. Install data/tokenizer dependencies

Keep your existing CUDA-enabled PyTorch installation.

```powershell
py -3.12 -m pip install numpy tokenizers datasets
```

## 2. Prepare real pretraining data

For the first run:

```powershell
py -3.12 prepare_data.py
```

This downloads the small WikiText-2 text corpus and trains LocalAI's own 4,096-token byte-level BPE vocabulary from scratch. It does **not** download pretrained model weights.

It creates:

```text
data/train.bin
data/val.bin
checkpoints/tokenizer.json
```

Once the entire pipeline works, a substantially larger WikiText-103 run can be prepared with:

```powershell
py -3.12 prepare_data.py --full
```

Do not jump to the full corpus until the small run works.

## 3. Pretrain

```powershell
py -3.12 train.py
```

The model starts with random weights and learns by predicting the next token across the corpus.

Training now reports both training loss and validation loss. A falling validation loss is much more meaningful than memorizing a tiny hand-written file.

Training can resume from a compatible `checkpoints/model.pt`.

## 4. Teach it to chat

Pretraining teaches language, but it does not automatically teach the model that it should act like an assistant.

Create:

```text
data/chat.txt
```

with varied conversations:

```text
User: hey
Assistant: hey! what's up?

User: what is ram?
Assistant: RAM is short-term memory that a computer uses for data it needs quickly.

User: bro i dont get it lol
Assistant: Think of RAM like a desk. Stuff you're actively working with stays on the desk because it's quick to reach.

User: Could you explain that more formally?
Assistant: RAM temporarily stores data and instructions that active programs need to access quickly.
```

These are **examples of behavior**, not a list of every possible question.

Then run:

```powershell
py -3.12 finetune.py
```

This starts from the knowledge learned during pretraining and saves `checkpoints/chat_model.pt`.

## 5. Talk to it

```powershell
py -3.12 chat.py
```

Then type arbitrary input:

```text
You: bro why does my pc need ram if i already have an ssd lol
AI: ...
```

The tokenizer can represent arbitrary UTF-8 text. Whether the model gives a useful answer depends on what it managed to learn during training.

## Data pipeline

```text
WikiText / future corpora
        |
        v
prepare_data.py
        |
        +--> train BPE tokenizer from scratch
        |
        +--> data/train.bin
        +--> data/val.bin
                 |
                 v
              train.py
                 |
                 v
        checkpoints/model.pt
                 |
        conversational examples
                 |
                 v
            finetune.py
                 |
                 v
      checkpoints/chat_model.pt
                 |
                 v
              chat.py
```

## Hardware

Current target:

- NVIDIA GeForce RTX 4060 Ti
- 8 GB VRAM
- Windows
- Python 3.12
- CUDA-enabled PyTorch

The current model is intentionally small enough to experiment with on consumer hardware.

## Important reality check

This pipeline implements the same core idea used by GPT-style language models: learn next-token prediction from large amounts of text, then adapt the model for useful interaction.

Scale matters enormously. A small model trained on a 4060 Ti will not approach frontier systems such as ChatGPT. The purpose here is to build the real mechanism rather than fake intelligence with hardcoded responses.

## AI-assisted development

AI is used to explain concepts, debug code, and help implement this educational project. LocalAI itself runs from its own locally trained weights.
