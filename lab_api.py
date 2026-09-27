"""Developer Lab API for OrcaAI.

Everything stays local. The lab exposes model internals, tokenization,
generation controls, checkpoint info, lightweight memory/knowledge tools,
and evaluation helpers for the localhost UI.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import re
import time

from flask import jsonify, render_template, request
import torch


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
MEMORY_PATH = DATA_DIR / "lab_memories.json"
KNOWLEDGE_PATH = DATA_DIR / "lab_knowledge.json"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_json(path, default):
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
    except Exception:
        pass
    return default


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _words(text):
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _safe_checkpoint_meta(path):
    meta = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "megabytes": round(path.stat().st_size / 1024 / 1024, 2),
        "modified": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).isoformat(),
    }

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta.update({
            "stage": checkpoint.get("stage"),
            "step": checkpoint.get("step"),
            "best_val": checkpoint.get("best_val"),
            "val_loss": checkpoint.get("val_loss"),
            "train_loss": checkpoint.get("train_loss"),
            "config": checkpoint.get("config"),
        })
        del checkpoint
        gc.collect()
    except Exception as exc:
        meta["error"] = str(exc)

    return meta


def register_lab(app, runtime_provider, store_provider):
    @app.get("/lab")
    def lab_page():
        return render_template("lab.html")

    @app.get("/api/lab/model")
    def lab_model():
        runtime = runtime_provider(load=True)
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        device = runtime["device"]
        path = runtime["model_path"]

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        gpu = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "allocated_mb": round(
                    torch.cuda.memory_allocated() / 1024 / 1024, 1
                ),
                "reserved_mb": round(
                    torch.cuda.memory_reserved() / 1024 / 1024, 1
                ),
                "free_mb": round(free / 1024 / 1024, 1),
                "total_mb": round(total / 1024 / 1024, 1),
            }

        result = {
            "checkpoint": path.name,
            "checkpoint_mb": round(path.stat().st_size / 1024 / 1024, 2),
            "version": runtime["version"],
            "chat_tuned": runtime["chat_tuned"],
            "device": str(device),
            "stage": checkpoint.get("stage"),
            "step": checkpoint.get("step"),
            "best_val": checkpoint.get("best_val"),
            "val_loss": checkpoint.get("val_loss"),
            "train_loss": checkpoint.get("train_loss"),
            "parameters": params,
            "trainable_parameters": trainable,
            "vocab_size": tokenizer.vocab_size,
            "context_length": model.context_length,
            "config": config,
            "gpu": gpu,
        }

        del checkpoint
        gc.collect()
        return jsonify(result)

    @app.post("/api/lab/tokenize")
    def lab_tokenize():
        runtime = runtime_provider(load=True)
        tokenizer = runtime["tokenizer"]
        text = str((request.get_json(silent=True) or {}).get("text") or "")

        ids = tokenizer.encode(text)
        tokens = []
        for index, token_id in enumerate(ids):
            raw = tokenizer.tokenizer.id_to_token(int(token_id))
            decoded = tokenizer.decode([token_id])
            tokens.append({
                "index": index,
                "id": int(token_id),
                "raw": raw,
                "decoded": decoded,
            })

        return jsonify({
            "text": text,
            "count": len(ids),
            "ids": ids,
            "tokens": tokens,
        })

    @app.post("/api/lab/next-token")
    def lab_next_token():
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt") or "")
        top_n = max(1, min(int(payload.get("top_n") or 12), 30))

        runtime = runtime_provider(load=True)
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        device = runtime["device"]

        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.bos_id]

        ids = ids[-model.context_length:]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.inference_mode():
            logits, _ = model(tokens)
            logits = logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k=min(top_n, probs.numel()))

        entropy = -torch.sum(
            probs * torch.log(probs.clamp_min(1e-12))
        ).item()

        items = []
        for probability, token_id in zip(values.tolist(), indices.tolist()):
            items.append({
                "id": int(token_id),
                "token": tokenizer.decode([token_id]),
                "raw": tokenizer.tokenizer.id_to_token(int(token_id)),
                "probability": probability,
                "percent": probability * 100,
            })

        return jsonify({
            "prompt_tokens": len(ids),
            "entropy": entropy,
            "predictions": items,
        })

    @app.post("/api/lab/generate")
    def lab_generate():
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt") or "").strip()
        mode = str(payload.get("mode") or "continuation")
        temperature = max(
            0.05, min(float(payload.get("temperature") or 0.55), 2.0)
        )
        top_k = max(1, min(int(payload.get("top_k") or 30), 200))
        max_tokens = max(
            1, min(int(payload.get("max_tokens") or 100), 400)
        )
        repetition_penalty = max(
            1.0,
            min(float(payload.get("repetition_penalty") or 1.05), 2.0),
        )
        seed = int(payload.get("seed") or 1337)

        runtime = runtime_provider(load=True)
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        device = runtime["device"]

        formatted = prompt
        if mode == "chat":
            formatted = f"User: {prompt}\nAssistant:"

        ids = tokenizer.encode(formatted)
        if not ids:
            ids = [tokenizer.bos_id]
        ids = ids[-model.context_length:]

        input_tensor = torch.tensor(
            [ids], dtype=torch.long, device=device
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            generated = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_token_ids={tokenizer.eos_id},
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1e-6)

        new_ids = generated[0, len(ids):].tolist()
        output = tokenizer.decode(new_ids)

        if mode == "chat":
            for marker in ("\nUser:", "\nAssistant:"):
                if marker in output:
                    output = output.split(marker, 1)[0]

        return jsonify({
            "output": output.strip(),
            "input_tokens": len(ids),
            "output_tokens": len(new_ids),
            "latency_ms": round(elapsed * 1000, 2),
            "tokens_per_second": round(len(new_ids) / elapsed, 2),
            "settings": {
                "mode": mode,
                "temperature": temperature,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "repetition_penalty": repetition_penalty,
                "seed": seed,
            },
        })

    @app.get("/api/lab/checkpoints")
    def lab_checkpoints():
        active = runtime_provider(load=False).get("model_path")
        items = []

        for path in sorted(
            CHECKPOINT_DIR.glob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            meta = _safe_checkpoint_meta(path)
            meta["active"] = bool(active and path == active)
            items.append(meta)

        return jsonify(items)

    @app.get("/api/lab/memories")
    def lab_memories_get():
        return jsonify(_load_json(MEMORY_PATH, []))

    @app.post("/api/lab/memories")
    def lab_memories_add():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Memory cannot be empty"}), 400

        items = _load_json(MEMORY_PATH, [])
        item = {
            "id": f"mem-{int(time.time() * 1000)}",
            "text": text[:2000],
            "created_at": _now(),
        }
        items.insert(0, item)
        _save_json(MEMORY_PATH, items)
        return jsonify(item), 201

    @app.delete("/api/lab/memories/<memory_id>")
    def lab_memories_delete(memory_id):
        items = _load_json(MEMORY_PATH, [])
        new_items = [x for x in items if x.get("id") != memory_id]
        _save_json(MEMORY_PATH, new_items)
        return jsonify({"ok": True})

    @app.get("/api/lab/knowledge")
    def lab_knowledge_get():
        return jsonify(_load_json(KNOWLEDGE_PATH, []))

    @app.post("/api/lab/knowledge")
    def lab_knowledge_add():
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title") or "Untitled").strip()[:120]
        content = str(payload.get("content") or "").strip()

        if not content:
            return jsonify({"error": "Knowledge content is empty"}), 400

        items = _load_json(KNOWLEDGE_PATH, [])
        item = {
            "id": f"doc-{int(time.time() * 1000)}",
            "title": title or "Untitled",
            "content": content[:100_000],
            "created_at": _now(),
        }
        items.insert(0, item)
        _save_json(KNOWLEDGE_PATH, items)
        return jsonify(item), 201

    @app.delete("/api/lab/knowledge/<doc_id>")
    def lab_knowledge_delete(doc_id):
        items = _load_json(KNOWLEDGE_PATH, [])
        new_items = [x for x in items if x.get("id") != doc_id]
        _save_json(KNOWLEDGE_PATH, new_items)
        return jsonify({"ok": True})

    @app.post("/api/lab/knowledge/search")
    def lab_knowledge_search():
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "")
        query_words = _words(query)

        scored = []
        for item in _load_json(KNOWLEDGE_PATH, []):
            doc_words = _words(
                f"{item.get('title', '')} {item.get('content', '')}"
            )
            overlap = len(query_words & doc_words)
            if overlap:
                score = overlap / max(math.sqrt(len(doc_words)), 1.0)
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, item in scored[:8]:
            results.append({
                "id": item["id"],
                "title": item["title"],
                "score": round(score, 4),
                "preview": item["content"][:600],
            })

        return jsonify(results)

    @app.get("/api/lab/stats")
    def lab_stats():
        store = store_provider()
        chats = store.get("chats", [])
        messages = [
            message
            for chat in chats
            for message in chat.get("messages", [])
        ]

        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [
            m for m in messages if m.get("role") == "assistant"
        ]

        ratings = {
            "up": sum(m.get("rating") == "up" for m in assistant_messages),
            "down": sum(
                m.get("rating") == "down" for m in assistant_messages
            ),
        }

        return jsonify({
            "chats": len(chats),
            "messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "ratings": ratings,
            "memories": len(_load_json(MEMORY_PATH, [])),
            "knowledge_documents": len(
                _load_json(KNOWLEDGE_PATH, [])
            ),
        })

    @app.post("/api/lab/evaluate")
    def lab_evaluate():
        payload = request.get_json(silent=True) or {}
        prompts = payload.get("prompts") or [
            "RAM is a type of",
            "The CPU is responsible for",
            "A computer uses memory to",
            "A function in programming is",
            "Paris is the capital of",
            "The opposite of hot is",
        ]

        prompts = [
            str(prompt)[:500]
            for prompt in prompts[:12]
            if str(prompt).strip()
        ]

        runtime = runtime_provider(load=True)
        model = runtime["model"]
        tokenizer = runtime["tokenizer"]
        device = runtime["device"]

        results = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            if not ids:
                ids = [tokenizer.bos_id]
            ids = ids[-model.context_length:]

            input_tensor = torch.tensor(
                [ids],
                dtype=torch.long,
                device=device,
            )

            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=50,
                    temperature=0.45,
                    top_k=25,
                    repetition_penalty=1.05,
                    stop_token_ids={tokenizer.eos_id},
                )

            elapsed = max(time.perf_counter() - started, 1e-6)
            new_ids = generated[0, len(ids):].tolist()
            output = tokenizer.decode(new_ids).strip()

            results.append({
                "prompt": prompt,
                "output": output,
                "tokens": len(new_ids),
                "latency_ms": round(elapsed * 1000, 1),
            })

        return jsonify({
            "checkpoint": runtime["model_path"].name,
            "results": results,
        })
