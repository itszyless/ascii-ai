"""Local web UI for chatting with LocalAI v2.

Run after training/fine-tuning:
    py -3.12 web_app.py
Then open http://127.0.0.1:5000

Chats are stored locally in data/web_chats.json.
No messages leave this computer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import uuid

from flask import Flask, jsonify, render_template, request
import torch

from model import LocalAI
from tokenizer import LocalTokenizer


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHAT_STORE = DATA_DIR / "web_chats.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v2.pt"
BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v2_best.pt"
FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v2.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"

MAX_NEW_TOKENS = 180
TEMPERATURE = 0.65
TOP_K = 40

app = Flask(__name__, template_folder="templates", static_folder="static")

_model = None
_tokenizer = None
_device = None
_model_path = None
_generation_lock = threading.Lock()
_store_lock = threading.Lock()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_store():
    DATA_DIR.mkdir(exist_ok=True)
    if not CHAT_STORE.exists():
        return {"chats": []}
    try:
        data = json.loads(CHAT_STORE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("chats"), list):
            raise ValueError
        return data
    except (json.JSONDecodeError, ValueError):
        return {"chats": []}


def save_store(store):
    DATA_DIR.mkdir(exist_ok=True)
    temp = CHAT_STORE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(CHAT_STORE)


def find_chat(store, chat_id):
    return next((c for c in store["chats"] if c["id"] == chat_id), None)


def choose_model_path():
    if CHAT_MODEL_PATH.exists():
        return CHAT_MODEL_PATH
    if BASE_MODEL_PATH.exists():
        return BASE_MODEL_PATH
    if FALLBACK_BASE_PATH.exists():
        return FALLBACK_BASE_PATH
    return None


def load_model():
    global _model, _tokenizer, _device, _model_path

    chosen = choose_model_path()
    if chosen is None:
        raise FileNotFoundError(
            "No v2 checkpoint found. Finish train.py, then run finetune.py."
        )
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Missing checkpoints/tokenizer_v2.json")

    if _model is not None and _model_path == chosen:
        return

    requested_device = os.getenv("LOCALAI_DEVICE", "").strip().lower()
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("LOCALAI_DEVICE=cuda but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(chosen, map_location=device, weights_only=False)
    model = LocalAI(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model = model
    _tokenizer = tokenizer
    _device = device
    _model_path = chosen


def build_history_prompt(messages):
    """Fit as many recent complete user/assistant turns as possible."""
    current_user = messages[-1]["content"]
    current_text = f"User: {current_user}\nAssistant:"
    current_ids = _tokenizer.encode(current_text)

    if len(current_ids) >= _model.context_length:
        return current_ids[-_model.context_length:]

    completed = []
    pending_user = None

    for message in messages[:-1]:
        if message["role"] == "user":
            pending_user = message["content"]
        elif message["role"] == "assistant" and pending_user is not None:
            completed.append((pending_user, message["content"]))
            pending_user = None

    selected = []
    used = len(current_ids)

    for user, assistant in reversed(completed):
        ids = _tokenizer.encode(
            f"User: {user}\nAssistant: {assistant}\n"
        )
        if used + len(ids) > _model.context_length:
            break
        selected.append(ids)
        used += len(ids)

    prompt_ids = []
    for ids in reversed(selected):
        prompt_ids.extend(ids)
    prompt_ids.extend(current_ids)
    return prompt_ids


def generate_reply(messages):
    load_model()

    if _model_path != CHAT_MODEL_PATH:
        # It still works as a raw LM, but the user should know chat tuning is absent.
        pass

    prompt_ids = build_history_prompt(messages)
    tokens = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=_device,
    )

    with _generation_lock:
        generated = _model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            stop_token_ids={_tokenizer.eos_id},
        )

    new_tokens = generated[0, len(prompt_ids):].tolist()
    response = _tokenizer.decode(new_tokens)

    for marker in ("\nUser:", "\nAssistant:"):
        if marker in response:
            response = response.split(marker, 1)[0]

    return response.strip() or "..."


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    path = choose_model_path()
    return jsonify(
        {
            "ready": path is not None and TOKENIZER_PATH.exists(),
            "checkpoint": path.name if path else None,
            "chat_tuned": path == CHAT_MODEL_PATH,
            "device": str(_device) if _device is not None else "not loaded",
        }
    )


@app.get("/api/chats")
def list_chats():
    with _store_lock:
        store = load_store()
        chats = [
            {
                "id": c["id"],
                "title": c["title"],
                "created_at": c["created_at"],
                "updated_at": c["updated_at"],
            }
            for c in store["chats"]
        ]

    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return jsonify(chats)


@app.post("/api/chats")
def create_chat():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title") or "New chat").strip()[:60] or "New chat"
    stamp = now_iso()

    chat = {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": stamp,
        "updated_at": stamp,
        "messages": [],
    }

    with _store_lock:
        store = load_store()
        store["chats"].append(chat)
        save_store(store)

    return jsonify(chat), 201


@app.get("/api/chats/<chat_id>")
def get_chat(chat_id):
    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)

    if chat is None:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat)


@app.delete("/api/chats/<chat_id>")
def delete_chat(chat_id):
    with _store_lock:
        store = load_store()
        before = len(store["chats"])
        store["chats"] = [c for c in store["chats"] if c["id"] != chat_id]
        if len(store["chats"]) == before:
            return jsonify({"error": "Chat not found"}), 404
        save_store(store)

    return jsonify({"ok": True})


@app.post("/api/chats/<chat_id>/messages")
def send_message(chat_id):
    payload = request.get_json(silent=True) or {}
    content = str(payload.get("message") or "").strip()

    if not content:
        return jsonify({"error": "Message cannot be empty"}), 400
    if len(content) > 8000:
        return jsonify({"error": "Message is too long"}), 400

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat not found"}), 404

        user_message = {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": content,
            "created_at": now_iso(),
        }
        chat["messages"].append(user_message)

        if chat["title"] == "New chat":
            chat["title"] = content.replace("\n", " ")[:42]
            if len(content.replace("\n", " ")) > 42:
                chat["title"] += "…"

        chat["updated_at"] = now_iso()
        save_store(store)
        snapshot = list(chat["messages"])

    try:
        reply = generate_reply(snapshot)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    assistant_message = {
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": reply,
        "created_at": now_iso(),
    }

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat was deleted"}), 409
        chat["messages"].append(assistant_message)
        chat["updated_at"] = now_iso()
        save_store(store)

    return jsonify(
        {
            "message": assistant_message,
            "title": chat["title"],
            "checkpoint": _model_path.name if _model_path else None,
            "device": str(_device) if _device else None,
        }
    )


if __name__ == "__main__":
    print("LocalAI web UI: http://127.0.0.1:5000")
    print("Chats stay local in data/web_chats.json")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
