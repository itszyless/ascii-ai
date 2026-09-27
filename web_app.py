"""Local web UI for chatting with LocalAI v3 (falls back to v2)."""

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
V3_CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v3.pt"
V3_BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v3_best.pt"
V3_FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v3.pt"
V3_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v3.json"

V2_CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v2.pt"
V2_BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v2_best.pt"
V2_FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v2.pt"
V2_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"

MAX_NEW_TOKENS = 180
TEMPERATURE = 0.35
TOP_K = 20
REPETITION_PENALTY = 1.08

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
    temp.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(CHAT_STORE)


def find_chat(store, chat_id):
    return next((c for c in store["chats"] if c["id"] == chat_id), None)


def choose_model():
    if V3_CHAT_MODEL_PATH.exists() and V3_TOKENIZER_PATH.exists():
        return V3_CHAT_MODEL_PATH, V3_TOKENIZER_PATH, "v3", True
    if V3_BASE_MODEL_PATH.exists() and V3_TOKENIZER_PATH.exists():
        return V3_BASE_MODEL_PATH, V3_TOKENIZER_PATH, "v3", False
    if V3_FALLBACK_BASE_PATH.exists() and V3_TOKENIZER_PATH.exists():
        return V3_FALLBACK_BASE_PATH, V3_TOKENIZER_PATH, "v3", False

    if V2_CHAT_MODEL_PATH.exists() and V2_TOKENIZER_PATH.exists():
        return V2_CHAT_MODEL_PATH, V2_TOKENIZER_PATH, "v2", True
    if V2_BASE_MODEL_PATH.exists() and V2_TOKENIZER_PATH.exists():
        return V2_BASE_MODEL_PATH, V2_TOKENIZER_PATH, "v2", False
    if V2_FALLBACK_BASE_PATH.exists() and V2_TOKENIZER_PATH.exists():
        return V2_FALLBACK_BASE_PATH, V2_TOKENIZER_PATH, "v2", False

    return None, None, None, False


def load_model():
    global _model, _tokenizer, _device, _model_path

    chosen, tokenizer_path, _, _ = choose_model()
    if chosen is None:
        raise FileNotFoundError("No compatible LocalAI checkpoint found.")

    if _model is not None and _model_path == chosen:
        return

    requested = os.getenv("LOCALAI_DEVICE", "").strip().lower()
    if requested == "cpu":
        device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LocalTokenizer.load(tokenizer_path)
    checkpoint = torch.load(chosen, map_location=device, weights_only=False)
    model = LocalAI(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model = model
    _tokenizer = tokenizer
    _device = device
    _model_path = chosen


def build_history_prompt(messages):
    current_user = messages[-1]["content"]
    current_ids = _tokenizer.encode(f"User: {current_user}\nAssistant:")

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
        ids = _tokenizer.encode(f"User: {user}\nAssistant: {assistant}\n")
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
    prompt_ids = build_history_prompt(messages)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=_device)

    with _generation_lock:
        generated = _model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
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
    path, tokenizer_path, version, chat_tuned = choose_model()
    return jsonify({
        "ready": path is not None and tokenizer_path is not None,
        "checkpoint": path.name if path else None,
        "chat_tuned": chat_tuned,
        "version": version,
        "device": str(_device) if _device is not None else "not loaded",
    })


@app.get("/api/chats")
def list_chats():
    with _store_lock:
        store = load_store()
        chats = [{
            "id": c["id"],
            "title": c["title"],
            "created_at": c["created_at"],
            "updated_at": c["updated_at"],
        } for c in store["chats"]]

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
            clean = content.replace("\n", " ")
            chat["title"] = clean[:42] + ("…" if len(clean) > 42 else "")

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
        "rating": None,
    }

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat was deleted"}), 409

        chat["messages"].append(assistant_message)
        chat["updated_at"] = now_iso()
        save_store(store)

    return jsonify({
        "message": assistant_message,
        "title": chat["title"],
        "checkpoint": _model_path.name if _model_path else None,
        "device": str(_device) if _device else None,
    })


@app.post("/api/chats/<chat_id>/regenerate")
def regenerate(chat_id):
    payload = request.get_json(silent=True) or {}
    message_id = payload.get("message_id")

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat not found"}), 404

        index = next(
            (
                i for i, message in enumerate(chat["messages"])
                if message.get("id") == message_id
            ),
            None,
        )

        if index is None:
            return jsonify({"error": "Response not found"}), 404
        if chat["messages"][index].get("role") != "assistant":
            return jsonify({"error": "Selected message is not an assistant response"}), 400
        if index == 0 or chat["messages"][index - 1].get("role") != "user":
            return jsonify({"error": "Missing original prompt"}), 400

        # Remove the selected answer and anything after it. The user prompt
        # immediately before it stays and is used again for generation.
        chat["messages"] = chat["messages"][:index]
        chat["updated_at"] = now_iso()
        save_store(store)
        snapshot = list(chat["messages"])

    try:
        reply = generate_reply(snapshot)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    message = {
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": reply,
        "created_at": now_iso(),
        "rating": None,
    }

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat was deleted"}), 409

        chat["messages"].append(message)
        chat["updated_at"] = now_iso()
        save_store(store)

    return jsonify({"message": message})


@app.post("/api/chats/<chat_id>/messages/<message_id>/rating")
def rate_message(chat_id, message_id):
    payload = request.get_json(silent=True) or {}
    rating = payload.get("rating")

    if rating not in ("up", "down", None):
        return jsonify({"error": "Invalid rating"}), 400

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat not found"}), 404

        message = next(
            (m for m in chat["messages"] if m.get("id") == message_id),
            None,
        )
        if message is None:
            return jsonify({"error": "Message not found"}), 404

        message["rating"] = rating
        save_store(store)

    return jsonify({"ok": True, "rating": rating})


if __name__ == "__main__":
    print("LocalAI web UI: http://127.0.0.1:5000")
    print("Chats stay local in data/web_chats.json")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
