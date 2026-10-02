"""Local web UI for OrcaAI. Prefers v4 and falls back to older checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import tempfile
import uuid

from flask import Flask, jsonify, render_template, request
import torch

from model import LocalAI
from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer
from lab_api import register_lab
from quality_layer import retrieve_context, response_quality


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHAT_STORE = DATA_DIR / "web_chats.json"
CHECKPOINT_DIR = ROOT / "checkpoints"
V4_CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v4.pt"
V4_GROUNDED_MODEL_PATH = CHECKPOINT_DIR / "model_v4_grounded.pt"
V4_BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v4_best.pt"
V4_FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v4.pt"
V4_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"

V3_CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v3.pt"
V3_BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v3_best.pt"
V3_FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v3.pt"
V3_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v3.json"

V2_CHAT_MODEL_PATH = CHECKPOINT_DIR / "chat_model_v2.pt"
V2_BASE_MODEL_PATH = CHECKPOINT_DIR / "model_v2_best.pt"
V2_FALLBACK_BASE_PATH = CHECKPOINT_DIR / "model_v2.pt"
V2_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.20
TOP_K = 1
TOP_P = 1.0
REPETITION_PENALTY = 1.08
QUALITY_CANDIDATES = 1

# Keep the terminal clean: Flask/Werkzeug normally prints one line for every
# browser request. Errors still surface through the app/API itself.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__, template_folder="templates", static_folder="static")

_model = None
_tokenizer = None
_device = None
_model_path = None
_model_signature = None
_generation_lock = threading.Lock()
_store_lock = threading.Lock()
_whisper_model = None
_whisper_lock = threading.Lock()


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


def checkpoint_signature(path):
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def choose_model():
    if V4_CHAT_MODEL_PATH.exists() and V4_TOKENIZER_PATH.exists():
        return V4_CHAT_MODEL_PATH, V4_TOKENIZER_PATH, "v4", True
    if V4_GROUNDED_MODEL_PATH.exists() and V4_TOKENIZER_PATH.exists():
        return V4_GROUNDED_MODEL_PATH, V4_TOKENIZER_PATH, "v4", False
    if V4_BASE_MODEL_PATH.exists() and V4_TOKENIZER_PATH.exists():
        return V4_BASE_MODEL_PATH, V4_TOKENIZER_PATH, "v4", False
    if V4_FALLBACK_BASE_PATH.exists() and V4_TOKENIZER_PATH.exists():
        return V4_FALLBACK_BASE_PATH, V4_TOKENIZER_PATH, "v4", False

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
    global _model, _tokenizer, _device, _model_path, _model_signature

    chosen, tokenizer_path, version, _ = choose_model()
    if chosen is None:
        raise FileNotFoundError("No compatible OrcaAI checkpoint found.")

    signature = checkpoint_signature(chosen)
    if (
        _model is not None
        and _model_path == chosen
        and _model_signature == signature
    ):
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
    model_class = OrcaAIv4 if version == "v4" else LocalAI
    model = model_class(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _model = model
    _tokenizer = tokenizer
    _device = device
    _model_path = chosen
    _model_signature = signature


def should_use_history(user_prompt):
    text = " ".join(str(user_prompt).lower().split())

    # Clear standalone questions/commands should not be contaminated by older
    # turns. The small model handles them much better as single-turn prompts.
    standalone_markers = (
        "who are you",
        "what are you",
        "your name",
        "how are you",
        "what is ",
        "what's ",
        "whats ",
        "who is ",
        "where is ",
        "when is ",
        "why is ",
        "how does ",
        "how do ",
        "say ",
        "just say",
        "tell me ",
        "answer this",
        "calculate",
    )
    if any(marker in text for marker in standalone_markers):
        return False

    # Very short follow-ups and references normally depend on the previous turn.
    followup_markers = (
        "really",
        "are you sure",
        "go deeper",
        "explain it",
        "explain that",
        "what do you mean",
        "why?",
        "how so",
        "that",
        "it",
        "this",
        "i never said",
    )
    if len(text.split()) <= 8 and any(marker in text for marker in followup_markers):
        return True

    return False


def build_history_prompt(messages):
    current_user = str(messages[-1].get("content") or "").strip()

    if not should_use_history(current_user):
        prompt_ids = _tokenizer.encode(
            f"User: {current_user}\nAssistant:"
        )
        return prompt_ids[-_model.context_length:]

    # Match the exact multi-turn format used during fine-tuning:
    # User prompt -> Assistant response -> EOS -> next User prompt.
    recent = messages[-5:]
    token_ids = []
    pending_user = None

    for message in recent[:-1]:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            token_ids.extend(
                _tokenizer.encode(
                    f"User: {pending_user}\nAssistant:"
                )
            )
            token_ids.extend(
                _tokenizer.encode(
                    f" {content}",
                    add_eos=True,
                )
            )
            pending_user = None

    token_ids.extend(
        _tokenizer.encode(
            f"User: {current_user}\nAssistant:"
        )
    )

    return token_ids[-_model.context_length:]


def polish_reply(response):
    response = response.strip()
    if not response:
        return response

    replacements = {
        "orcaai": "OrcaAI",
        "ram": "RAM",
        "cpu": "CPU",
        "gpu": "GPU",
        "ssd": "SSD",
        "html": "HTML",
        "css": "CSS",
        "h2o": "H2O",
    }

    words = response.split()
    for index, word in enumerate(words):
        prefix = ""
        suffix = ""
        core = word

        while core and not core[0].isalnum():
            prefix += core[0]
            core = core[1:]
        while core and not core[-1].isalnum():
            suffix = core[-1] + suffix
            core = core[:-1]

        replacement = replacements.get(core.lower())
        if replacement:
            words[index] = prefix + replacement + suffix

    response = " ".join(words)

    for index, char in enumerate(response):
        if char.isalpha():
            response = response[:index] + char.upper() + response[index + 1:]
            break

    return response


def _decode_candidate(generated, prompt_length):
    new_tokens = generated[0, prompt_length:].tolist()
    response = _tokenizer.decode(new_tokens)

    for marker in ("\nUser:", "\nAssistant:"):
        if marker in response:
            response = response.split(marker, 1)[0]

    return polish_reply(response)


def generate_reply(messages, effort="instant"):
    load_model()
    user_prompt = messages[-1]["content"]
    _, _, _, chat_tuned = choose_model()

    # A raw pretrained checkpoint has never learned the User:/Assistant:
    # conversation format. Feeding that syntax into a base model makes its
    # outputs look much worse than the model actually is, so base-mode testing
    # uses plain continuation instead.
    if chat_tuned:
        prompt_ids = build_history_prompt(messages)
    else:
        prompt_ids = _tokenizer.encode(user_prompt)
        prompt_ids = prompt_ids[-_model.context_length:]

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=_device)

    effort = str(effort or "instant").lower()
    effort_settings = {
        "instant": [
            (0.20, 1, 1.0, 100),
        ],
        "medium": [
            (0.20, 1, 1.0, 110),
            (0.32, 16, 0.90, 110),
            (0.42, 24, 0.92, 110),
        ],
        "high": [
            (0.20, 1, 1.0, 130),
            (0.28, 12, 0.88, 130),
            (0.36, 20, 0.90, 130),
            (0.44, 28, 0.92, 130),
            (0.52, 36, 0.94, 130),
        ],
    }
    settings = effort_settings.get(effort, effort_settings["medium"])

    # Medium and High spend more inference compute generating alternatives and
    # keep the strongest candidate. This is local inference-time search.
    candidates = []

    with _generation_lock:
        for temperature, top_k, top_p, max_new_tokens in settings:
            generated = _model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids={_tokenizer.eos_id},
            )
            candidates.append(
                _decode_candidate(generated, len(prompt_ids))
            )

    best = max(
        candidates,
        key=lambda text: response_quality(text, user_prompt),
        default="",
    )

    return best or "..."


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
    title = str(payload.get("title") or "New chat").strip()[:15] or "New chat"
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


@app.patch("/api/chats/<chat_id>")
def rename_chat(chat_id):
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title") or "").strip()[:15]

    if not title:
        return jsonify({"error": "Chat name cannot be empty"}), 400

    with _store_lock:
        store = load_store()
        chat = find_chat(store, chat_id)
        if chat is None:
            return jsonify({"error": "Chat not found"}), 404

        chat["title"] = title
        chat["updated_at"] = now_iso()
        save_store(store)

    return jsonify({"ok": True, "title": title})


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
    effort = str(payload.get("effort") or "medium").lower()
    if effort not in {"instant", "medium", "high"}:
        effort = "medium"

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
            chat["title"] = clean[:15] + ("…" if len(clean) > 15 else "")

        chat["updated_at"] = now_iso()
        save_store(store)
        snapshot = list(chat["messages"])

    try:
        reply = generate_reply(snapshot, effort=effort)
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
    effort = str(payload.get("effort") or "medium").lower()
    if effort not in {"instant", "medium", "high"}:
        effort = "medium"

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
        reply = generate_reply(snapshot, effort=effort)
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


def get_whisper_model():
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    with _whisper_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Local voice transcription needs faster-whisper. "
                    "Run: py -3.12 -m pip install faster-whisper"
                ) from exc

            model_name = os.getenv("ORCA_WHISPER_MODEL", "small")
            _whisper_model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
            )

    return _whisper_model


@app.post("/api/transcribe")
def transcribe_audio():
    audio = request.files.get("audio")
    if audio is None or not audio.filename:
        return jsonify({"error": "No audio recording received"}), 400

    if request.content_length and request.content_length > 25 * 1024 * 1024:
        return jsonify({"error": "Recording is too large"}), 413

    suffix = Path(audio.filename).suffix or ".webm"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="orca_voice_",
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            audio.save(temp_file)

        model = get_whisper_model()
        with _whisper_lock:
            segments, info = model.transcribe(
                str(temp_path),
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

        return jsonify({
            "text": text,
            "language": getattr(info, "language", None),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


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


def lab_runtime(load=False):
    if load:
        load_model()

    path, tokenizer_path, version, chat_tuned = choose_model()
    return {
        "model": _model,
        "tokenizer": _tokenizer,
        "device": _device,
        "model_path": _model_path or path,
        "tokenizer_path": tokenizer_path,
        "version": version,
        "chat_tuned": chat_tuned,
    }


def lab_store_snapshot():
    with _store_lock:
        return load_store()


register_lab(app, lab_runtime, lab_store_snapshot)


if __name__ == "__main__":
    print("OrcaAI web UI: http://127.0.0.1:5000")
    print("Chats stay local in data/web_chats.json")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
