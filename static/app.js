const state = {
  chats: [],
  activeChatId: null,
  sending: false,
  effort: localStorage.getItem("orcaThinkingEffort") || "medium",
  recording: false,
  transcribing: false,
};

const chatList = document.getElementById("chatList");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const composer = document.getElementById("composer");
const input = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChat");
const statusDot = document.getElementById("statusDot");
const statusTitle = document.getElementById("statusTitle");
const statusSub = document.getElementById("statusSub");
const sidebar = document.getElementById("sidebar");
const toggleSidebar = document.getElementById("toggleSidebar");
const desktopSidebarOpen = document.getElementById("desktopSidebarOpen");
const renameModal = document.getElementById("renameModal");
const renameInput = document.getElementById("renameInput");
const renameSave = document.getElementById("renameSave");
const renameCancel = document.getElementById("renameCancel");
const deleteModal = document.getElementById("deleteModal");
const deleteConfirm = document.getElementById("deleteConfirm");
const deleteCancel = document.getElementById("deleteCancel");
const effortPicker = document.getElementById("effortPicker");
const effortButton = document.getElementById("effortButton");
const effortLabel = document.getElementById("effortLabel");
const effortPopover = document.getElementById("effortPopover");
const effortHint = document.getElementById("effortHint");
const micBtn = document.getElementById("micBtn");
const voicePanel = document.getElementById("voicePanel");
const voiceWave = document.getElementById("voiceWave");
const voiceCancel = document.getElementById("voiceCancel");
const voiceStop = document.getElementById("voiceStop");
const voiceSave = document.getElementById("voiceSave");
let renameChatId = null;
let deleteChatId = null;

let voiceStream = null;
let mediaRecorder = null;
let voiceChunks = [];
let voiceAudioContext = null;
let voiceAnalyser = null;
let voiceAnimationFrame = null;
let voiceInsertStart = 0;
let voiceInsertEnd = 0;

const icons = {
  copy: '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"/></svg>',
  check: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
  up: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 10v10H4V10h3Zm3 10h7.2a2 2 0 0 0 1.9-1.37l2-6A2 2 0 0 0 19.2 10H15l.7-3.5A2.1 2.1 0 0 0 13.64 4H13l-4 6v10Z"/></svg>',
  down: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 14V4H4v10h3Zm3-10h7.2a2 2 0 0 1 1.9 1.37l2 6A2 2 0 0 1 19.2 14H15l.7 3.5A2.1 2.1 0 0 1 13.64 20H13l-4-6V4Z"/></svg>',
  share: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 16V4m0 0 4 4m-4-4L8 8"/><path d="M5 12v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6"/></svg>',
  download: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 4v11m0 0 4-4m-4 4-4-4"/><path d="M5 19h14"/></svg>',
  speak: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 10v4h4l5 4V6l-5 4H4Z"/><path d="M16 9.5a3.5 3.5 0 0 1 0 5"/><path d="M18.5 7a7 7 0 0 1 0 10"/></svg>',
  regen: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M20 6v5h-5"/><path d="M19 11a7 7 0 1 0 .2 5"/></svg>'
};

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}


const effortCopy = {
  instant: { label: "Instant", hint: "Fastest response" },
  medium: { label: "Medium", hint: "Balanced speed and quality" },
  high: { label: "High", hint: "More local inference for the best candidate" },
};

function setEffort(level) {
  if (!effortCopy[level]) level = "medium";
  state.effort = level;
  localStorage.setItem("orcaThinkingEffort", level);
  effortPicker.dataset.effort = level;
  effortLabel.textContent = effortCopy[level].label;
  effortHint.textContent = effortCopy[level].hint;

  for (const option of effortPopover.querySelectorAll("[data-effort-option]")) {
    const selected = option.dataset.effortOption === level;
    option.classList.toggle("selected", selected);
    option.setAttribute("aria-checked", selected ? "true" : "false");
  }
}

function setEffortOpen(open) {
  effortPopover.hidden = !open;
  effortButton.setAttribute("aria-expanded", open ? "true" : "false");
  effortPicker.classList.toggle("open", open);
}

effortButton.addEventListener("click", event => {
  event.stopPropagation();
  setEffortOpen(effortPopover.hidden);
});

for (const option of effortPopover.querySelectorAll("[data-effort-option]")) {
  option.addEventListener("click", () => {
    setEffort(option.dataset.effortOption);
    window.setTimeout(() => setEffortOpen(false), 120);
  });
}

document.addEventListener("click", event => {
  if (!effortPicker.contains(event.target)) setEffortOpen(false);
});

function setVoiceUi(active, processing = false) {
  composer.classList.toggle("voice-active", active);
  composer.classList.toggle("voice-processing", processing);
  voicePanel.hidden = !active;
  state.recording = active && !processing;
  state.transcribing = processing;

  micBtn.classList.toggle("recording", active);
  micBtn.innerHTML = active
    ? '<span class="mic-stop-square" aria-hidden="true"></span>'
    : '<svg class="mic-icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="12" rx="3"/><path d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6"/></svg>';

  voiceCancel.disabled = processing;
  voiceStop.disabled = processing;
  voiceSave.disabled = processing;
  voiceSave.classList.toggle("loading", processing);
}

function stopVoiceMeter() {
  if (voiceAnimationFrame) cancelAnimationFrame(voiceAnimationFrame);
  voiceAnimationFrame = null;

  if (voiceAudioContext) {
    voiceAudioContext.close().catch(() => {});
    voiceAudioContext = null;
  }
  voiceAnalyser = null;

  for (const bar of voiceWave.children) {
    bar.style.transform = "scaleY(.18)";
  }
}

function startVoiceMeter(stream) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;

  voiceAudioContext = new AudioCtx();
  voiceAnalyser = voiceAudioContext.createAnalyser();
  voiceAnalyser.fftSize = 256;
  voiceAnalyser.smoothingTimeConstant = 0.74;

  const source = voiceAudioContext.createMediaStreamSource(stream);
  source.connect(voiceAnalyser);

  const data = new Uint8Array(voiceAnalyser.frequencyBinCount);
  const bars = [...voiceWave.children];

  function draw() {
    if (!voiceAnalyser || !state.recording) return;
    voiceAnalyser.getByteFrequencyData(data);

    const bucket = Math.max(1, Math.floor(data.length / bars.length));
    bars.forEach((bar, index) => {
      let total = 0;
      const start = index * bucket;
      const end = Math.min(data.length, start + bucket);
      for (let i = start; i < end; i += 1) total += data[i];
      const average = total / Math.max(1, end - start);
      const level = Math.max(.18, Math.min(1, average / 110));
      bar.style.transform = `scaleY(${level})`;
    });

    voiceAnimationFrame = requestAnimationFrame(draw);
  }

  draw();
}

function cleanupVoiceStream() {
  stopVoiceMeter();
  if (voiceStream) {
    for (const track of voiceStream.getTracks()) track.stop();
    voiceStream = null;
  }
  mediaRecorder = null;
}

async function startVoiceRecording() {
  if (state.recording || state.transcribing) return;

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    alert("Voice recording is not supported by this browser.");
    return;
  }

  voiceInsertStart = input.selectionStart ?? input.value.length;
  voiceInsertEnd = input.selectionEnd ?? voiceInsertStart;
  voiceChunks = [];

  try {
    voiceStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const preferred = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
    ].find(type => MediaRecorder.isTypeSupported(type));

    mediaRecorder = preferred
      ? new MediaRecorder(voiceStream, { mimeType: preferred })
      : new MediaRecorder(voiceStream);

    mediaRecorder.addEventListener("dataavailable", event => {
      if (event.data?.size) voiceChunks.push(event.data);
    });

    mediaRecorder.start(120);
    setVoiceUi(true, false);
    startVoiceMeter(voiceStream);
  } catch (error) {
    cleanupVoiceStream();
    setVoiceUi(false, false);
    alert(error?.message || "Microphone access failed.");
  }
}

function discardVoiceRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.ondataavailable = null;
    mediaRecorder.stop();
  }
  voiceChunks = [];
  cleanupVoiceStream();
  setVoiceUi(false, false);
  input.focus();
  input.setSelectionRange(voiceInsertStart, voiceInsertEnd);
}

function stopRecorderToBlob() {
  return new Promise(resolve => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      const type = voiceChunks[0]?.type || "audio/webm";
      resolve(new Blob(voiceChunks, { type }));
      return;
    }

    mediaRecorder.addEventListener("stop", () => {
      const type = mediaRecorder?.mimeType || voiceChunks[0]?.type || "audio/webm";
      resolve(new Blob(voiceChunks, { type }));
    }, { once: true });

    mediaRecorder.stop();
  });
}

function insertTranscript(text) {
  if (!text) return;
  const value = input.value;
  const start = Math.min(voiceInsertStart, value.length);
  const end = Math.min(Math.max(voiceInsertEnd, start), value.length);

  const before = value.slice(0, start);
  const after = value.slice(end);
  const needsLeadingSpace = before && !/\s$/.test(before) && !/^[,.;!?]/.test(text);
  const needsTrailingSpace = after && !/^\s/.test(after) && !/[\s([{]$/.test(text);

  const inserted =
    (needsLeadingSpace ? " " : "") +
    text +
    (needsTrailingSpace ? " " : "");

  input.value = before + inserted + after;
  const caret = before.length + inserted.length;
  input.focus();
  input.setSelectionRange(caret, caret);
  resizeInput();
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

async function finishVoiceRecording() {
  if ((!state.recording && !mediaRecorder) || state.transcribing) return;

  setVoiceUi(true, true);
  stopVoiceMeter();

  try {
    const blob = await stopRecorderToBlob();
    cleanupVoiceStream();

    if (!blob.size) throw new Error("No audio was recorded.");

    const extension = blob.type.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("audio", blob, `recording.${extension}`);

    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: form,
    });
    const result = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(result.error || "Voice transcription failed.");
    }

    insertTranscript((result.text || "").trim());
    voiceChunks = [];
    setVoiceUi(false, false);
  } catch (error) {
    setVoiceUi(false, false);
    alert(error?.message || "Voice transcription failed.");
  } finally {
    cleanupVoiceStream();
    input.focus();
  }
}

micBtn.addEventListener("click", () => {
  if (state.recording) {
    finishVoiceRecording();
  } else if (!state.transcribing) {
    startVoiceRecording();
  }
});
voiceCancel.addEventListener("click", discardVoiceRecording);
voiceStop.addEventListener("click", finishVoiceRecording);
voiceSave.addEventListener("click", finishVoiceRecording);

setEffort(state.effort);

function renderSidebar() {
  chatList.innerHTML = "";

  if (!state.chats.length) {
    chatList.innerHTML = '<div class="empty-chats">No chats yet</div>';
    return;
  }

  for (const chat of state.chats) {
    const row = document.createElement("div");
    row.className = "chat-item" + (chat.id === state.activeChatId ? " active" : "");

    const open = document.createElement("button");
    open.className = "chat-open";
    open.textContent = chat.title;
    open.onclick = () => openChat(chat.id);

    const edit = document.createElement("button");
    edit.className = "chat-edit";
    edit.title = "Rename chat";
    edit.setAttribute("aria-label", "Rename chat");
    edit.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4"/></svg>';
    edit.onclick = event => {
      event.stopPropagation();
      openRenameModal(chat);
    };

    const del = document.createElement("button");
    del.className = "chat-delete";
    del.title = "Delete chat";
    del.setAttribute("aria-label", "Delete chat");
    del.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></svg>';
    del.onclick = event => {
      event.stopPropagation();
      openDeleteModal(chat.id);
    };

    row.append(open, edit, del);
    chatList.appendChild(row);
  }
}

function actionButton(icon, title, fn) {
  const button = document.createElement("button");
  button.className = "response-action";
  button.type = "button";
  button.title = title;
  button.setAttribute("aria-label", title);
  button.innerHTML = icons[icon];
  button.onclick = fn;
  return button;
}

function addActions(contentWrap, message, messageRow) {
  const bar = document.createElement("div");
  bar.className = "response-actions";

  const copy = actionButton("copy", "Copy", async () => {
    await navigator.clipboard.writeText(message.content);
    copy.innerHTML = icons.check;
    copy.title = "Copied";
    copy.setAttribute("aria-label", "Copied");
    setTimeout(() => {
      copy.innerHTML = icons.copy;
      copy.title = "Copy";
      copy.setAttribute("aria-label", "Copy");
    }, 900);
  });

  const up = actionButton("up", "Good response", async () => {
    const next = up.classList.contains("selected") ? null : "up";
    await rateMessage(message.id, next);
    up.classList.toggle("selected", next === "up");
    down.classList.remove("selected");
  });

  const down = actionButton("down", "Bad response", async () => {
    const next = down.classList.contains("selected") ? null : "down";
    await rateMessage(message.id, next);
    down.classList.toggle("selected", next === "down");
    up.classList.remove("selected");
  });

  const share = actionButton("share", "Share", async () => {
    if (navigator.share) {
      try {
        await navigator.share({ text: message.content });
      } catch {}
    } else {
      await navigator.clipboard.writeText(message.content);
    }
  });

  const download = actionButton("download", "Download", () => {
    const blob = new Blob([message.content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "orcaai-response.txt";
    a.click();
    URL.revokeObjectURL(url);
  });

  const speak = actionButton("speak", "Read aloud", () => {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(message.content));
  });

  const regen = actionButton("regen", "Regenerate response", async () => {
    await regenerateResponse(message.id, messageRow);
  });

  if (message.rating === "up") up.classList.add("selected");
  if (message.rating === "down") down.classList.add("selected");

  bar.append(copy, up, down, share, download, speak, regen);
  contentWrap.appendChild(bar);
}

function addMessageToDom(role, content, message = null) {
  welcome.style.display = "none";

  const row = document.createElement("div");
  row.className = `message ${role}`;
  if (message?.id) row.dataset.messageId = message.id;

  if (role === "assistant") {
    const avatar = document.createElement("img");
    avatar.className = "assistant-avatar";
    avatar.src = "/static/favicon.svg?v=orca2";
    avatar.alt = "OrcaAI";
    row.appendChild(avatar);
  }

  const contentWrap = document.createElement("div");
  contentWrap.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(content);
  contentWrap.appendChild(bubble);

  row.appendChild(contentWrap);

  if (role === "assistant" && message) {
    addActions(contentWrap, message, row);
  }

  messages.appendChild(row);
  scrollBottom();
  return row;
}

function renderMessages(chat) {
  messages.innerHTML = "";
  const hasMessages = chat?.messages?.length;
  welcome.style.display = hasMessages ? "none" : "grid";

  if (!hasMessages) return;

  for (const message of chat.messages) {
    addMessageToDom(message.role, message.content, message);
  }

  scrollBottom();
}

function addThinking() {
  const row = document.createElement("div");
  row.className = "message assistant thinking-row";
  row.id = "thinking";
  row.innerHTML = `
    <img class="assistant-avatar" src="/static/favicon.svg?v=orca2" alt="OrcaAI">
    <div class="message-content">
      <div class="thinking-label" aria-live="polite">
        <span class="thinking-text">Thinking</span>
        <span class="thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span>
      </div>
    </div>`;
  messages.appendChild(row);
  scrollBottom();
}

async function removeThinkingSmooth() {
  const row = document.getElementById("thinking");
  if (!row) return;
  row.classList.add("thinking-exit");
  await new Promise(resolve => setTimeout(resolve, 180));
  row.remove();
}

function removeThinking() {
  document.getElementById("thinking")?.remove();
}

function randomThinkingDelay() {
  return 1000 + Math.floor(Math.random() * 1001);
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function revealAssistantMessage(message) {
  welcome.style.display = "none";

  const row = document.createElement("div");
  row.className = "message assistant answer-reveal";
  if (message?.id) row.dataset.messageId = message.id;

  const avatar = document.createElement("img");
  avatar.className = "assistant-avatar";
  avatar.src = "/static/favicon.svg?v=orca2";
  avatar.alt = "OrcaAI";
  row.appendChild(avatar);

  const contentWrap = document.createElement("div");
  contentWrap.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "bubble streaming-bubble";
  contentWrap.appendChild(bubble);
  row.appendChild(contentWrap);
  messages.appendChild(row);

  const fullText = message.content || "";
  const total = fullText.length;

  if (!total) {
    bubble.textContent = "...";
  } else {
    const duration = Math.min(1100, Math.max(280, total * 7));
    const started = performance.now();

    await new Promise(resolve => {
      function frame(now) {
        const progress = Math.min(1, (now - started) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);
        const count = Math.max(1, Math.floor(total * eased));
        bubble.textContent = fullText.slice(0, count);
        scrollBottom();

        if (progress < 1) {
          requestAnimationFrame(frame);
        } else {
          bubble.textContent = fullText;
          resolve();
        }
      }
      requestAnimationFrame(frame);
    });
  }

  bubble.classList.remove("streaming-bubble");
  addActions(contentWrap, message, row);
  scrollBottom();
  return row;
}

function scrollBottom() {
  const pane = document.getElementById("conversation");
  requestAnimationFrame(() => {
    pane.scrollTop = pane.scrollHeight;
  });
}

async function refreshChats() {
  state.chats = await api("/api/chats");
  renderSidebar();
}

async function createChat() {
  if (state.activeChatId) {
    try {
      const current = await api(`/api/chats/${state.activeChatId}`);
      if (!current.messages || current.messages.length === 0) {
        input.focus();
        closeSidebarOnMobile();
        return current;
      }
    } catch {}
  }

  const chat = await api("/api/chats", {
    method: "POST",
    body: "{}",
  });

  state.activeChatId = chat.id;
  await refreshChats();
  renderMessages(chat);
  input.focus();
  closeSidebarOnMobile();
  return chat;
}

async function openChat(id) {
  state.activeChatId = id;
  const chat = await api(`/api/chats/${id}`);
  renderSidebar();
  renderMessages(chat);
  closeSidebarOnMobile();
}

async function deleteChat(id) {
  await api(`/api/chats/${id}`, { method: "DELETE" });

  if (state.activeChatId === id) {
    state.activeChatId = null;
    messages.innerHTML = "";
    welcome.style.display = "grid";
  }

  await refreshChats();
}

function openDeleteModal(id) {
  deleteChatId = id;
  deleteModal.hidden = false;
}

function closeDeleteModal() {
  deleteModal.hidden = true;
  deleteChatId = null;
}

async function confirmDeleteChat() {
  if (!deleteChatId) return;
  const id = deleteChatId;
  closeDeleteModal();
  await deleteChat(id);
}

deleteConfirm.onclick = confirmDeleteChat;
deleteCancel.onclick = closeDeleteModal;
deleteModal.addEventListener("click", event => {
  if (event.target === deleteModal) closeDeleteModal();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !deleteModal.hidden) {
    closeDeleteModal();
  }
});

function openRenameModal(chat) {
  renameChatId = chat.id;
  renameInput.value = chat.title || "";
  renameModal.hidden = false;
  requestAnimationFrame(() => {
    renameInput.focus();
    renameInput.select();
  });
}

function closeRenameModal() {
  renameModal.hidden = true;
  renameChatId = null;
}

async function saveRename() {
  const title = renameInput.value.trim();
  if (!renameChatId || !title) return;

  await api(`/api/chats/${renameChatId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  closeRenameModal();
  await refreshChats();
}

renameSave.onclick = saveRename;
renameCancel.onclick = closeRenameModal;
renameModal.addEventListener("click", event => {
  if (event.target === renameModal) closeRenameModal();
});
renameInput.addEventListener("keydown", event => {
  if (event.key === "Enter") {
    event.preventDefault();
    saveRename();
  } else if (event.key === "Escape") {
    closeRenameModal();
  }
});

async function rateMessage(messageId, rating) {
  await api(
    `/api/chats/${state.activeChatId}/messages/${messageId}/rating`,
    {
      method: "POST",
      body: JSON.stringify({ rating }),
    }
  );
}

async function regenerateResponse(messageId, oldRow) {
  if (state.sending || !state.activeChatId) return;

  state.sending = true;

  // Remove the old answer immediately so regeneration feels like replacement,
  // not an additional answer below it.
  oldRow.remove();
  addThinking();

  try {
    const minimumWait = wait(randomThinkingDelay());
    const requestPromise = api(`/api/chats/${state.activeChatId}/regenerate`, {
      method: "POST",
      body: JSON.stringify({
        message_id: messageId,
        effort: state.effort,
      }),
    });

    const [result] = await Promise.all([requestPromise, minimumWait]);
    await removeThinkingSmooth();
    await openChat(state.activeChatId);
  } catch (error) {
    removeThinking();
    alert(error.message);
    await openChat(state.activeChatId);
  } finally {
    state.sending = false;
  }
}

async function sendMessage(text) {
  if (state.sending) return;
  if (!state.activeChatId) await createChat();

  state.sending = true;
  sendBtn.disabled = true;

  addMessageToDom("user", text);
  addThinking();

  try {
    const minimumWait = wait(randomThinkingDelay());
    const requestPromise = api(
      `/api/chats/${state.activeChatId}/messages`,
      {
        method: "POST",
        body: JSON.stringify({
          message: text,
          effort: state.effort,
        }),
      }
    );

    const [result] = await Promise.all([requestPromise, minimumWait]);
    await removeThinkingSmooth();
    await revealAssistantMessage(result.message);
    await refreshChats();

    // Keep the sidebar footer as stable product branding.
  } catch (error) {
    removeThinking();
    addMessageToDom("assistant", `Error: ${error.message}`);
  } finally {
    state.sending = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

async function checkStatus() {
  try {
    await api("/api/status");
  } catch {
    // Product branding remains visible even if the backend is unavailable.
  }
}

composer.addEventListener("submit", async event => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || state.sending) return;

  input.value = "";
  resizeInput();
  await sendMessage(text);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

newChatBtn.onclick = createChat;
document.getElementById("openSidebar").onclick = () => sidebar.classList.add("open");
document.getElementById("closeSidebar").onclick = () => sidebar.classList.remove("open");

function setSidebarCollapsed(collapsed) {
  document.body.classList.add("sidebar-animating");
  document.body.classList.toggle("sidebar-collapsed", collapsed);
  localStorage.setItem("orcaSidebarCollapsed", collapsed ? "1" : "0");
  window.setTimeout(() => {
    document.body.classList.remove("sidebar-animating");
  }, 320);
}

toggleSidebar.onclick = () => {
  setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
};

desktopSidebarOpen.onclick = () => {
  setSidebarCollapsed(false);
};

if (localStorage.getItem("orcaSidebarCollapsed") === "1") {
  setSidebarCollapsed(true);
}

document.addEventListener("keydown", event => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
  }
});

function closeSidebarOnMobile() {
  if (window.innerWidth <= 760) sidebar.classList.remove("open");
}

(async function boot() {
  await checkStatus();
  await refreshChats();

  if (state.chats.length) {
    await openChat(state.chats[0].id);
  } else {
    input.focus();
  }
})();
