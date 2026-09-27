const state = { chats: [], activeChatId: null, sending: false };

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

const icons = {
  copy: '<svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"/></svg>',
  up: '<svg viewBox="0 0 24 24"><path d="M7 10v10H4V10h3Zm3 10h7.2a2 2 0 0 0 1.9-1.37l2-6A2 2 0 0 0 19.2 10H15l.7-3.5A2.1 2.1 0 0 0 13.64 4H13l-4 6v10Z"/></svg>',
  down: '<svg viewBox="0 0 24 24"><path d="M7 14V4H4v10h3Zm3-10h7.2a2 2 0 0 1 1.9 1.37l2 6A2 2 0 0 1 19.2 14H15l.7 3.5A2.1 2.1 0 0 1 13.64 20H13l-4-6V4Z"/></svg>',
  share: '<svg viewBox="0 0 24 24"><path d="M12 16V4m0 0 4 4m-4-4L8 8"/><path d="M5 12v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11m0 0 4-4m-4 4-4-4"/><path d="M5 19h14"/></svg>',
  speak: '<svg viewBox="0 0 24 24"><path d="M5 10v4h4l5 4V6l-5 4H5Z"/><path d="M17 9a4 4 0 0 1 0 6m2-8a7 7 0 0 1 0 10"/></svg>',
  regen: '<svg viewBox="0 0 24 24"><path d="M20 6v5h-5"/><path d="M19 11a7 7 0 1 0 .2 5"/></svg>',
  more: '<svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></svg>'
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
    const del = document.createElement("button");
    del.className = "chat-delete";
    del.textContent = "×";
    del.onclick = async e => { e.stopPropagation(); await deleteChat(chat.id); };
    row.append(open, del);
    chatList.appendChild(row);
  }
}

function actionButton(icon, title, fn) {
  const button = document.createElement("button");
  button.className = "response-action";
  button.type = "button";
  button.title = title;
  button.innerHTML = icons[icon];
  button.onclick = fn;
  return button;
}

function addActions(row, message) {
  const bar = document.createElement("div");
  bar.className = "response-actions";

  const copy = actionButton("copy", "Copy", async () => {
    await navigator.clipboard.writeText(message.content);
    copy.classList.add("done");
    setTimeout(() => copy.classList.remove("done"), 800);
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
      try { await navigator.share({ text: message.content }); } catch {}
    } else {
      await navigator.clipboard.writeText(message.content);
    }
  });

  const download = actionButton("download", "Download", () => {
    const blob = new Blob([message.content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "localai-response.txt";
    a.click();
    URL.revokeObjectURL(url);
  });

  const speak = actionButton("speak", "Read aloud", () => {
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(message.content));
  });

  const regen = actionButton("regen", "Regenerate response", regenerateResponse);

  const moreWrap = document.createElement("div");
  moreWrap.className = "more-wrap";
  const more = actionButton("more", "More", () => {
    moreWrap.classList.toggle("open");
  });
  const menu = document.createElement("div");
  menu.className = "more-menu";
  const branch = document.createElement("button");
  branch.textContent = "Branch in new chat";
  branch.onclick = async () => {
    const result = await api(`/api/chats/${state.activeChatId}/branch`, {
      method: "POST",
      body: JSON.stringify({ message_id: message.id }),
    });
    await refreshChats();
    await openChat(result.id);
  };
  menu.appendChild(branch);
  moreWrap.append(more, menu);

  if (message.rating === "up") up.classList.add("selected");
  if (message.rating === "down") down.classList.add("selected");

  bar.append(copy, up, down, share, download, speak, regen, moreWrap);
  row.appendChild(bar);
}

function addMessageToDom(role, content, message = null) {
  welcome.style.display = "none";
  const row = document.createElement("div");
  row.className = `message ${role}`;

  const contentWrap = document.createElement("div");
  contentWrap.className = "message-content";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(content);
  contentWrap.appendChild(bubble);

  if (role === "assistant" && message) addActions(contentWrap, message);
  row.appendChild(contentWrap);
  messages.appendChild(row);
  scrollBottom();
  return row;
}

function renderMessages(chat) {
  messages.innerHTML = "";
  const has = chat?.messages?.length;
  welcome.style.display = has ? "none" : "grid";
  if (!has) return;
  for (const message of chat.messages) {
    addMessageToDom(message.role, message.content, message);
  }
  scrollBottom();
}

function addThinking() {
  const row = document.createElement("div");
  row.className = "message assistant";
  row.id = "thinking";
  row.innerHTML = '<div class="bubble"><div class="thinking"><span></span><span></span><span></span></div></div>';
  messages.appendChild(row);
  scrollBottom();
}

function removeThinking() { document.getElementById("thinking")?.remove(); }
function scrollBottom() {
  const pane = document.getElementById("conversation");
  requestAnimationFrame(() => pane.scrollTop = pane.scrollHeight);
}

async function refreshChats() {
  state.chats = await api("/api/chats");
  renderSidebar();
}

async function createChat() {
  const chat = await api("/api/chats", { method: "POST", body: "{}" });
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

async function rateMessage(messageId, rating) {
  await api(`/api/chats/${state.activeChatId}/messages/${messageId}/rating`, {
    method: "POST",
    body: JSON.stringify({ rating }),
  });
}

async function regenerateResponse() {
  if (state.sending || !state.activeChatId) return;
  state.sending = true;
  addThinking();
  try {
    await api(`/api/chats/${state.activeChatId}/regenerate`, { method: "POST", body: "{}" });
    await openChat(state.activeChatId);
  } catch (error) {
    removeThinking();
    alert(error.message);
  } finally {
    state.sending = false;
  }
}

async function sendMessage(text) {
  if (state.sending) return;
  if (!state.activeChatId) await createChat();

  state.sending = true;
  sendBtn.disabled = true;
  input.disabled = true;
  addMessageToDom("user", text);
  addThinking();

  try {
    const result = await api(`/api/chats/${state.activeChatId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message: text }),
    });
    removeThinking();
    addMessageToDom("assistant", result.message.content, result.message);
    await refreshChats();
    if (result.checkpoint) {
      statusTitle.textContent = result.checkpoint;
      statusSub.textContent = result.device ? `Running on ${result.device}` : "Loaded";
      statusDot.className = "status-dot ready";
    }
  } catch (error) {
    removeThinking();
    addMessageToDom("assistant", `Error: ${error.message}`);
  } finally {
    state.sending = false;
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

async function checkStatus() {
  try {
    const status = await api("/api/status");
    statusDot.className = status.ready ? "status-dot ready" : "status-dot bad";
    statusTitle.textContent = status.checkpoint || "Model not ready";
    statusSub.textContent = status.chat_tuned ? "Chat model ready" : "Base model only";
  } catch {
    statusDot.className = "status-dot bad";
    statusTitle.textContent = "Backend unavailable";
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
function closeSidebarOnMobile() {
  if (window.innerWidth <= 760) sidebar.classList.remove("open");
}

document.addEventListener("click", event => {
  if (!event.target.closest(".more-wrap")) {
    document.querySelectorAll(".more-wrap.open").forEach(x => x.classList.remove("open"));
  }
});

(async function boot() {
  await checkStatus();
  await refreshChats();
  if (state.chats.length) await openChat(state.chats[0].id);
  else input.focus();
})();
