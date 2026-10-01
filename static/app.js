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
const renameModal = document.getElementById("renameModal");
const renameInput = document.getElementById("renameInput");
const renameSave = document.getElementById("renameSave");
const renameCancel = document.getElementById("renameCancel");
let renameChatId = null;

const icons = {
  copy: '<svg aria-hidden="true" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"/></svg>',
  check: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
  up: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 10v10H4V10h3Zm3 10h7.2a2 2 0 0 0 1.9-1.37l2-6A2 2 0 0 0 19.2 10H15l.7-3.5A2.1 2.1 0 0 0 13.64 4H13l-4 6v10Z"/></svg>',
  down: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 14V4H4v10h3Zm3-10h7.2a2 2 0 0 1 1.9 1.37l2 6A2 2 0 0 1 19.2 14H15l.7 3.5A2.1 2.1 0 0 1 13.64 20H13l-4-6V4Z"/></svg>',
  share: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 16V4m0 0 4 4m-4-4L8 8"/><path d="M5 12v6a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-6"/></svg>',
  download: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M12 4v11m0 0 4-4m-4 4-4-4"/><path d="M5 19h14"/></svg>',
  speak: '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M5 10v4h4l5 4V6l-5 4H5Z"/><path d="M17 9a4 4 0 0 1 0 6m2-8a7 7 0 0 1 0 10"/></svg>',
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
    del.textContent = "×";
    del.title = "Delete chat";
    del.onclick = async event => {
      event.stopPropagation();
      await deleteChat(chat.id);
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
      body: JSON.stringify({ message_id: messageId }),
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
        body: JSON.stringify({ message: text }),
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
