const state = {
  chats: [],
  activeChatId: null,
  sending: false,
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

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return body;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderSidebar() {
  chatList.innerHTML = "";

  if (!state.chats.length) {
    const empty = document.createElement("div");
    empty.style.cssText = "color:#929398;font-size:13px;padding:10px;";
    empty.textContent = "No chats yet";
    chatList.appendChild(empty);
    return;
  }

  for (const chat of state.chats) {
    const row = document.createElement("div");
    row.className = "chat-item" + (chat.id === state.activeChatId ? " active" : "");

    const open = document.createElement("button");
    open.className = "chat-open";
    open.textContent = chat.title;
    open.title = chat.title;
    open.onclick = () => openChat(chat.id);

    const del = document.createElement("button");
    del.className = "chat-delete";
    del.textContent = "×";
    del.title = "Delete chat";
    del.onclick = async (event) => {
      event.stopPropagation();
      await deleteChat(chat.id);
    };

    row.append(open, del);
    chatList.appendChild(row);
  }
}

function renderMessages(chat) {
  messages.innerHTML = "";
  const hasMessages = chat && chat.messages && chat.messages.length;
  welcome.style.display = hasMessages ? "none" : "grid";

  if (!hasMessages) return;

  for (const message of chat.messages) {
    addMessageToDom(message.role, message.content);
  }

  scrollBottom();
}

function addMessageToDom(role, content, id = null) {
  welcome.style.display = "none";

  const row = document.createElement("div");
  row.className = `message ${role}`;
  if (id) row.id = id;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(content);

  row.appendChild(bubble);
  messages.appendChild(row);
  scrollBottom();
  return row;
}

function addThinking() {
  const row = document.createElement("div");
  row.className = "message assistant";
  row.id = "thinking";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';

  row.appendChild(bubble);
  messages.appendChild(row);
  scrollBottom();
}

function removeThinking() {
  document.getElementById("thinking")?.remove();
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
    body: JSON.stringify({}),
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
  input.focus();
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

async function sendMessage(text) {
  if (state.sending) return;

  if (!state.activeChatId) {
    await createChat();
  }

  state.sending = true;
  sendBtn.disabled = true;
  input.disabled = true;

  addMessageToDom("user", text);
  addThinking();

  try {
    const result = await api(
      `/api/chats/${state.activeChatId}/messages`,
      {
        method: "POST",
        body: JSON.stringify({ message: text }),
      }
    );

    removeThinking();
    addMessageToDom("assistant", result.message.content);
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
    if (status.ready) {
      statusDot.className = "status-dot ready";
      statusTitle.textContent = status.checkpoint || "Model ready";
      statusSub.textContent = status.chat_tuned
        ? "Chat model ready"
        : "Base model only — fine-tune next";
    } else {
      statusDot.className = "status-dot bad";
      statusTitle.textContent = "Model not ready";
      statusSub.textContent = "Finish training first";
    }
  } catch {
    statusDot.className = "status-dot bad";
    statusTitle.textContent = "Backend unavailable";
    statusSub.textContent = "Check web_app.py";
  }
}

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text || state.sending) return;

  input.value = "";
  resizeInput();
  await sendMessage(text);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

newChatBtn.onclick = createChat;

document.getElementById("openSidebar").onclick = () => {
  sidebar.classList.add("open");
};
document.getElementById("closeSidebar").onclick = () => {
  sidebar.classList.remove("open");
};

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
