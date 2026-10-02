const state = {
  chats: [],
  activeChatId: null,
  sending: false,
  effort: localStorage.getItem("orcaThinkingEffortV2") || "instant",
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
const sidebarAccount = document.getElementById("sidebarAccount");
const versionMenu = document.getElementById("versionMenu");
const effortPicker = document.getElementById("effortPicker");
const effortButton = document.getElementById("effortButton");
const effortLabel = document.getElementById("effortLabel");
const effortPopover = document.getElementById("effortPopover");
const effortCurrent = document.getElementById("effortCurrent");
const effortTrack = document.getElementById("effortTrack");
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
let voiceOriginalValue = "";
let voiceLiveTimer = null;
let voicePartialBusy = false;
let voiceSession = 0;
let voiceLastPartial = "";

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


const tooltipShortcuts = new Map([
  ["toggleSidebar", "Ctrl+Shift+S"],
  ["desktopSidebarOpen", "Ctrl+Shift+S"],
  ["effortButton", "Ctrl+Shift+M"],
  ["micBtn", "Ctrl+Shift+D"],
  ["newChat", "Ctrl+Shift+O"],
]);

const tooltipLabels = new Map([
  ["toggleSidebar", "Toggle sidebar"],
  ["desktopSidebarOpen", "Toggle sidebar"],
  ["openSidebar", "Open sidebar"],
  ["closeSidebar", "Close sidebar"],
  ["effortButton", "Thinking effort"],
  ["micBtn", "Dictate"],
  ["newChat", "New chat"],
  ["sidebarAccount", "OrcaAI"],
  ["sendBtn", "Send"],
  ["voiceCancel", "Cancel"],
  ["voiceStop", "Stop recording"],
  ["voiceSave", "Use transcription"],
]);

const tooltip = document.createElement("div");
tooltip.className = "orca-tooltip";
tooltip.hidden = true;
tooltip.innerHTML = '<span class="orca-tooltip-label"></span><span class="orca-tooltip-shortcut" hidden></span>';
document.body.appendChild(tooltip);

const tooltipLabel = tooltip.querySelector(".orca-tooltip-label");
const tooltipShortcut = tooltip.querySelector(".orca-tooltip-shortcut");

let tooltipShowTimer = null;
let tooltipHideTimer = null;
let tooltipTarget = null;

function tooltipAllowed() {
  return window.matchMedia("(hover: hover) and (pointer: fine)").matches;
}

function prepareTooltipTarget(element) {
  if (!element || element.closest(".chat-edit, .chat-delete, .chat-open")) return null;

  if (element.id && tooltipLabels.has(element.id)) {
    element.dataset.tooltip = tooltipLabels.get(element.id);
  } else if (!element.dataset.tooltip && element.getAttribute("title")) {
    element.dataset.tooltip = element.getAttribute("title");
  }

  if (element.dataset.tooltip) {
    element.removeAttribute("title");
    return element;
  }

  return null;
}

function findTooltipTarget(node) {
  if (!(node instanceof Element)) return null;
  const element = node.closest("[data-tooltip], [title], button");
  return prepareTooltipTarget(element);
}

function positionTooltip(target) {
  const rect = target.getBoundingClientRect();
  const tipRect = tooltip.getBoundingClientRect();
  const margin = 8;

  let left = rect.left + rect.width / 2 - tipRect.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - tipRect.width - margin));

  let top = rect.top - tipRect.height - 9;
  let placement = "top";

  if (top < margin) {
    top = rect.bottom + 9;
    placement = "bottom";
  }

  tooltip.dataset.placement = placement;
  tooltip.style.left = Math.round(left) + "px";
  tooltip.style.top = Math.round(top) + "px";
}

function showTooltip(target) {
  if (!tooltipAllowed() || !target?.dataset.tooltip) return;

  clearTimeout(tooltipHideTimer);
  tooltipTarget = target;
  tooltipLabel.textContent = target.dataset.tooltip;

  const shortcut = target.id ? tooltipShortcuts.get(target.id) : null;
  tooltipShortcut.hidden = !shortcut;
  tooltipShortcut.textContent = shortcut || "";

  tooltip.hidden = false;
  requestAnimationFrame(() => {
    positionTooltip(target);
    tooltip.classList.add("visible");
  });
}

function scheduleTooltip(target) {
  if (!tooltipAllowed() || !target?.dataset.tooltip) return;
  clearTimeout(tooltipShowTimer);
  clearTimeout(tooltipHideTimer);
  tooltipShowTimer = window.setTimeout(() => showTooltip(target), 750);
}

function hideTooltip() {
  clearTimeout(tooltipShowTimer);
  clearTimeout(tooltipHideTimer);
  tooltipTarget = null;
  tooltip.classList.remove("visible");
  tooltipHideTimer = window.setTimeout(() => {
    if (!tooltip.classList.contains("visible")) tooltip.hidden = true;
  }, 120);
}

document.addEventListener("pointerover", event => {
  const target = findTooltipTarget(event.target);
  if (!target) return;
  if (target.contains(event.relatedTarget)) return;
  scheduleTooltip(target);
});

document.addEventListener("pointerout", event => {
  if (!tooltipTarget && !tooltipShowTimer) return;
  const target = findTooltipTarget(event.target);
  if (!target) return;
  if (target.contains(event.relatedTarget)) return;
  hideTooltip();
});

document.addEventListener("focusin", event => {
  const target = findTooltipTarget(event.target);
  if (target) scheduleTooltip(target);
});

document.addEventListener("focusout", hideTooltip);

window.addEventListener("resize", () => {
  if (tooltipTarget && !tooltip.hidden) positionTooltip(tooltipTarget);
});


const effortCopy = {
  instant: { label: "Instant", value: 0 },
  medium: { label: "Medium", value: 1 },
  high: { label: "High", value: 2 },
};

const effortLevels = ["instant", "medium", "high"];
let draggingEffort = false;

function setEffort(level) {
  if (!effortCopy[level]) level = "instant";
  state.effort = level;
  localStorage.setItem("orcaThinkingEffortV2", level);
  effortPicker.dataset.effort = level;
  effortCurrent.textContent = effortCopy[level].label;
  effortTrack.setAttribute("aria-valuenow", String(effortCopy[level].value));
  effortTrack.style.removeProperty("--effort-position");
  effortLabel.textContent = effortPicker.classList.contains("open")
    ? "Thinking effort"
    : effortCopy[level].label;
}

function setEffortOpen(open) {
  effortPopover.hidden = !open;
  effortButton.setAttribute("aria-expanded", open ? "true" : "false");
  effortPicker.classList.toggle("open", open);
  effortLabel.textContent = open ? "Thinking effort" : effortCopy[state.effort].label;
}

function effortFromClientX(clientX, commit = false) {
  const rect = effortTrack.getBoundingClientRect();
  const inset = 15;
  const usable = Math.max(1, rect.width - inset * 2);
  const x = Math.max(inset, Math.min(rect.width - inset, clientX - rect.left));
  const position = (x - inset) / usable;
  const visualPercent = (x / Math.max(rect.width, 1)) * 100;

  effortTrack.style.setProperty("--effort-position", visualPercent + "%");

  const index = Math.max(0, Math.min(2, Math.round(position * 2)));
  const level = effortLevels[index];
  effortCurrent.textContent = effortCopy[level].label;

  if (commit) setEffort(level);
}

effortButton.addEventListener("click", event => {
  event.stopPropagation();
  setEffortOpen(effortPopover.hidden);
});

effortTrack.addEventListener("pointerdown", event => {
  event.preventDefault();
  draggingEffort = true;
  effortTrack.classList.add("dragging");
  effortFromClientX(event.clientX, false);
});

window.addEventListener("pointermove", event => {
  if (!draggingEffort) return;
  event.preventDefault();
  effortFromClientX(event.clientX, false);
}, { passive: false });

window.addEventListener("pointerup", event => {
  if (!draggingEffort) return;
  draggingEffort = false;
  effortTrack.classList.remove("dragging");
  effortFromClientX(event.clientX, true);
});

window.addEventListener("pointercancel", () => {
  if (!draggingEffort) return;
  draggingEffort = false;
  effortTrack.classList.remove("dragging");
  setEffort(state.effort);
});

effortTrack.addEventListener("keydown", event => {
  const current = effortCopy[state.effort].value;
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
    event.preventDefault();
    setEffort(effortLevels[Math.max(0, current - 1)]);
  } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
    event.preventDefault();
    setEffort(effortLevels[Math.min(2, current + 1)]);
  } else if (event.key === "Home") {
    event.preventDefault();
    setEffort("instant");
  } else if (event.key === "End") {
    event.preventDefault();
    setEffort("high");
  }
});

document.addEventListener("click", event => {
  if (!effortPicker.contains(event.target)) setEffortOpen(false);
});

function setVersionMenuOpen(open) {
  versionMenu.hidden = !open;
  sidebarAccount.setAttribute("aria-expanded", open ? "true" : "false");
}

sidebarAccount.addEventListener("click", event => {
  event.stopPropagation();
  setVersionMenuOpen(versionMenu.hidden);
});

versionMenu.addEventListener("click", event => {
  event.stopPropagation();
});

document.addEventListener("click", event => {
  if (!versionMenu.hidden && !sidebarAccount.contains(event.target)) {
    setVersionMenuOpen(false);
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !versionMenu.hidden) {
    setVersionMenuOpen(false);
    sidebarAccount.focus();
  }
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    setEffortOpen(false);
    setVersionMenuOpen(false);
  }
});

window.addEventListener("blur", () => {
  setEffortOpen(false);
  setVersionMenuOpen(false);
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

function stopLiveTranscription() {
  if (voiceLiveTimer) {
    clearInterval(voiceLiveTimer);
    voiceLiveTimer = null;
  }
  voicePartialBusy = false;
}

function buildVoiceInsertedValue(text) {
  const value = voiceOriginalValue;
  const start = Math.min(voiceInsertStart, value.length);
  const end = Math.min(Math.max(voiceInsertEnd, start), value.length);

  const before = value.slice(0, start);
  const after = value.slice(end);
  const clean = String(text || "").trim();

  if (!clean) {
    return {
      value,
      caret: start,
    };
  }

  const needsLeadingSpace =
    before &&
    !/\s$/.test(before) &&
    !/^[,.;!?]/.test(clean);
  const needsTrailingSpace =
    after &&
    !/^\s/.test(after) &&
    !/[\s([{]$/.test(clean);

  const inserted =
    (needsLeadingSpace ? " " : "") +
    clean +
    (needsTrailingSpace ? " " : "");

  return {
    value: before + inserted + after,
    caret: before.length + inserted.length,
  };
}

function renderVoiceTranscript(text) {
  const clean = String(text || "").trim();
  if (!clean) return;

  voiceLastPartial = clean;
  const rendered = buildVoiceInsertedValue(clean);
  input.value = rendered.value;
  resizeInput();
  input.setSelectionRange(rendered.caret, rendered.caret);
}

async function transcribeVoiceBlob(blob, partial, session) {
  if (!blob?.size) return null;

  const extension = blob.type.includes("ogg") ? "ogg" : "webm";
  const form = new FormData();
  form.append("audio", blob, `recording.${extension}`);
  if (partial) form.append("partial", "1");

  const response = await fetch("/api/transcribe", {
    method: "POST",
    body: form,
  });
  const result = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(result.error || "Voice transcription failed.");
  }

  if (session !== voiceSession) return null;
  return (result.text || "").trim();
}

async function updateLiveTranscript(session) {
  if (
    session !== voiceSession ||
    !state.recording ||
    voicePartialBusy ||
    voiceChunks.length < 2
  ) {
    return;
  }

  voicePartialBusy = true;
  try {
    const type = mediaRecorder?.mimeType || voiceChunks[0]?.type || "audio/webm";
    const blob = new Blob([...voiceChunks], { type });
    const text = await transcribeVoiceBlob(blob, true, session);

    if (text && session === voiceSession && state.recording) {
      renderVoiceTranscript(text);
    }
  } catch {
    // Partial transcription is best-effort. Final transcription still runs
    // when the user stops recording.
  } finally {
    voicePartialBusy = false;
  }
}

async function startVoiceRecording() {
  if (state.recording || state.transcribing) return;

  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    alert("Voice recording is not supported by this browser.");
    return;
  }

  voiceInsertStart = input.selectionStart ?? input.value.length;
  voiceInsertEnd = input.selectionEnd ?? voiceInsertStart;
  voiceOriginalValue = input.value;
  voiceLastPartial = "";
  voiceChunks = [];
  voiceSession += 1;
  const session = voiceSession;

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

    mediaRecorder.start(160);
    input.readOnly = true;
    setVoiceUi(true, false);
    startVoiceMeter(voiceStream);

    voiceLiveTimer = window.setInterval(() => {
      updateLiveTranscript(session);
    }, 1600);
  } catch (error) {
    stopLiveTranscription();
    cleanupVoiceStream();
    input.readOnly = false;
    setVoiceUi(false, false);
    alert(error?.message || "Microphone access failed.");
  }
}

function discardVoiceRecording() {
  voiceSession += 1;
  stopLiveTranscription();

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  voiceChunks = [];
  cleanupVoiceStream();
  setVoiceUi(false, false);

  input.readOnly = false;
  input.value = voiceOriginalValue;
  resizeInput();
  input.focus();
  input.setSelectionRange(voiceInsertStart, voiceInsertEnd);
}

function stopRecorderToBlob() {
  return new Promise(resolve => {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
      const type = voiceChunks[0]?.type || "audio/webm";
      resolve(new Blob([...voiceChunks], { type }));
      return;
    }

    const recorder = mediaRecorder;
    recorder.addEventListener("stop", () => {
      const type = recorder.mimeType || voiceChunks[0]?.type || "audio/webm";
      resolve(new Blob([...voiceChunks], { type }));
    }, { once: true });

    recorder.stop();
  });
}

async function finishVoiceRecording() {
  if ((!state.recording && !mediaRecorder) || state.transcribing) return;

  stopLiveTranscription();
  setVoiceUi(true, true);
  stopVoiceMeter();

  const session = voiceSession;

  try {
    const blob = await stopRecorderToBlob();
    cleanupVoiceStream();

    if (!blob.size) throw new Error("No audio was recorded.");

    const text = await transcribeVoiceBlob(blob, false, session);
    const finalText = text || voiceLastPartial;

    if (finalText) {
      const rendered = buildVoiceInsertedValue(finalText);
      input.value = rendered.value;
      resizeInput();
      input.setSelectionRange(rendered.caret, rendered.caret);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    } else {
      input.value = voiceOriginalValue;
      resizeInput();
    }

    voiceChunks = [];
    setVoiceUi(false, false);
  } catch (error) {
    input.value = voiceOriginalValue;
    resizeInput();
    setVoiceUi(false, false);
    alert(error?.message || "Voice transcription failed.");
  } finally {
    cleanupVoiceStream();
    input.readOnly = false;
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
    edit.setAttribute("aria-label", "Rename chat");
    edit.innerHTML = '<svg viewBox="0 0 24 24"><path d="M4 20h4l11-11a2.8 2.8 0 0 0-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4"/></svg>';
    edit.onclick = event => {
      event.stopPropagation();
      openRenameModal(chat);
    };

    const del = document.createElement("button");
    del.className = "chat-delete";
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
  button.dataset.tooltip = title;
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
    copy.dataset.tooltip = "Copied";
    copy.setAttribute("aria-label", "Copied");
    setTimeout(() => {
      copy.innerHTML = icons.copy;
      copy.dataset.tooltip = "Copy";
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

function setChatRoute(chat, replace = false) {
  const url = chat?.id ? `/c/${encodeURIComponent(chat.id)}` : "/";
  document.title = chat?.title || "OrcaAI";

  const sameRoute = window.location.pathname === url;
  const method = (replace || sameRoute) ? "replaceState" : "pushState";
  history[method]({ chatId: chat?.id || null }, "", url);
}

function showNewChat(pushRoute = true) {
  if (!state.activeChatId && welcome.style.display !== "none") {
    input.focus();
    return;
  }

  state.activeChatId = null;
  messages.innerHTML = "";
  welcome.style.display = "grid";
  renderSidebar();
  if (pushRoute) setChatRoute(null);
  else document.title = "OrcaAI";
  input.focus();
  closeSidebarOnMobile();
}

function chatIdFromPath() {
  const match = window.location.pathname.match(/^\/c\/([^/]+)$/);
  return match ? decodeURIComponent(match[1]) : null;
}

async function createChat() {
  const chat = await api("/api/chats", {
    method: "POST",
    body: "{}",
  });

  state.activeChatId = chat.id;
  await refreshChats();
  renderMessages(chat);
  setChatRoute(chat);
  input.focus();
  closeSidebarOnMobile();
  return chat;
}

async function openChat(id, options = {}) {
  state.activeChatId = id;
  const chat = await api(`/api/chats/${id}`);
  renderSidebar();
  renderMessages(chat);
  setChatRoute(chat, Boolean(options.replace));
  closeSidebarOnMobile();
  return chat;
}

async function deleteChat(id) {
  await api(`/api/chats/${id}`, { method: "DELETE" });

  if (state.activeChatId === id) {
    state.activeChatId = null;
    messages.innerHTML = "";
    welcome.style.display = "grid";
    setChatRoute(null, true);
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

  const renamed = await api(`/api/chats/${renameChatId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
  if (state.activeChatId === renameChatId) {
    document.title = renamed.title;
  }
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

    const active = state.chats.find(chat => chat.id === state.activeChatId);
    if (active) {
      document.title = active.title;
      history.replaceState(
        { chatId: active.id },
        "",
        `/c/${encodeURIComponent(active.id)}`
      );
    }

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

newChatBtn.onclick = () => showNewChat(true);
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
  if (!event.ctrlKey || !event.shiftKey || event.repeat) return;

  const key = event.key.toLowerCase();

  if (key === "s") {
    event.preventDefault();
    setSidebarCollapsed(!document.body.classList.contains("sidebar-collapsed"));
  } else if (key === "m") {
    event.preventDefault();
    setEffortOpen(effortPopover.hidden);
  } else if (key === "d") {
    event.preventDefault();
    if (state.recording) finishVoiceRecording();
    else if (!state.transcribing) startVoiceRecording();
  } else if (key === "o") {
    event.preventDefault();
    showNewChat(true);
  }
});

function closeSidebarOnMobile() {
  if (window.innerWidth <= 760) sidebar.classList.remove("open");
}

window.addEventListener("popstate", async () => {
  const chatId = chatIdFromPath();

  if (!chatId) {
    showNewChat(false);
    return;
  }

  try {
    await openChat(chatId, { replace: true });
  } catch {
    state.activeChatId = null;
    messages.innerHTML = "";
    welcome.style.display = "grid";
    renderSidebar();
    setChatRoute(null, true);
  }
});

(async function boot() {
  await checkStatus();
  await refreshChats();

  const chatId = chatIdFromPath();
  if (chatId) {
    try {
      await openChat(chatId, { replace: true });
      return;
    } catch {
      setChatRoute(null, true);
    }
  }

  state.activeChatId = null;
  messages.innerHTML = "";
  welcome.style.display = "grid";
  document.title = "OrcaAI";
  history.replaceState({ chatId: null }, "", "/");
  renderSidebar();
  input.focus();
})();
