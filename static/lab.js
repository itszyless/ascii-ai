const $ = id => document.getElementById(id);

const FEATURES = [
  ["Model status","Active checkpoint, version and tuning stage"],
  ["GPU monitor","CUDA device and VRAM usage"],
  ["Parameter counter","Total and trainable parameters"],
  ["Architecture inspector","Layers, heads, embeddings and context"],
  ["Checkpoint step","See exactly where training stopped"],
  ["Validation metrics","Best and saved losses"],
  ["Checkpoint size","On-disk model footprint"],
  ["Tokenizer visualizer","Inspect every token and token ID"],
  ["Token counter","Prompt token count"],
  ["Context meter","See context-window usage"],
  ["Raw token IDs","Exact numerical model input"],
  ["Decoded tokens","Human-readable token pieces"],
  ["Next-token viewer","Top likely next tokens"],
  ["Probability bars","Visual token probability distribution"],
  ["Entropy meter","How uncertain the model is"],
  ["Raw continuation","Test the pretrained base behavior"],
  ["Chat prompt mode","Test assistant formatting"],
  ["Temperature control","Adjust generation randomness"],
  ["Top-k control","Limit candidate token pool"],
  ["Max-token control","Set response length ceiling"],
  ["Repetition penalty","Reduce looping outputs"],
  ["Seed control","Reproduce generations"],
  ["Latency meter","Measure generation response time"],
  ["Tokens/sec","Measure inference throughput"],
  ["Output token count","Track generated length"],
  ["Checkpoint explorer","Inspect every local .pt file"],
  ["Active checkpoint badge","Know exactly which model is loaded"],
  ["Model fallback visibility","See v3/v2 fallback state"],
  ["Persistent local memory","Store facts outside weights"],
  ["Memory manager","Review and delete memories"],
  ["Knowledge documents","Add local reference material"],
  ["Knowledge retrieval","Search documents locally"],
  ["Evaluation suite","Run repeatable sanity prompts"],
  ["Chat statistics","Counts chats and messages"],
  ["Feedback statistics","Track good/bad response ratings"],
  ["Knowledge statistics","Count local reference docs"],
  ["Privacy indicator","Clearly shows local-only operation"],
  ["Architecture flow","Visual next-token pipeline"],
  ["Generation presets","Reproducible parameter combinations"],
  ["Developer lab","All internals in one presentation-ready UI"]
];

async function api(url, options={}) {
  const response = await fetch(url, {
    headers: {"Content-Type":"application/json"},
    ...options
  });
  const body = await response.json().catch(()=>({}));
  if (!response.ok) throw new Error(body.error || `Request failed ${response.status}`);
  return body;
}

function number(value) {
  if (value == null) return "—";
  return Number(value).toLocaleString();
}

function metric(label, value) {
  return `<div class="metric"><small>${label}</small><strong>${value}</strong></div>`;
}

async function loadModel() {
  const data = await api("/api/lab/model");
  const cfg = data.config || {};
  $("metricGrid").innerHTML = [
    metric("Checkpoint", data.checkpoint),
    metric("Parameters", number(data.parameters)),
    metric("Context", number(data.context_length)),
    metric("Vocabulary", number(data.vocab_size)),
    metric("Step", number(data.step)),
    metric("Stage", data.stage || "—"),
    metric("Best val loss", data.best_val == null ? "—" : Number(data.best_val).toFixed(4)),
    metric("Device", data.device)
  ].join("");

  const layers = cfg.num_layers ?? "?";
  const heads = cfg.num_heads ?? "?";
  const embed = cfg.embedding_size ?? "?";
  $("architectureLabel").textContent = `${layers} layers · ${heads} heads · ${embed}d`;
  $("blocksNode").textContent = `Transformer × ${layers}`;

  if (data.gpu) {
    $("metricGrid").insertAdjacentHTML("beforeend",
      metric("GPU", data.gpu.name) +
      metric("VRAM allocated", `${data.gpu.allocated_mb} MB`) +
      metric("VRAM free", `${data.gpu.free_mb} MB`) +
      metric("Model size", `${data.checkpoint_mb} MB`)
    );
  }
}

async function tokenize() {
  const data = await api("/api/lab/tokenize", {
    method:"POST",
    body:JSON.stringify({text:$("tokenText").value})
  });
  $("tokenSummary").textContent = `${data.count} tokens · ${data.ids.join(", ")}`;
  $("tokenList").innerHTML = data.tokens.map(t =>
    `<div class="token"><b>#${t.index} · ID ${t.id}</b>${escapeHtml(t.decoded || t.raw || "∅")}</div>`
  ).join("");
}

async function predict() {
  const data = await api("/api/lab/next-token", {
    method:"POST",
    body:JSON.stringify({prompt:$("predictionPrompt").value, top_n:12})
  });
  $("predictionMeta").textContent =
    `${data.prompt_tokens} prompt tokens · entropy ${data.entropy.toFixed(3)}`;
  const max = Math.max(...data.predictions.map(x=>x.percent), 0.0001);
  $("predictionList").innerHTML = data.predictions.map(item => {
    const width = item.percent / max * 100;
    return `<div class="prediction">
      <code>${escapeHtml(item.token || item.raw || "∅")}</code>
      <div class="bar"><span style="width:${width}%"></span></div>
      <strong>${item.percent.toFixed(2)}%</strong>
    </div>`;
  }).join("");
}

async function generate() {
  $("generateBtn").disabled = true;
  $("generationOutput").textContent = "Generating…";
  try {
    const data = await api("/api/lab/generate", {
      method:"POST",
      body:JSON.stringify({
        prompt:$("playgroundPrompt").value,
        mode:$("mode").value,
        temperature:Number($("temperature").value),
        top_k:Number($("topK").value),
        max_tokens:Number($("maxTokens").value),
        repetition_penalty:Number($("repetitionPenalty").value),
        seed:Number($("seed").value)
      })
    });
    $("generationOutput").textContent = data.output || "(empty)";
    $("generationStats").innerHTML =
      `<span>${data.input_tokens} input tokens</span>
       <span>${data.output_tokens} output tokens</span>
       <span>${data.latency_ms} ms</span>
       <span>${data.tokens_per_second} tok/s</span>`;
  } finally {
    $("generateBtn").disabled = false;
  }
}

async function loadCheckpoints() {
  const items = await api("/api/lab/checkpoints");
  $("checkpointList").innerHTML = items.map(item =>
    `<div class="checkpoint ${item.active ? "active":""}">
      <h4>${escapeHtml(item.name)} ${item.active ? "· ACTIVE":""}</h4>
      <p>${item.megabytes} MB · step ${item.step ?? "—"}</p>
      <p>${escapeHtml(item.stage || "unknown stage")}</p>
      <p>best val: ${item.best_val == null ? "—" : Number(item.best_val).toFixed(4)}</p>
    </div>`
  ).join("") || "<p>No checkpoints yet.</p>";
}

async function loadMemories() {
  const items = await api("/api/lab/memories");
  $("memoryList").innerHTML = items.map(item =>
    `<div class="stack-item"><div><h4>Memory</h4><p>${escapeHtml(item.text)}</p></div>
      <button onclick="deleteMemory('${item.id}')">Delete</button></div>`
  ).join("") || '<div class="tiny-stats">No memories stored.</div>';
}

async function deleteMemory(id) {
  await api(`/api/lab/memories/${id}`, {method:"DELETE"});
  loadMemories();
}
window.deleteMemory = deleteMemory;

async function addMemory() {
  const text = $("memoryText").value.trim();
  if (!text) return;
  await api("/api/lab/memories", {
    method:"POST",
    body:JSON.stringify({text})
  });
  $("memoryText").value = "";
  loadMemories();
}

async function loadKnowledge() {
  const items = await api("/api/lab/knowledge");
  $("knowledgeList").innerHTML = items.map(item =>
    `<div class="stack-item"><div><h4>${escapeHtml(item.title)}</h4>
      <p>${escapeHtml(item.content.slice(0,260))}</p></div>
      <button onclick="deleteKnowledge('${item.id}')">Delete</button></div>`
  ).join("") || '<div class="tiny-stats">No local documents.</div>';
}

async function deleteKnowledge(id) {
  await api(`/api/lab/knowledge/${id}`, {method:"DELETE"});
  loadKnowledge();
}
window.deleteKnowledge = deleteKnowledge;

async function addKnowledge() {
  const title = $("knowledgeTitle").value.trim();
  const content = $("knowledgeContent").value.trim();
  if (!content) return;
  await api("/api/lab/knowledge", {
    method:"POST",
    body:JSON.stringify({title,content})
  });
  $("knowledgeTitle").value = "";
  $("knowledgeContent").value = "";
  loadKnowledge();
}

async function searchKnowledge() {
  const results = await api("/api/lab/knowledge/search", {
    method:"POST",
    body:JSON.stringify({query:$("knowledgeQuery").value})
  });
  $("knowledgeResults").innerHTML = results.map(item =>
    `<div class="stack-item"><div><h4>${escapeHtml(item.title)}</h4>
    <p>score ${item.score} · ${escapeHtml(item.preview)}</p></div></div>`
  ).join("") || '<div class="tiny-stats">No matches.</div>';
}

async function runEvaluation() {
  $("runEvalBtn").disabled = true;
  $("evalResults").innerHTML = '<div class="tiny-stats">Running evaluation…</div>';
  try {
    const data = await api("/api/lab/evaluate", {
      method:"POST",
      body:"{}"
    });
    $("evalResults").innerHTML = data.results.map(item =>
      `<div class="eval-card"><strong>${escapeHtml(item.prompt)}</strong>
      <p>${escapeHtml(item.output || "(empty)")}</p>
      <div class="tiny-stats">${item.tokens} tokens · ${item.latency_ms} ms</div></div>`
    ).join("");
  } finally {
    $("runEvalBtn").disabled = false;
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}

function renderFeatures() {
  $("featureGrid").innerHTML = FEATURES.map((item,index)=>
    `<div class="feature"><h4>${String(index+1).padStart(2,"0")} · ${item[0]}</h4><p>${item[1]}</p></div>`
  ).join("");
}

$("temperature").oninput = e => $("temperatureValue").textContent = e.target.value;
$("topK").oninput = e => $("topKValue").textContent = e.target.value;
$("generateBtn").onclick = generate;
$("clearOutput").onclick = () => {
  $("generationOutput").textContent = "Waiting for prompt.";
  $("generationStats").innerHTML = "";
};
$("tokenizeBtn").onclick = tokenize;
$("predictBtn").onclick = predict;
$("addMemoryBtn").onclick = addMemory;
$("addKnowledgeBtn").onclick = addKnowledge;
$("searchKnowledgeBtn").onclick = searchKnowledge;
$("runEvalBtn").onclick = runEvaluation;
$("refreshAll").onclick = boot;

async function boot() {
  renderFeatures();
  await Promise.allSettled([
    loadModel(),
    loadCheckpoints(),
    loadMemories(),
    loadKnowledge()
  ]);
  tokenize();
  predict();
}

boot();
