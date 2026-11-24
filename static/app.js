// Tokenizer Lab front-end. Talks to app.py over a tiny JSON API.
const $ = (id) => document.getElementById(id);
const SPACE = "▁";
const TYPE_COLORS = { word: "#7c9cff", subword: "#5fd39a", char: "#f2c14e", added: "#ff6fb5", special: "#8a8f9c" };
const TYPE_NAMES = { word: "whole word", subword: "subword", char: "character", added: "added word", special: "special" };
const CHIP_COLORS = ["#a5b8ff", "#8fe3b8", "#f7d488", "#ffb3a7", "#c9b2ff", "#8ad8e8", "#f5b3d9", "#d4e69a"];

let state = null;          // /api/state
let encoded = null;        // last /api/encode result
let selected = null;       // { token, neighbours }
let pointByToken = new Map();

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (t) => t.replaceAll("\n", "↵");

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function setStatus(el, msg, kind = "") { el.className = "status " + kind; el.innerHTML = msg; }

function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

// ------------------------------------------------------------------ state
async function refresh() {
  state = await api("/api/state");
  pointByToken = new Map(state.points.map((p) => [p.token, p]));
  renderStats();
  renderSettings();
  renderAdded();
  renderVocab();
  $("tokenList").innerHTML = state.points.slice(0, 3000).map((p) => `<option value="${esc(p.token)}">`).join("");
  await encodeNow();
  if (selected) await select(selected.token, false);
}

function renderStats() {
  const s = state.stats;
  const items = [
    [s.vocab_size, "tokens in vocab"], [s.merges, "BPE merges"], [s.added, "added words"],
    [s.corpus_words.toLocaleString(), "corpus words"], [s.embedding_dim + "→3", "embedding dims"],
  ];
  $("stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

function renderSettings() {
  const s = state.settings;
  $("vocabSize").value = s.vocab_size;
  $("minFreq").value = s.min_frequency;
  $("dim").value = s.dim;
  $("window").value = s.window;
  $("lowercase").checked = s.lowercase;
}

// ------------------------------------------------------------------ 1. tokeniser
const encodeSoon = debounce(encodeNow, 120);

async function encodeNow() {
  encoded = await api("/api/encode", { text: $("input").value });
  renderChips();
  drawPlot();
}

function renderChips() {
  const showIds = $("showIds").checked, showSp = $("showSpaces").checked;
  const toks = encoded.tokens;
  $("chips").innerHTML = toks.map((t, i) => {
    let label = esc(pretty(t.token));
    if (t.token.startsWith(SPACE)) label = showSp ? `<span class="sp">${SPACE}</span>${esc(t.token.slice(1))}` : " " + esc(t.token.slice(1));
    const cls = ["chip", t.unknown ? "unknown" : "", t.type === "added" ? "added" : ""].join(" ");
    const title = `${t.token}  ·  id ${t.id}  ·  ${TYPE_NAMES[t.type] || t.type}${t.unknown ? " (unknown → <unk>)" : ""}`;
    return `<span class="${cls}" style="background:${CHIP_COLORS[i % CHIP_COLORS.length]}" data-token="${esc(t.token)}" title="${esc(title)}"><span>${label}</span>${showIds ? `<small>${t.id}</small>` : ""}</span>`;
  }).join("") || `<span class="muted">Type something above…</span>`;

  const n = toks.length, chars = encoded.characters;
  const unknown = toks.filter((t) => t.unknown).length;
  $("metrics").innerHTML = `
    <span><b>${n}</b> tokens</span>
    <span><b>${chars}</b> characters</span>
    <span><b>${encoded.words}</b> words</span>
    <span><b>${n ? (chars / n).toFixed(2) : "–"}</b> chars / token</span>
    ${unknown ? `<span style="color:var(--bad)"><b>${unknown}</b> unknown</span>` : ""}`;
  $("ids").textContent = "[" + encoded.ids.join(", ") + "]";
  $("decoded").textContent = encoded.decoded;
}

$("chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (chip) select(chip.dataset.token);
});

// ------------------------------------------------------------------ 2. build vocabulary
async function loadCorpus() {
  const { corpus } = await api("/api/corpus");
  $("corpus").value = corpus;
}

$("buildBtn").addEventListener("click", async () => {
  const btn = $("buildBtn");
  btn.disabled = true;
  setStatus($("buildStatus"), "Learning merges and training embeddings…");
  try {
    const { build } = await api("/api/build", {
      corpus: $("corpus").value,
      options: {
        vocab_size: +$("vocabSize").value, min_frequency: +$("minFreq").value,
        dim: +$("dim").value, window: +$("window").value, lowercase: $("lowercase").checked,
      },
    });
    selected = null;
    await refresh();
    const note = build.stopped_early ? ` (stopped early: no pair appears ≥ ${$("minFreq").value} times — add more text or lower “min pair count”)` : "";
    setStatus($("buildStatus"),
      `✓ ${build.unique_words} unique words · ${build.alphabet} base characters · ${build.merges} merges → <b>${build.vocab_size}</b> tokens. ` +
      `${build.tokens_with_embeddings} tokens embedded in ${build.dim} dims.${note}`, "ok");
  } catch (err) {
    setStatus($("buildStatus"), "✗ " + esc(err.message), "err");
  } finally {
    btn.disabled = false;
  }
});

$("fileInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("corpus").value = await file.text();
  setStatus($("buildStatus"), `Loaded ${esc(file.name)} (${$("corpus").value.split(/\s+/).length.toLocaleString()} words). Press “Build vocabulary”.`);
  e.target.value = "";
});

// ------------------------------------------------------------------ 3. embeddings in 3D
function visible(p) {
  const f = $("filter").value;
  if (p.type === "special") return false;
  if (f === "words") return p.type === "word" || p.type === "added";
  if (f === "subword") return p.type === "subword" || p.type === "char";
  if (f === "added") return p.type === "added";
  return true;
}

function hover(p) {
  return `<b>${esc(p.token)}</b><br>id ${p.id} · ${TYPE_NAMES[p.type]}<br>seen ${p.freq}× in corpus` +
    (p.source && p.source !== "trained" ? `<br><i>embedding ${p.source}</i>` : "");
}

function drawPlot() {
  if (!state || typeof Plotly === "undefined") return;
  const allLabels = $("labels").checked;
  const traces = [];

  for (const type of ["word", "subword", "char", "added"]) {
    const pts = state.points.filter((p) => p.type === type && visible(p));
    if (!pts.length) continue;
    traces.push({
      type: "scatter3d", mode: allLabels ? "markers+text" : "markers", name: TYPE_NAMES[type],
      x: pts.map((p) => p.x), y: pts.map((p) => p.y), z: pts.map((p) => p.z),
      text: pts.map((p) => p.token.replace(SPACE, "")), customdata: pts.map((p) => p.token),
      hovertext: pts.map(hover), hoverinfo: "text",
      textfont: { color: "#c8cdd8", size: 10 },
      marker: {
        size: pts.map((p) => (type === "added" ? 7 : 2.5 + Math.min(4, Math.log2(1 + p.freq) * 0.7))),
        color: TYPE_COLORS[type], opacity: type === "added" ? 1 : 0.75,
        symbol: type === "added" ? "diamond" : "circle",
      },
    });
  }

  // tokens from the tokeniser input
  if (encoded) {
    const seq = encoded.tokens.map((t) => pointByToken.get(t.token)).filter(Boolean);
    if ($("path").checked && seq.length > 1) {
      traces.push({
        type: "scatter3d", mode: "lines", name: "input path", hoverinfo: "skip",
        x: seq.map((p) => p.x), y: seq.map((p) => p.y), z: seq.map((p) => p.z),
        line: { color: "rgba(255,255,255,0.35)", width: 3 },
      });
    }
    const uniq = [...new Map(seq.map((p) => [p.token, p])).values()];
    if (uniq.length) {
      traces.push({
        type: "scatter3d", mode: "markers+text", name: "your input",
        x: uniq.map((p) => p.x), y: uniq.map((p) => p.y), z: uniq.map((p) => p.z),
        text: uniq.map((p) => p.token.replace(SPACE, "")), customdata: uniq.map((p) => p.token),
        hovertext: uniq.map(hover), hoverinfo: "text", textposition: "top center",
        textfont: { color: "#ffffff", size: 13 },
        marker: { size: 7, color: uniq.map((p) => TYPE_COLORS[p.type]), line: { color: "#ffffff", width: 2 } },
      });
    }
  }

  // selected token + lines to its nearest neighbours
  if (selected && pointByToken.has(selected.token)) {
    const s = pointByToken.get(selected.token);
    const nbs = selected.neighbours.map((n) => pointByToken.get(n.token)).filter(Boolean);
    const lx = [], ly = [], lz = [];
    for (const n of nbs) { lx.push(s.x, n.x, null); ly.push(s.y, n.y, null); lz.push(s.z, n.z, null); }
    traces.push({
      type: "scatter3d", mode: "lines", name: "neighbours", hoverinfo: "skip",
      x: lx, y: ly, z: lz, line: { color: "#ffd166", width: 2 },
    });
    traces.push({
      type: "scatter3d", mode: "markers+text", name: "neighbour tokens",
      x: nbs.map((p) => p.x), y: nbs.map((p) => p.y), z: nbs.map((p) => p.z),
      text: nbs.map((p) => p.token.replace(SPACE, "")), customdata: nbs.map((p) => p.token),
      hovertext: nbs.map(hover), hoverinfo: "text", textfont: { color: "#ffd166", size: 11 },
      marker: { size: 5, color: "#ffd166" }, showlegend: false,
    });
    traces.push({
      type: "scatter3d", mode: "markers+text", name: "selected",
      x: [s.x], y: [s.y], z: [s.z], text: [s.token.replace(SPACE, "")], customdata: [s.token],
      hovertext: [hover(s)], hoverinfo: "text", textposition: "top center",
      textfont: { color: "#ffd166", size: 15 },
      marker: { size: 11, color: "#ffd166", symbol: "diamond", line: { color: "#0b0d12", width: 1 } },
    });
  }

  const axis = (title) => ({
    title, color: "#6f7686", gridcolor: "#232835", zerolinecolor: "#353b49", showbackground: false,
    showspikes: false,
  });
  const layout = {
    uirevision: "keep", paper_bgcolor: "#0b0d12", plot_bgcolor: "#0b0d12",
    margin: { l: 0, r: 0, t: 0, b: 0 }, font: { color: "#c8cdd8" },
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(23,26,33,0.8)", bordercolor: "#2a2f3a", borderwidth: 1 },
    scene: { xaxis: axis("PC 1"), yaxis: axis("PC 2"), zaxis: axis("PC 3"), aspectmode: "cube",
             camera: { eye: { x: 1.4, y: 1.4, z: 0.9 } } },
    hoverlabel: { bgcolor: "#171a21", bordercolor: "#7c9cff", font: { family: "Menlo, monospace" } },
  };
  Plotly.react("plot", traces, layout, { displaylogo: false, responsive: true, scrollZoom: false });
}

function attachPlotEvents() {
  const plot = $("plot");
  if (typeof Plotly === "undefined") {
    plot.innerHTML = `<p class="muted" style="padding:20px">Could not load Plotly from the CDN. Check your internet connection and reload.</p>`;
    return;
  }
  drawPlot();
  plot.on("plotly_click", (ev) => {
    const tok = ev.points[0] && ev.points[0].customdata;
    if (tok) select(tok);
  });
}

async function select(token, redraw = true) {
  const data = await api("/api/neighbours?token=" + encodeURIComponent(token));
  const p = state.points.find((x) => x.token === token);
  const tid = state.vocab.find((v) => v.token === token);
  if (!data.coords) {
    selected = { token, neighbours: [] };
    $("selected").className = "";
    $("selected").innerHTML = `${esc(token)} <span class="muted small">${tid ? "id " + tid.id : "not in vocabulary"} · no embedding</span>`;
    $("neighbours").innerHTML = "";
  } else {
    selected = { token, neighbours: data.neighbours };
    $("selected").className = "";
    $("selected").innerHTML = `${esc(token)}<div class="muted small">id ${p ? p.id : "?"} · ${p ? TYPE_NAMES[p.type] : ""} · seen ${p ? p.freq : 0}× · ${esc(data.source || "")}</div>`;
    $("neighbours").innerHTML = data.neighbours.map((n) =>
      `<li data-token="${esc(n.token)}">${esc(n.token)} <span class="sim">${n.similarity.toFixed(2)}</span></li>`).join("");
  }
  if (redraw) drawPlot();
}

$("neighbours").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (li) select(li.dataset.token);
});

$("find").addEventListener("change", () => {
  const q = $("find").value.trim().toLowerCase();
  if (!q) return;
  const tok = [q, SPACE + q].find((t) => pointByToken.has(t)) ||
    state.points.find((p) => p.token.replace(SPACE, "") === q)?.token;
  if (tok) select(tok);
  else $("selected").innerHTML = `<span class="muted">“${esc(q)}” is not a token with an embedding.</span>`;
});

for (const id of ["filter", "labels", "path"]) $(id).addEventListener("change", drawPlot);

$("retrainBtn").addEventListener("click", async () => {
  const btn = $("retrainBtn");
  btn.disabled = true;
  try { await api("/api/retrain_embeddings", {}); await refresh(); } finally { btn.disabled = false; }
});

// ------------------------------------------------------------------ 4. add new words
function renderAdded() {
  $("addedList").innerHTML = state.added.length
    ? state.added.map((a) => `<li>${esc(a.word)} <small>#${a.id}</small><button data-token="${esc(a.token)}" title="Remove">×</button></li>`).join("")
    : `<li class="muted" style="border:none;background:none;padding:0">None yet.</li>`;
}

async function addWord() {
  const word = $("newWord").value.trim();
  if (!word) return;
  const btn = $("addBtn");
  btn.disabled = true;
  try {
    const r = await api("/api/add_word", { word, examples: $("examples").value });
    const before = r.before.map((t) => `<code>${esc(t)}</code>`).join(" ");
    if (r.status === "added") {
      setStatus($("addStatus"),
        `✓ Added <code>${esc(r.token)}</code> as id <b>${r.id}</b>. Before: ${before} (${r.before.length} tokens) → now 1 token.` +
        `<br>Embedding: ${esc(r.embedding || "")}.`, "ok");
    } else {
      setStatus($("addStatus"), `<code>${esc(r.token)}</code> is ${esc(r.status)} (id ${r.id}).` +
        (r.embedding ? ` Embedding ${esc(r.embedding)}.` : ""), "ok");
    }
    $("newWord").value = "";
    $("examples").value = "";
    if (!$("input").value.toLowerCase().includes(word.toLowerCase())) $("input").value += " " + word;
    selected = { token: r.token, neighbours: [] };
    await refresh();
    await loadCorpus();
  } catch (err) {
    setStatus($("addStatus"), "✗ " + esc(err.message), "err");
  } finally {
    btn.disabled = false;
  }
}

$("addBtn").addEventListener("click", addWord);
$("newWord").addEventListener("keydown", (e) => { if (e.key === "Enter") addWord(); });
$("addedList").addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  await api("/api/remove_word", { token: b.dataset.token });
  if (selected && selected.token === b.dataset.token) selected = null;
  setStatus($("addStatus"), `Removed <code>${esc(b.dataset.token)}</code>.`);
  await refresh();
});

// ------------------------------------------------------------------ vocabulary table
function renderVocab() {
  const q = $("vocabSearch").value.trim().toLowerCase();
  const sort = $("vocabSort").value;
  let rows = state.vocab.filter((v) => !q || v.token.toLowerCase().includes(q) || String(v.id) === q);
  if (sort === "freq") rows = [...rows].sort((a, b) => b.freq - a.freq || a.id - b.id);
  if (sort === "len") rows = [...rows].sort((a, b) => b.token.length - a.token.length || a.id - b.id);
  const shown = rows.slice(0, 1500);
  $("vocabBody").innerHTML = shown.map((v) =>
    `<tr data-token="${esc(v.token)}"><td>${v.id}</td><td class="tok">${esc(pretty(v.token))}</td>` +
    `<td><span class="tag" style="background:${TYPE_COLORS[v.type]}">${TYPE_NAMES[v.type]}</span></td><td>${v.freq}</td></tr>`).join("");
  $("vocabCount").textContent = `${rows.length} of ${state.vocab.length} tokens` + (rows.length > shown.length ? ` (showing first ${shown.length})` : "");
}

$("vocabSearch").addEventListener("input", renderVocab);
$("vocabSort").addEventListener("change", renderVocab);
$("vocabBody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr");
  if (tr) { select(tr.dataset.token); $("embed").scrollIntoView({ behavior: "smooth", block: "start" }); }
});

// ------------------------------------------------------------------ start
$("input").addEventListener("input", encodeSoon);
$("showIds").addEventListener("change", renderChips);
$("showSpaces").addEventListener("change", renderChips);

(async () => {
  try {
    await Promise.all([refresh(), loadCorpus()]);
    attachPlotEvents();
  } catch (err) {
    document.body.insertAdjacentHTML("afterbegin", `<p style="color:#ff7a7a;padding:16px">Could not reach the server: ${esc(err.message)}. Is <code>python3 app.py</code> running?</p>`);
  }
})();
