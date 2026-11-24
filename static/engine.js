// engine.js - the same BPE tokenizer and embedding model as tokenizer.py and
// embeddings.py, ported to JavaScript so the whole app can run in the browser
// with no Python server (for example when it is hosted on Vercel).
//
// LocalBackend answers the same /api/... requests that app.py answers, so the
// page (app.js) works the same way in both modes. In the browser, your
// vocabulary and added words are saved in localStorage.
(function (root) {
  "use strict";

  const SPACE = "▁";
  const SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"];
  const UNK = "<unk>", BOS = "<bos>", EOS = "<eos>";
  const PRETOKEN_RE = /[\p{L}\p{M}\p{N}_]+|[^\p{L}\p{M}\p{N}_\s]/gu;
  const ASCII_BASE = "abcdefghijklmnopqrstuvwxyz0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  const ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  const pairKey = (a, b) => a + "\u0000" + b;
  const tokenToText = (t) => t.replaceAll(SPACE, " ").trim();
  const countWords = (s) => s.split(/\s+/).filter(Boolean).length;

  function mergePair(symbols, a, b) {
    const out = [];
    for (let i = 0; i < symbols.length;) {
      if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
        out.push(a + b);
        i += 2;
      } else {
        out.push(symbols[i]);
        i += 1;
      }
    }
    return out;
  }

  // ===================================================================== tokenizer
  class BPETokenizer {
    constructor(lowercase = true) {
      this.lowercase = lowercase;
      this.merges = [];
      this.mergeRanks = new Map();
      this.idToToken = [...SPECIAL_TOKENS];
      this.tokenToId = new Map(this.idToToken.map((t, i) => [t, i]));
      this.addedTokens = [];
      this.addedIndex = new Map();
      this.cache = new Map();
    }

    get vocabSize() { return this.idToToken.length; }

    preTokenize(text) {
      if (this.lowercase) text = text.toLowerCase();
      const words = [];
      for (const m of text.matchAll(PRETOKEN_RE)) {
        const spaced = m.index === 0 || /\s/.test(text[m.index - 1]);
        words.push((spaced ? SPACE : "") + m[0]);
      }
      return words;
    }

    tokenType(token) {
      if (SPECIAL_TOKENS.includes(token)) return "special";
      if (this.addedTokens.includes(token)) return "added";
      const core = token.startsWith(SPACE) ? token.slice(1) : token;
      if (Array.from(core).length <= 1) return "char";
      return token.startsWith(SPACE) ? "word" : "subword";
    }

    // ---- build vocabulary
    train(corpus, vocabSize = 500, minFrequency = 2) {
      const keepAdded = this.addedTokens.map(tokenToText);

      const wordFreq = new Map();
      for (const w of this.preTokenize(corpus)) wordFreq.set(w, (wordFreq.get(w) || 0) + 1);
      const splits = new Map([...wordFreq.keys()].map((w) => [w, Array.from(w)]));

      const alphaSet = new Set([SPACE, ...ASCII_BASE, ...(this.lowercase ? "" : ASCII_UPPER)]);
      for (const w of wordFreq.keys()) for (const ch of w) alphaSet.add(ch);
      const alphabet = [...alphaSet].sort();

      const vocab = [...SPECIAL_TOKENS, ...alphabet];
      const seen = new Set(vocab);
      const merges = [];

      while (vocab.length < vocabSize) {
        const counts = new Map();
        for (const [w, f] of wordFreq) {
          const s = splits.get(w);
          for (let i = 0; i < s.length - 1; i++) {
            const k = pairKey(s[i], s[i + 1]);
            const e = counts.get(k);
            if (e) e.count += f;
            else counts.set(k, { a: s[i], b: s[i + 1], count: f });
          }
        }
        if (!counts.size) break;
        // most frequent pair; ties broken alphabetically (same rule as tokenizer.py)
        let best = null, bestKey = null;
        for (const [k, e] of counts) {
          if (!best || e.count > best.count || (e.count === best.count && k < bestKey)) {
            best = e;
            bestKey = k;
          }
        }
        if (best.count < minFrequency) break;
        merges.push([best.a, best.b]);
        const nt = best.a + best.b;
        if (!seen.has(nt)) { vocab.push(nt); seen.add(nt); }
        for (const [w, s] of splits) if (s.length > 1) splits.set(w, mergePair(s, best.a, best.b));
      }

      this.merges = merges;
      this.mergeRanks = new Map(merges.map(([a, b], r) => [pairKey(a, b), r]));
      this.idToToken = vocab;
      this.tokenToId = new Map(vocab.map((t, i) => [t, i]));
      this.addedTokens = [];
      this.addedIndex = new Map();
      this.cache = new Map();
      for (const t of keepAdded) this.addWord(t);

      return {
        unique_words: wordFreq.size, alphabet: alphabet.length, merges: merges.length,
        vocab_size: this.vocabSize, stopped_early: vocab.length < vocabSize,
      };
    }

    // ---- encode
    bpe(word) {
      const cached = this.cache.get(word);
      if (cached) return cached;
      let symbols = Array.from(word);
      while (symbols.length > 1) {
        let bestRank = Infinity, bestI = -1;
        for (let i = 0; i < symbols.length - 1; i++) {
          const r = this.mergeRanks.get(pairKey(symbols[i], symbols[i + 1]));
          if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
        }
        if (bestI < 0) break;
        symbols = mergePair(symbols, symbols[bestI], symbols[bestI + 1]);
      }
      this.cache.set(word, symbols);
      return symbols;
    }

    tokenize(text, useAdded = true) {
      const pre = this.preTokenize(text);
      const tokens = [];
      for (let i = 0; i < pre.length;) {
        if (useAdded) {
          const match = (this.addedIndex.get(pre[i]) || []).find(
            ([seq]) => seq.every((p, j) => pre[i + j] === p));
          if (match) {
            tokens.push(match[1]);
            i += match[0].length;
            continue;
          }
        }
        tokens.push(...this.bpe(pre[i]));
        i += 1;
      }
      return tokens;
    }

    encode(text, addSpecialTokens = false) {
      const unk = this.tokenToId.get(UNK);
      const ids = this.tokenize(text).map((t) => this.tokenToId.get(t) ?? unk);
      return addSpecialTokens ? [this.tokenToId.get(BOS), ...ids, this.tokenToId.get(EOS)] : ids;
    }

    encodeDetailed(text) {
      const unk = this.tokenToId.get(UNK);
      return this.tokenize(text).map((t) => {
        const id = this.tokenToId.get(t);
        return { token: t, id: id === undefined ? unk : id, unknown: id === undefined };
      });
    }

    // ---- decode
    decode(ids, skipSpecialTokens = true) {
      const pieces = [];
      for (const i of ids) {
        const tok = i >= 0 && i < this.vocabSize ? this.idToToken[i] : UNK;
        if (skipSpecialTokens && SPECIAL_TOKENS.includes(tok) && tok !== UNK) continue;
        pieces.push(tok);
      }
      const text = pieces.join("").replaceAll(SPACE, " ");
      return text.startsWith(" ") ? text.slice(1) : text;
    }

    // ---- add new words
    rebuildAddedIndex() {
      this.addedIndex = new Map();
      for (const tok of this.addedTokens) {
        const seq = this.preTokenize(tokenToText(tok));
        if (!this.addedIndex.has(seq[0])) this.addedIndex.set(seq[0], []);
        this.addedIndex.get(seq[0]).push([seq, tok]);
      }
      for (const entries of this.addedIndex.values()) entries.sort((x, y) => y[0].length - x[0].length);
    }

    addWord(word) {
      word = word.split(/\s+/).filter(Boolean).join(" ");
      const pre = this.preTokenize(word);
      if (!pre.length) throw new Error("Please enter a word that contains letters, digits or symbols.");
      const token = pre.join("");
      const before = this.tokenize(word, false);
      if (this.tokenToId.has(token)) {
        return {
          token, id: this.tokenToId.get(token), before,
          status: this.addedTokens.includes(token) ? "added already" : "already in vocabulary",
        };
      }
      this.tokenToId.set(token, this.idToToken.length);
      this.idToToken.push(token);
      this.addedTokens.push(token);
      this.rebuildAddedIndex();
      return { token, id: this.tokenToId.get(token), before, status: "added" };
    }

    removeWord(tokenOrWord) {
      const token = this.addedTokens.includes(tokenOrWord) ? tokenOrWord : this.preTokenize(tokenOrWord).join("");
      if (!this.addedTokens.includes(token)) return false;
      this.addedTokens = this.addedTokens.filter((t) => t !== token);
      this.idToToken = this.idToToken.filter((t) => t !== token);
      this.tokenToId = new Map(this.idToToken.map((t, i) => [t, i]));
      this.rebuildAddedIndex();
      return true;
    }

    // ---- save / load (same JSON format as tokenizer.py)
    toDict() {
      return {
        type: "BPE", lowercase: this.lowercase, special_tokens: SPECIAL_TOKENS,
        vocab: this.idToToken, merges: this.merges, added_tokens: this.addedTokens,
      };
    }

    static fromDict(d) {
      const tok = new BPETokenizer(d.lowercase ?? true);
      tok.merges = d.merges.map(([a, b]) => [a, b]);
      tok.mergeRanks = new Map(tok.merges.map(([a, b], r) => [pairKey(a, b), r]));
      tok.idToToken = [...d.vocab];
      tok.tokenToId = new Map(tok.idToToken.map((t, i) => [t, i]));
      tok.addedTokens = [...(d.added_tokens || [])];
      tok.rebuildAddedIndex();
      return tok;
    }
  }

  // ============================================================ linear algebra
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const normalize = (v) => { const n = Math.sqrt(dot(v, v)); return n > 1e-12 ? v.map((x) => x / n) : v; };
  const mean = (vs) => vs[0].map((_, c) => vs.reduce((s, v) => s + v[c], 0) / vs.length);

  function orthonormalize(cols) {
    const out = [];
    for (const col of cols) {
      let v = col.slice();
      for (const q of out) {
        const d = dot(v, q);
        v = v.map((x, i) => x - d * q[i]);
      }
      const n = Math.sqrt(dot(v, v));
      if (n > 1e-10) out.push(v.map((x) => x / n));
    }
    return out;
  }

  function jacobiEigen(input, sweeps = 60) {
    const n = input.length;
    const a = input.map((r) => r.slice());
    const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let s = 0; s < sweeps; s++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-18) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-15) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
          for (let k = 0; k < n; k++) {
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - sn * akq; a[k][q] = sn * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - sn * aqk; a[q][k] = sn * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = v[k][p], vkq = v[k][q];
            v[k][p] = c * vkp - sn * vkq; v[k][q] = sn * vkp + c * vkq;
          }
        }
      }
    }
    return { values: a.map((r, i) => r[i]), vectors: Array.from({ length: n }, (_, c) => v.map((r) => r[c])) };
  }

  function seededGauss(seed) {
    let s = seed >>> 0;
    const rand = () => { // mulberry32
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    return () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  }

  // ============================================================ embeddings
  class EmbeddingModel {
    constructor() {
      this.dim = 0;
      this.vectors = new Map(); // token -> unit vector
      this.source = new Map();  // token -> "trained" | "inferred" | "from pieces"
      this.freq = new Map();    // token -> count in encoded corpus
      this.pcaMean = [];
      this.pcaAxes = [];
      this.scale = 1;
    }

    train(tokenizer, corpus, dim = 16, window = 3, iterations = 15, seed = 0) {
      const gauss = seededGauss(seed);

      // 1) co-occurrence counts inside a sliding window (closer = counts more)
      const freq = new Map(), cooc = new Map();
      const add = (t, u, w) => {
        let m = cooc.get(t);
        if (!m) cooc.set(t, (m = new Map()));
        m.set(u, (m.get(u) || 0) + w);
      };
      for (const line of corpus.split(/\r?\n/)) {
        const toks = tokenizer.tokenize(line);
        for (const t of toks) freq.set(t, (freq.get(t) || 0) + 1);
        for (let i = 0; i < toks.length; i++) {
          for (let d = 1; d <= window && i + d < toks.length; d++) {
            add(toks[i], toks[i + d], 1 / d);
            add(toks[i + d], toks[i], 1 / d);
          }
        }
      }
      this.freq = freq;

      // 2) PPMI matrix in sparse row form
      const active = [...cooc.keys()].sort((a, b) => freq.get(b) - freq.get(a));
      const index = new Map(active.map((t, i) => [t, i]));
      const rowSum = new Map(active.map((t) => [t, [...cooc.get(t).values()].reduce((s, x) => s + x, 0)]));
      const total = [...rowSum.values()].reduce((s, x) => s + x, 0);
      const rows = active.map((t) => {
        const j = [], v = [];
        for (const [u, c] of cooc.get(t)) {
          const pmi = Math.log((c * total) / (rowSum.get(t) * rowSum.get(u)));
          if (pmi > 0) { j.push(index.get(u)); v.push(pmi); }
        }
        return { j, v };
      });

      const n = active.length;
      if (n < 4) throw new Error("The corpus is too small to learn embeddings - add more text.");
      const k = Math.max(3, Math.min(dim, n - 1));
      const matmul = (cols) => cols.map((col) => rows.map((r) => {
        let s = 0;
        for (let x = 0; x < r.j.length; x++) s += r.v[x] * col[r.j[x]];
        return s;
      }));

      // 3) truncated SVD by randomised subspace iteration
      let q = orthonormalize(Array.from({ length: k }, () => Array.from({ length: n }, gauss)));
      for (let it = 0; it < iterations; it++) q = orthonormalize(matmul(q));
      const proj = matmul(q);
      this.dim = proj.length;

      this.vectors = new Map();
      this.source = new Map();
      active.forEach((t, i) => {
        const vec = proj.map((col) => col[i]);
        if (vec.some((x) => Math.abs(x) > 1e-12)) {
          this.vectors.set(t, normalize(vec));
          this.source.set(t, "trained");
        }
      });

      // 4) tokens that never appear on their own get an inferred embedding
      const trained = [...this.vectors.keys()];
      for (const t of tokenizer.idToToken) {
        if (this.vectors.has(t) || SPECIAL_TOKENS.includes(t) || tokenizer.addedTokens.includes(t)) continue;
        const containing = trained.filter((u) => u !== t && u.includes(t)).map((u) => this.vectors.get(u));
        if (containing.length) {
          this.vectors.set(t, normalize(mean(containing)));
          this.source.set(t, "inferred");
        }
      }
      for (const t of tokenizer.addedTokens) if (!this.vectors.has(t)) this.embedNewToken(tokenizer, t);

      // 5) PCA -> 3D
      this.fitPca();
      return { tokens_with_embeddings: this.vectors.size, trained: trained.length, dim: this.dim, co_occurring_tokens: n };
    }

    fitPca() {
      const vecs = [...this.vectors.values()];
      this.pcaMean = mean(vecs);
      const centered = vecs.map((v) => v.map((x, i) => x - this.pcaMean[i]));
      const k = this.dim;
      const cov = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) =>
        centered.reduce((s, r) => s + r[i] * r[j], 0) / centered.length));
      const { values, vectors } = jacobiEigen(cov);
      const order = values.map((_, i) => i).sort((a, b) => values[b] - values[a]).slice(0, 3);
      this.pcaAxes = order.map((i) => vectors[i]);
      // scale so 95% of points fall inside [-1, 1]
      const mags = vecs.flatMap((v) => this.projectRaw(v).map(Math.abs)).sort((a, b) => a - b);
      this.scale = mags[Math.floor(0.95 * (mags.length - 1))] || 1;
    }

    projectRaw(v) {
      const c = v.map((x, i) => x - this.pcaMean[i]);
      return this.pcaAxes.map((axis) => dot(c, axis));
    }

    coords(token) {
      const v = this.vectors.get(token);
      if (!v || !this.pcaAxes.length) return null;
      return this.projectRaw(v).map((x) => Math.round((x / this.scale) * 1e4) / 1e4);
    }

    embedNewToken(tokenizer, token) {
      const pieces = tokenizer.tokenize(tokenToText(token), false);
      const known = pieces.filter((p) => this.vectors.has(p)).map((p) => this.vectors.get(p));
      if (known.length) {
        this.vectors.set(token, normalize(mean(known)));
        this.source.set(token, "from pieces");
      }
      return pieces;
    }

    forget(token) {
      this.vectors.delete(token);
      this.source.delete(token);
    }

    neighbours(token, top = 10) {
      const v = this.vectors.get(token);
      if (!v) return [];
      const scored = [];
      for (const [u, w] of this.vectors) if (u !== token) scored.push([dot(v, w), u]);
      scored.sort((a, b) => b[0] - a[0]);
      return scored.slice(0, top).map(([s, u]) => ({ token: u, similarity: Math.round(s * 1000) / 1000 }));
    }

    toDict() {
      return {
        dim: this.dim, vectors: [...this.vectors], source: [...this.source], freq: [...this.freq],
        pca_mean: this.pcaMean, pca_axes: this.pcaAxes, scale: this.scale,
      };
    }

    static fromDict(d) {
      const m = new EmbeddingModel();
      m.dim = d.dim;
      m.vectors = new Map(d.vectors);
      m.source = new Map(d.source);
      m.freq = new Map(d.freq);
      m.pcaMean = d.pca_mean;
      m.pcaAxes = d.pca_axes;
      m.scale = d.scale;
      return m;
    }
  }

  // ============================================================ in-browser backend
  const STORE_KEY = "tokenizer-lab-state-v1";
  const DEFAULT_SETTINGS = { vocab_size: 500, min_frequency: 2, lowercase: true, dim: 16, window: 3 };
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  class LocalBackend {
    constructor(corpusUrl = "data/corpus.txt") {
      this.corpusUrl = corpusUrl;
      this.settings = { ...DEFAULT_SETTINGS };
      this.corpus = "";
      this.tokenizer = new BPETokenizer();
      this.embeddings = new EmbeddingModel();
      this.lastBuild = {};
    }

    async init() {
      if (this.load()) return;
      await this.buildFromSample();
    }

    async buildFromSample() {
      const res = await fetch(this.corpusUrl);
      if (!res.ok) throw new Error("Could not load " + this.corpusUrl);
      this.build(await res.text(), {});
    }

    // ---- persistence (per browser, via localStorage)
    load() {
      try {
        const raw = localStorage.getItem(STORE_KEY);
        if (!raw) return false;
        const d = JSON.parse(raw);
        this.settings = { ...DEFAULT_SETTINGS, ...d.settings };
        this.corpus = d.corpus;
        this.tokenizer = BPETokenizer.fromDict(d.tokenizer);
        this.embeddings = EmbeddingModel.fromDict(d.embeddings);
        this.lastBuild = d.lastBuild || {};
        return true;
      } catch (e) {
        return false;
      }
    }

    save() {
      try {
        localStorage.setItem(STORE_KEY, JSON.stringify({
          settings: this.settings, corpus: this.corpus, lastBuild: this.lastBuild,
          tokenizer: this.tokenizer.toDict(), embeddings: this.embeddings.toDict(),
        }));
      } catch (e) { /* storage full or blocked: the app still works, it just won't remember */ }
    }

    // ---- actions (mirror app.py)
    build(corpus, options) {
      if (countWords(corpus) < 20) throw new Error("Please give at least 20 words of text to build a vocabulary from.");
      const s = { ...this.settings };
      for (const key of Object.keys(DEFAULT_SETTINGS)) {
        if (key in options) s[key] = typeof DEFAULT_SETTINGS[key] === "boolean" ? Boolean(options[key]) : Math.trunc(Number(options[key]) || 0);
      }
      s.vocab_size = clamp(s.vocab_size, 40, 5000);
      s.dim = clamp(s.dim, 3, 64);
      s.window = clamp(s.window, 1, 10);
      s.min_frequency = Math.max(1, s.min_frequency);

      const tok = new BPETokenizer(s.lowercase);
      tok.addedTokens = [...this.tokenizer.addedTokens];
      const bpeStats = tok.train(corpus, s.vocab_size, s.min_frequency);
      const emb = new EmbeddingModel();
      const embStats = emb.train(tok, corpus, s.dim, s.window);

      Object.assign(this, { tokenizer: tok, embeddings: emb, corpus, settings: s });
      this.lastBuild = { ...bpeStats, ...embStats };
      this.save();
      return this.lastBuild;
    }

    retrainEmbeddings() {
      const emb = new EmbeddingModel();
      const stats = emb.train(this.tokenizer, this.corpus, this.settings.dim, this.settings.window);
      this.embeddings = emb;
      this.save();
      return stats;
    }

    addWord(word, examples) {
      const result = this.tokenizer.addWord(word);
      const token = result.token;
      if (examples.trim()) {
        this.corpus = this.corpus.replace(/\n+$/, "") + "\n\n" + examples.trim() + "\n";
        this.retrainEmbeddings();
        result.embedding = "learned from your example sentences";
      } else if (result.status === "added") {
        const pieces = this.embeddings.embedNewToken(this.tokenizer, token);
        result.embedding = this.embeddings.vectors.has(token)
          ? "average of its old pieces " + pieces.join(" + ")
          : "none (pieces have no embeddings)";
      }
      this.save();
      result.coords = this.embeddings.coords(token);
      return result;
    }

    removeWord(token) {
      const ok = this.tokenizer.removeWord(token);
      if (ok) {
        this.embeddings.forget(token);
        this.save();
      }
      return ok;
    }

    async reset() {
      try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
      this.settings = { ...DEFAULT_SETTINGS };
      this.tokenizer = new BPETokenizer();
      await this.buildFromSample();
      return this.lastBuild;
    }

    // ---- views
    summary() {
      const tok = this.tokenizer, emb = this.embeddings;
      const vocab = [], points = [];
      tok.idToToken.forEach((t, i) => {
        const type = tok.tokenType(t);
        const freq = emb.freq.get(t) || 0;
        vocab.push({ id: i, token: t, type, freq });
        const c = emb.coords(t);
        if (c) points.push({ id: i, token: t, type, freq, source: emb.source.get(t) || "", x: c[0], y: c[1], z: c[2] });
      });
      return {
        settings: this.settings,
        stats: {
          vocab_size: tok.vocabSize, merges: tok.merges.length, added: tok.addedTokens.length,
          special: SPECIAL_TOKENS.length, corpus_words: countWords(this.corpus), corpus_chars: this.corpus.length,
          embedding_dim: emb.dim, embedded_tokens: points.length,
        },
        last_build: this.lastBuild,
        added: tok.addedTokens.map((t) => ({ token: t, id: tok.tokenToId.get(t), word: tokenToText(t) })),
        vocab, points,
      };
    }

    encode(text) {
      const tok = this.tokenizer;
      const tokens = tok.encodeDetailed(text);
      for (const d of tokens) d.type = tok.tokenType(d.token);
      const ids = tokens.map((d) => d.id);
      return { tokens, ids, decoded: tok.decode(ids), characters: text.length, words: countWords(text) };
    }

    // ---- the same API app.py serves
    async handle(path, body) {
      const url = new URL(path, "http://local");
      const b = body || {};
      switch (url.pathname) {
        case "/api/state": return this.summary();
        case "/api/corpus": return { corpus: this.corpus };
        case "/api/neighbours": {
          const token = url.searchParams.get("token") || "";
          return { token, coords: this.embeddings.coords(token), source: this.embeddings.source.get(token) ?? null,
                   neighbours: this.embeddings.neighbours(token, 10) };
        }
        case "/api/encode": return this.encode(String(b.text ?? ""));
        case "/api/build": return { ok: true, build: this.build(String(b.corpus ?? ""), b.options || {}) };
        case "/api/add_word": return this.addWord(String(b.word ?? ""), String(b.examples ?? ""));
        case "/api/remove_word": return { ok: this.removeWord(String(b.token ?? "")) };
        case "/api/retrain_embeddings": return { ok: true, build: this.retrainEmbeddings() };
        case "/api/reset": return { ok: true, build: await this.reset() };
        default: throw new Error("unknown endpoint " + url.pathname);
      }
    }
  }

  const api = { SPACE, SPECIAL_TOKENS, BPETokenizer, EmbeddingModel, LocalBackend };
  root.TokenizerEngine = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
