"""
embeddings.py - learn a vector (embedding) for every token, then squash it to 3D.

How it works (classic count-based word vectors, pure Python):

  1. Encode the corpus with the tokenizer.
  2. Count how often each pair of tokens appears close together (a window of
     a few tokens on each side). Closer neighbours count more.
  3. Turn counts into PPMI (positive pointwise mutual information): "do these
     two tokens appear together more often than chance would predict?"
  4. Compress the big PPMI matrix to `dim` numbers per token with a truncated
     SVD (randomised subspace iteration). Those numbers are the embedding.
  5. Run PCA on the embeddings to get 3 coordinates per token for the plot.

Tokens that never appear on their own in the encoded corpus (for example
"▁th", an in-between merge) get an embedding inferred from the tokens that
contain them. Newly added words get the average embedding of the pieces they
used to be split into - the same trick real LLMs use when their vocabulary is
extended - until you retrain with example sentences.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from tokenizer import SPECIAL_TOKENS, BPETokenizer


# ------------------------------------------------------------- tiny linear algebra
def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else v


def _mean(vectors: list[list[float]]) -> list[float]:
    k = len(vectors[0])
    return [sum(v[c] for v in vectors) / len(vectors) for c in range(k)]


def _orthonormalize(cols: list[list[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt on a list of column vectors."""
    out: list[list[float]] = []
    for col in cols:
        v = col[:]
        for q in out:
            d = sum(a * b for a, b in zip(v, q))
            v = [a - d * b for a, b in zip(v, q)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-10:
            out.append([x / n for x in v])
    return out


def _jacobi_eigen(a: list[list[float]], sweeps: int = 60):
    """Eigen-decomposition of a small symmetric matrix. Returns (values, vectors as columns)."""
    n = len(a)
    a = [row[:] for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off < 1e-18:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-15:
                    continue
                theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    values = [a[i][i] for i in range(n)]
    vectors = [[v[r][c] for r in range(n)] for c in range(n)]
    return values, vectors


# --------------------------------------------------------------------- the model
class EmbeddingModel:
    def __init__(self):
        self.dim = 0
        self.vectors: dict[str, list[float]] = {}   # token -> unit vector
        self.source: dict[str, str] = {}            # token -> "trained" | "inferred" | "from pieces"
        self.freq: dict[str, int] = {}              # token -> count in encoded corpus
        self.pca_mean: list[float] = []
        self.pca_axes: list[list[float]] = []       # 3 principal directions
        self.scale = 1.0

    # ---------------------------------------------------------------- train
    def train(self, tokenizer: BPETokenizer, corpus: str, dim: int = 16,
              window: int = 3, iterations: int = 15, seed: int = 0) -> dict:
        rng = random.Random(seed)

        # 1) encode corpus, count co-occurrences inside a sliding window
        freq: Counter = Counter()
        cooc: dict[str, Counter] = defaultdict(Counter)
        for line in corpus.splitlines():
            toks = tokenizer.tokenize(line)
            freq.update(toks)
            for i, t in enumerate(toks):
                for d in range(1, window + 1):
                    if i + d >= len(toks):
                        break
                    u = toks[i + d]
                    w = 1.0 / d
                    cooc[t][u] += w
                    cooc[u][t] += w
        self.freq = dict(freq)

        # 2) PPMI matrix in sparse row form
        active = sorted(cooc, key=lambda t: -freq[t])
        index = {t: i for i, t in enumerate(active)}
        row_sum = {t: sum(cooc[t].values()) for t in active}
        total = sum(row_sum.values())
        rows: list[list[tuple[int, float]]] = []
        for t in active:
            row = []
            for u, c in cooc[t].items():
                pmi = math.log(c * total / (row_sum[t] * row_sum[u]))
                if pmi > 0:
                    row.append((index[u], pmi))
            rows.append(row)

        n = len(active)
        if n < 4:
            raise ValueError("The corpus is too small to learn embeddings - add more text.")
        k = max(3, min(dim, n - 1))

        def matmul(cols: list[list[float]]) -> list[list[float]]:
            return [[sum(val * col[j] for j, val in row) for row in rows] for col in cols]

        # 3) truncated SVD by randomised subspace iteration
        q = _orthonormalize([[rng.gauss(0, 1) for _ in range(n)] for _ in range(k)])
        for _ in range(iterations):
            q = _orthonormalize(matmul(q))
        proj = matmul(q)  # n x k (stored as k columns)
        self.dim = len(proj)

        self.vectors, self.source = {}, {}
        for i, t in enumerate(active):
            vec = [proj[c][i] for c in range(self.dim)]
            if any(abs(x) > 1e-12 for x in vec):
                self.vectors[t] = _normalize(vec)
                self.source[t] = "trained"

        # 4) fill in tokens that never appear on their own in the encoded corpus
        trained = [t for t in self.vectors]
        for t in tokenizer.id_to_token:
            if t in self.vectors or t in SPECIAL_TOKENS or t in tokenizer.added_tokens:
                continue
            containing = [self.vectors[u] for u in trained if t in u and u != t]
            if containing:
                self.vectors[t] = _normalize(_mean(containing))
                self.source[t] = "inferred"
        for t in tokenizer.added_tokens:
            if t not in self.vectors:
                self.embed_new_token(tokenizer, t, project=False)

        # 5) PCA -> 3D
        self._fit_pca()
        return {"tokens_with_embeddings": len(self.vectors), "trained": len(trained),
                "dim": self.dim, "co_occurring_tokens": n}

    def _fit_pca(self) -> None:
        vecs = list(self.vectors.values())
        self.pca_mean = _mean(vecs)
        centered = [[x - m for x, m in zip(v, self.pca_mean)] for v in vecs]
        k = self.dim
        cov = [[sum(r[i] * r[j] for r in centered) / len(centered) for j in range(k)] for i in range(k)]
        values, vectors = _jacobi_eigen(cov)
        order = sorted(range(k), key=lambda i: -values[i])[:3]
        self.pca_axes = [vectors[i] for i in order]
        coords = [self._project_raw(v) for v in vecs]
        # scale so 95% of points fall inside [-1, 1] (a few outliers shouldn't squash the rest)
        mags = sorted(abs(x) for c in coords for x in c)
        self.scale = mags[int(0.95 * (len(mags) - 1))] or 1.0

    def _project_raw(self, v: list[float]) -> list[float]:
        c = [x - m for x, m in zip(v, self.pca_mean)]
        return [sum(a * b for a, b in zip(c, axis)) for axis in self.pca_axes]

    def coords(self, token: str) -> list[float] | None:
        v = self.vectors.get(token)
        if v is None or not self.pca_axes:
            return None
        return [round(x / self.scale, 4) for x in self._project_raw(v)]

    # -------------------------------------------------------- new tokens
    def embed_new_token(self, tokenizer: BPETokenizer, token: str, project: bool = True) -> list[str]:
        """Give an added word an embedding = mean of the pieces BPE used to split it into."""
        pieces = tokenizer.tokenize(tokenizer._token_to_text(token), use_added=False)
        known = [self.vectors[p] for p in pieces if p in self.vectors]
        if known:
            self.vectors[token] = _normalize(_mean(known))
            self.source[token] = "from pieces"
        return pieces

    def forget(self, token: str) -> None:
        self.vectors.pop(token, None)
        self.source.pop(token, None)

    # -------------------------------------------------------- similarity
    def neighbours(self, token: str, top: int = 10) -> list[dict]:
        v = self.vectors.get(token)
        if v is None:
            return []
        scored = []
        for u, w in self.vectors.items():
            if u != token:
                scored.append((sum(a * b for a, b in zip(v, w)), u))
        scored.sort(reverse=True)
        return [{"token": u, "similarity": round(s, 3)} for s, u in scored[:top]]

    # ---------------------------------------------------------- save/load
    def to_dict(self) -> dict:
        return {"dim": self.dim, "vectors": self.vectors, "source": self.source, "freq": self.freq,
                "pca_mean": self.pca_mean, "pca_axes": self.pca_axes, "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingModel":
        m = cls()
        m.dim = data["dim"]
        m.vectors = data["vectors"]
        m.source = data["source"]
        m.freq = data.get("freq", {})
        m.pca_mean = data["pca_mean"]
        m.pca_axes = data["pca_axes"]
        m.scale = data["scale"]
        return m
