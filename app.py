"""
app.py - a small web app around the tokenizer (standard library only).

    python3 app.py            # then open http://localhost:8000
    python3 app.py 8080       # different port

The tokenizer, the corpus and the embeddings are saved in the model/ folder
after every change, so your vocabulary and added words survive a restart.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from embeddings import EmbeddingModel
from tokenizer import SPACE, SPECIAL_TOKENS, BPETokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                 ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}
MODEL_DIR = os.path.join(ROOT, "model")
DEFAULT_CORPUS = os.path.join(ROOT, "data", "corpus.txt")

TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "embeddings.json")
CORPUS_FILE = os.path.join(MODEL_DIR, "corpus.txt")
SETTINGS_FILE = os.path.join(MODEL_DIR, "settings.json")

DEFAULT_SETTINGS = {"vocab_size": 500, "min_frequency": 2, "lowercase": True, "dim": 16, "window": 3}


class State:
    """Everything the app knows: tokenizer + corpus + embeddings."""

    def __init__(self):
        self.lock = threading.Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.corpus = ""
        self.tokenizer = BPETokenizer()
        self.embeddings = EmbeddingModel()
        self.last_build: dict = {}

    # ------------------------------------------------------------ persistence
    def load_or_build(self) -> None:
        if all(os.path.exists(p) for p in (TOKENIZER_FILE, EMBEDDINGS_FILE, CORPUS_FILE)):
            self.tokenizer = BPETokenizer.load(TOKENIZER_FILE)
            with open(EMBEDDINGS_FILE, encoding="utf-8") as f:
                self.embeddings = EmbeddingModel.from_dict(json.load(f))
            with open(CORPUS_FILE, encoding="utf-8") as f:
                self.corpus = f.read()
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, encoding="utf-8") as f:
                    self.settings.update(json.load(f))
            print(f"Loaded saved tokenizer ({self.tokenizer.vocab_size} tokens) from model/")
        else:
            with open(DEFAULT_CORPUS, encoding="utf-8") as f:
                self.build(f.read(), {})
            print(f"Built a new vocabulary ({self.tokenizer.vocab_size} tokens) from data/corpus.txt")

    def save(self) -> None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.tokenizer.save(TOKENIZER_FILE)
        with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.embeddings.to_dict(), f, ensure_ascii=False)
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            f.write(self.corpus)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=1)

    # ------------------------------------------------------------- actions
    def build(self, corpus: str, options: dict) -> dict:
        if len(corpus.split()) < 20:
            raise ValueError("Please give at least 20 words of text to build a vocabulary from.")
        s = dict(self.settings)
        for key in DEFAULT_SETTINGS:
            if key in options:
                s[key] = type(DEFAULT_SETTINGS[key])(options[key])
        s["vocab_size"] = max(40, min(5000, s["vocab_size"]))
        s["dim"] = max(3, min(64, s["dim"]))
        s["window"] = max(1, min(10, s["window"]))
        s["min_frequency"] = max(1, s["min_frequency"])

        tok = BPETokenizer(lowercase=s["lowercase"])
        tok.added_tokens = list(self.tokenizer.added_tokens)  # keep the user's added words
        bpe_stats = tok.train(corpus, vocab_size=s["vocab_size"], min_frequency=s["min_frequency"])
        emb = EmbeddingModel()
        emb_stats = emb.train(tok, corpus, dim=s["dim"], window=s["window"])

        self.tokenizer, self.embeddings, self.corpus, self.settings = tok, emb, corpus, s
        self.last_build = {**bpe_stats, **emb_stats}
        self.save()
        return self.last_build

    def retrain_embeddings(self) -> dict:
        emb = EmbeddingModel()
        stats = emb.train(self.tokenizer, self.corpus, dim=self.settings["dim"], window=self.settings["window"])
        self.embeddings = emb
        self.save()
        return stats

    def add_word(self, word: str, examples: str) -> dict:
        result = self.tokenizer.add_word(word)
        token = result["token"]
        if examples.strip():
            # learn from real context: add the sentences to the corpus and retrain embeddings
            self.corpus = self.corpus.rstrip("\n") + "\n\n" + examples.strip() + "\n"
            self.retrain_embeddings()
            result["embedding"] = "learned from your example sentences"
        elif result["status"] == "added":
            pieces = self.embeddings.embed_new_token(self.tokenizer, token)
            result["embedding"] = ("average of its old pieces " + " + ".join(pieces)
                                   if token in self.embeddings.vectors else "none (pieces have no embeddings)")
        self.save()
        result["coords"] = self.embeddings.coords(token)
        return result

    def reset(self) -> dict:
        """Forget added words and settings, rebuild from the sample corpus."""
        self.tokenizer = BPETokenizer()
        self.settings = dict(DEFAULT_SETTINGS)
        with open(DEFAULT_CORPUS, encoding="utf-8") as f:
            return self.build(f.read(), {})

    def remove_word(self, token: str) -> bool:
        ok = self.tokenizer.remove_word(token)
        if ok:
            self.embeddings.forget(token)
            self.save()
        return ok

    # ------------------------------------------------------------- views
    def summary(self) -> dict:
        tok, emb = self.tokenizer, self.embeddings
        vocab, points = [], []
        for i, t in enumerate(tok.id_to_token):
            kind = tok.token_type(t)
            freq = emb.freq.get(t, 0)
            vocab.append({"id": i, "token": t, "type": kind, "freq": freq})
            c = emb.coords(t)
            if c:
                points.append({"id": i, "token": t, "type": kind, "freq": freq,
                               "source": emb.source.get(t, ""), "x": c[0], "y": c[1], "z": c[2]})
        return {
            "settings": self.settings,
            "stats": {
                "vocab_size": tok.vocab_size,
                "merges": len(tok.merges),
                "added": len(tok.added_tokens),
                "special": len(SPECIAL_TOKENS),
                "corpus_words": len(self.corpus.split()),
                "corpus_chars": len(self.corpus),
                "embedding_dim": emb.dim,
                "embedded_tokens": len(points),
            },
            "last_build": self.last_build,
            "added": [{"token": t, "id": tok.token_to_id[t], "word": t.replace(SPACE, " ").strip()}
                      for t in tok.added_tokens],
            "vocab": vocab,
            "points": points,
        }

    def encode(self, text: str) -> dict:
        tok = self.tokenizer
        detailed = tok.encode_detailed(text)
        ids = [d["id"] for d in detailed]
        for d in detailed:
            d["type"] = tok.token_type(d["token"])
        return {
            "tokens": detailed,
            "ids": ids,
            "decoded": tok.decode(ids),
            "characters": len(text),
            "words": len(text.split()),
        }


STATE = State()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        with STATE.lock:
            if url.path == "/api/state":
                return self._json(STATE.summary())
            if url.path == "/api/corpus":
                return self._json({"corpus": STATE.corpus})
            if url.path == "/api/neighbours":
                token = q.get("token", [""])[0]
                return self._json({"token": token, "coords": STATE.embeddings.coords(token),
                                   "source": STATE.embeddings.source.get(token),
                                   "neighbours": STATE.embeddings.neighbours(token, 10)})
            if url.path == "/api/export":
                body = json.dumps(STATE.tokenizer.to_dict(), ensure_ascii=False, indent=1).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="tokenizer.json"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        # static files: the page itself plus anything in static/ and data/
        rel = "index.html" if url.path in ("/", "/index.html") else url.path.lstrip("/")
        full = os.path.normpath(os.path.join(ROOT, rel))
        allowed = full == os.path.join(ROOT, "index.html") or any(
            full.startswith(os.path.join(ROOT, d) + os.sep) for d in ("static", "data"))
        ctype = CONTENT_TYPES.get(os.path.splitext(full)[1])
        if not (allowed and ctype and os.path.isfile(full)):
            return self._send(404, b"Not found", "text/plain")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self._body()
            with STATE.lock:
                if path == "/api/encode":
                    return self._json(STATE.encode(str(data.get("text", ""))))
                if path == "/api/build":
                    stats = STATE.build(str(data.get("corpus", "")), data.get("options", {}))
                    return self._json({"ok": True, "build": stats})
                if path == "/api/add_word":
                    return self._json(STATE.add_word(str(data.get("word", "")), str(data.get("examples", ""))))
                if path == "/api/remove_word":
                    return self._json({"ok": STATE.remove_word(str(data.get("token", "")))})
                if path == "/api/reset":
                    return self._json({"ok": True, "build": STATE.reset()})
                if path == "/api/retrain_embeddings":
                    return self._json({"ok": True, "build": STATE.retrain_embeddings()})
            self._json({"error": "unknown endpoint"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # report unexpected problems to the page instead of hanging
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    STATE.load_or_build()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Tokenizer Lab running at http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
