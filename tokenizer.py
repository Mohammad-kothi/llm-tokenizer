"""
tokenizer.py - a Byte-Pair-Encoding (BPE) tokenizer written from scratch.

This is the same family of algorithm used by GPT, LLaMA and most modern LLMs:

  1. Pre-tokenise: split text into words and punctuation. A word that follows
     a space (or starts the text) gets a "▁" prefix, so spaces survive the
     round trip (SentencePiece style).
  2. Build vocabulary (train): start from single characters, then repeatedly
     merge the most frequent adjacent pair of symbols into a new token until
     the vocabulary reaches the size you asked for.
  3. Encode: split each word into characters and replay the learned merges in
     the order they were learned. Map the resulting tokens to integer ids.
  4. Decode: map ids back to token strings, join them, turn "▁" back into " ".

You can also add whole new words (or phrases such as "new york") to the
vocabulary. Added words are matched before BPE runs, so they always come out
as one single token.

Only the Python standard library is used.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

SPACE = "▁"  # marks "this token starts with a space"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = SPECIAL_TOKENS
PRETOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """Replace every adjacent occurrence of `pair` in `symbols` by one merged symbol."""
    a, b = pair
    merged = a + b
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.merges: list[tuple[str, str]] = []     # in the order they were learned
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.id_to_token)}
        self.added_tokens: list[str] = []           # token strings of user-added words
        self._added_index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        self._cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ basics
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def pre_tokenize(self, text: str) -> list[str]:
        """'Hello, world' -> ['▁hello', ',', '▁world']"""
        if self.lowercase:
            text = text.lower()
        words = []
        for m in PRETOKEN_RE.finditer(text):
            start = m.start()
            spaced = start == 0 or text[start - 1].isspace()
            words.append((SPACE if spaced else "") + m.group())
        return words

    def token_type(self, token: str) -> str:
        if token in SPECIAL_TOKENS:
            return "special"
        if token in self.added_tokens:
            return "added"
        core = token[1:] if token.startswith(SPACE) else token
        if len(core) <= 1:
            return "char"
        return "word" if token.startswith(SPACE) else "subword"

    # ---------------------------------------------------------- build vocab
    def train(self, corpus: str, vocab_size: int = 500, min_frequency: int = 2) -> dict:
        """Learn BPE merges from `corpus` until the vocabulary has `vocab_size` tokens.

        Words added with add_word() are kept and re-appended after training.
        Returns a few statistics about the training run.
        """
        keep_added = [self._token_to_text(t) for t in self.added_tokens]

        word_freq = Counter(self.pre_tokenize(corpus))
        splits = {w: list(w) for w in word_freq}  # every word starts as characters
        # base alphabet = every character in the corpus + printable ASCII, so common
        # letters, digits and punctuation never become <unk> even if the corpus lacks them
        ascii_chars = string.ascii_lowercase + string.digits + string.punctuation
        if not self.lowercase:
            ascii_chars += string.ascii_uppercase
        alphabet = sorted({ch for w in word_freq for ch in w} | set(ascii_chars) | {SPACE})

        vocab = list(SPECIAL_TOKENS) + alphabet
        seen = set(vocab)
        merges: list[tuple[str, str]] = []

        while len(vocab) < vocab_size:
            pair_counts: Counter = Counter()
            for w, freq in word_freq.items():
                sym = splits[w]
                for i in range(len(sym) - 1):
                    pair_counts[(sym[i], sym[i + 1])] += freq
            if not pair_counts:
                break  # every word is already a single token
            # most frequent pair; ties broken alphabetically so training is deterministic
            best, count = min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if count < min_frequency:
                break
            merges.append(best)
            new_token = best[0] + best[1]
            if new_token not in seen:
                vocab.append(new_token)
                seen.add(new_token)
            for w in splits:
                if len(splits[w]) > 1:
                    splits[w] = merge_pair(splits[w], best)

        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.id_to_token = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self.added_tokens = []
        self._added_index = {}
        self._cache = {}
        for text in keep_added:
            self.add_word(text)

        return {
            "unique_words": len(word_freq),
            "alphabet": len(alphabet),
            "merges": len(merges),
            "vocab_size": self.vocab_size,
            "stopped_early": len(vocab) < vocab_size,
        }

    # --------------------------------------------------------------- encode
    def _bpe(self, word: str) -> list[str]:
        """Split one pre-token into BPE tokens by replaying merges in learned order."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        while len(symbols) > 1:
            best_rank, best_pair = None, None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (symbols[i], symbols[i + 1])
            if best_pair is None:
                break
            symbols = merge_pair(symbols, best_pair)
        self._cache[word] = symbols
        return symbols

    def tokenize(self, text: str, use_added: bool = True) -> list[str]:
        """Text -> list of token strings (unknown characters are kept as-is)."""
        pre = self.pre_tokenize(text)
        tokens: list[str] = []
        i = 0
        while i < len(pre):
            if use_added:
                match = None
                for seq, tok in self._added_index.get(pre[i], ()):
                    if tuple(pre[i:i + len(seq)]) == seq:
                        match = (seq, tok)
                        break
                if match:
                    tokens.append(match[1])
                    i += len(match[0])
                    continue
            tokens.extend(self._bpe(pre[i]))
            i += 1
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Text -> list of token ids."""
        unk = self.token_to_id[UNK]
        ids = [self.token_to_id.get(t, unk) for t in self.tokenize(text)]
        if add_special_tokens:
            ids = [self.token_to_id[BOS]] + ids + [self.token_to_id[EOS]]
        return ids

    def encode_detailed(self, text: str) -> list[dict]:
        """Like encode() but returns token string, id and whether it was unknown."""
        unk = self.token_to_id[UNK]
        out = []
        for t in self.tokenize(text):
            tid = self.token_to_id.get(t)
            out.append({"token": t, "id": unk if tid is None else tid, "unknown": tid is None})
        return out

    # --------------------------------------------------------------- decode
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for i in ids:
            tok = self.id_to_token[i] if 0 <= i < self.vocab_size else UNK
            if skip_special_tokens and tok in SPECIAL_TOKENS and tok != UNK:
                continue
            pieces.append(tok)
        text = "".join(pieces).replace(SPACE, " ")
        return text[1:] if text.startswith(" ") else text

    # ----------------------------------------------------- add new words
    @staticmethod
    def _token_to_text(token: str) -> str:
        return token.replace(SPACE, " ").strip()

    def _rebuild_added_index(self) -> None:
        self._added_index = {}
        for tok in self.added_tokens:
            seq = tuple(self.pre_tokenize(self._token_to_text(tok)))
            self._added_index.setdefault(seq[0], []).append((seq, tok))
        for entries in self._added_index.values():
            entries.sort(key=lambda e: -len(e[0]))  # longest phrase wins

    def add_word(self, word: str) -> dict:
        """Add a new word or phrase to the vocabulary as a single token.

        Returns the new token, its id and how the word was split before.
        """
        word = " ".join(word.split())
        pre = self.pre_tokenize(word)
        if not pre:
            raise ValueError("Please enter a word that contains letters, digits or symbols.")
        token = "".join(pre)
        before = self.tokenize(word, use_added=False)

        if token in self.token_to_id:
            return {"token": token, "id": self.token_to_id[token], "before": before,
                    "status": "added already" if token in self.added_tokens else "already in vocabulary"}

        self.token_to_id[token] = len(self.id_to_token)
        self.id_to_token.append(token)
        self.added_tokens.append(token)
        self._rebuild_added_index()
        return {"token": token, "id": self.token_to_id[token], "before": before, "status": "added"}

    def remove_word(self, token_or_word: str) -> bool:
        """Remove a previously added word. Ids of later added words shift down by one."""
        token = token_or_word if token_or_word in self.added_tokens else "".join(self.pre_tokenize(token_or_word))
        if token not in self.added_tokens:
            return False
        self.added_tokens.remove(token)
        self.id_to_token.remove(token)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self._rebuild_added_index()
        return True

    # ---------------------------------------------------------- save/load
    def to_dict(self) -> dict:
        return {
            "type": "BPE",
            "lowercase": self.lowercase,
            "special_tokens": SPECIAL_TOKENS,
            "vocab": self.id_to_token,
            "merges": [list(m) for m in self.merges],
            "added_tokens": self.added_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        tok = cls(lowercase=data.get("lowercase", True))
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {pair: rank for rank, pair in enumerate(tok.merges)}
        tok.id_to_token = list(data["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        tok.added_tokens = list(data.get("added_tokens", []))
        tok._rebuild_added_index()
        return tok

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    # Small command-line demo:  python3 tokenizer.py
    import os

    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus.txt")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = f.read()

    tok = BPETokenizer()
    stats = tok.train(corpus, vocab_size=400)
    print(f"Built vocabulary: {stats}")

    text = "The king and the queen trained a neural network."
    print("\nText:   ", text)
    print("Tokens: ", tok.tokenize(text))
    print("Ids:    ", tok.encode(text))
    print("Decoded:", tok.decode(tok.encode(text)))

    print("\nAdding the word 'transformer'...")
    print(tok.add_word("transformer"))
    print("Tokens now:", tok.tokenize("A transformer is a neural network."))
