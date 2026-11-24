# Tokenizer Lab: an LLM tokenizer from scratch

A Byte-Pair-Encoding (BPE) tokenizer (the kind GPT and LLaMA use) written in plain Python, with a web app around it.
It needs **no installs**: only the Python standard library. The 3D plot loads Plotly from a CDN, so you need internet access.

## Run it

```bash
python3 app.py
```

Then open <http://localhost:8000>. Use `python3 app.py 8080` to pick a different port.

For a quick command-line demo without the web app:

```bash
python3 tokenizer.py
```

## What's inside

| Feature | Where | How it works |
|---|---|---|
| **1. Tokeniser** | `tokenizer.py` → `tokenize()`, `encode()`, `decode()` | Splits text into words, marks word starts with `▁`, then applies the learned BPE merges in order. Shows colour-coded tokens, ids, chars per token and the decoded text. |
| **2. Build vocabulary** | `tokenizer.py` → `train()` | Starts from single characters and keeps merging the most frequent adjacent pair until the vocabulary reaches your chosen size. Paste text or load a `.txt` file in the app. |
| **3. Embedding visualisation in 3D** | `embeddings.py` | Counts which tokens appear near each other, turns the counts into PPMI scores, compresses them with SVD into 16-dimensional vectors, then uses PCA to get 3 axes. Similar tokens end up close together. Click a point to see its nearest neighbours. |
| **4. Add new words** | `tokenizer.py` → `add_word()` | Adds a word or phrase (`transformer`, `new york`, `gpt-4`) as one token. Its embedding starts as the average of the pieces it used to be split into. If you give example sentences, it learns its embedding from context instead. |

Everything is saved in `model/` (tokenizer, corpus, embeddings), so your vocabulary and added words are still there after a restart. Delete `model/` to start over from `data/corpus.txt`.
Use **Export tokenizer.json** in the app to download the vocabulary and merges.

## Use it from Python

```python
from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(open("data/corpus.txt").read(), vocab_size=500)

ids = tok.encode("The queen trained a neural network.")
print(tok.tokenize("The queen trained a neural network."))
print(tok.decode(ids))

tok.add_word("transformer")       # now one token
tok.save("my_tokenizer.json")
tok = BPETokenizer.load("my_tokenizer.json")
```

## Files

```
app.py            web server + JSON API (standard library http.server)
tokenizer.py      BPE tokenizer: train / encode / decode / add_word / save / load
embeddings.py     co-occurrence → PPMI → SVD embeddings → PCA to 3D, nearest neighbours
data/corpus.txt   sample corpus (royalty, animals, food, tech, weather, sports…)
static/           the web page (HTML, CSS, JS with Plotly for 3D)
```
