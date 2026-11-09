"""
Llama 3.2 1B letter prior for the speller, built on the team's word-tree predictor (pytorch.py).

prior_fn(text_so_far) -> 36 probabilities in grid order (A..Z, 1..9, sp).
  1. Finished words become the prompt; the word being typed is the 'partial'.
  2. word_tree(prompt) predicts whole next words with probabilities.
  3. Keep words starting with the partial; the next letter gets the word's probability
     (or 'sp' if the word is already complete).
  4. Mix in 5% uniform so no key is ever ~impossible.
Results are cached by prompt, so Llama only runs when a new word starts.

Setup (once): accept the license at huggingface.co/meta-llama/Llama-3.2-1B, create a Read
token, then run:  python -c "from huggingface_hub import login; login()"
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "meta-llama/Llama-3.2-1B"
TOP_K = 5          # branches per node
MAX_DEPTH = 10     # max tokens per word
MIN_PROB = 0.005   # prune branches below this cumulative probability
FIRST_K = 10       # possibilities on the first token
FLOOR = 0.05       # 5% uniform mix
GRID = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(i) for i in range(1, 10)] + ["sp"]
CACHE = Path("results/llm_word_cache.json")

print("Loading", NAME, "...")
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

# Which token IDs continue a word (start with a letter, no leading space)
V = model.get_output_embeddings().weight.shape[0]
is_cont = torch.zeros(V, dtype=torch.bool)
for i in range(len(tok)):
    s = tok.decode([i])
    is_cont[i] = bool(s) and s[0].isalpha()
is_cont = is_cont.to(model.device)


def next_probs(ids):
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


def expand(ids, word, prob, depth, words):
    probs = next_probs(ids)
    end_p = probs[~is_cont].sum().item()  # chance the word ends here
    words[word] += prob * end_p
    if depth >= MAX_DEPTH:
        return
    cont = probs.masked_fill(~is_cont, 0.0)
    top = torch.topk(cont, TOP_K)
    for p, i in zip(top.values, top.indices):
        child_prob = prob * p.item()
        if child_prob < MIN_PROB:
            continue
        new_ids = torch.cat([ids, i.view(1, 1)], dim=1)
        expand(new_ids, word + tok.decode(i), child_prob, depth + 1, words)


def word_tree(prompt):
    """Next-word predictions {word: probability} for a prompt (team's original logic)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    words = defaultdict(float)
    top = torch.topk(next_probs(ids), FIRST_K)
    for p, i in zip(top.values, top.indices):
        new_ids = torch.cat([ids, i.view(1, 1)], dim=1)
        expand(new_ids, tok.decode(i), p.item(), 1, words)
    return dict(words)


_cache = json.load(open(CACHE)) if CACHE.exists() else {}


def _words(prompt):
    if prompt not in _cache:
        _cache[prompt] = word_tree(prompt)
    return _cache[prompt]


def save_cache():
    CACHE.parent.mkdir(exist_ok=True)
    json.dump(_cache, open(CACHE, "w"))


def prior_fn(text):
    """P(next grid key | text typed so far). Text uses normal spaces, e.g. 'BROWN F'."""
    text = text.upper()
    cut = text.rfind(" ")
    context, partial = text[:cut + 1].strip(), text[cut + 1:]
    prompt = context.capitalize()  # e.g. 'Brown' (natural casing for Llama)
    p = np.zeros(len(GRID))
    for w, pr in _words(prompt).items():
        w = w.strip().upper()
        if not w or not w.startswith(partial):
            continue
        nxt = w[len(partial):len(partial) + 1]
        if nxt == "":
            p[GRID.index("sp")] += pr  # word complete -> space
        elif nxt in GRID:
            p[GRID.index(nxt)] += pr
    if p.sum() == 0:
        p[:] = 1.0  # nothing matched (e.g., after a typo)
    p /= p.sum()
    return (1 - FLOOR) * p + FLOOR / len(GRID)


if __name__ == "__main__":  # Quick test
    for t in ["", "THE ", "THE QU", "BROWN F", "I NEED W"]:
        pr = prior_fn(t)
        top = np.argsort(pr)[::-1][:5]
        print(f"{t!r:>12} -> " + ", ".join(f"{GRID[i]}:{pr[i]:.2f}" for i in top))
    save_cache()
