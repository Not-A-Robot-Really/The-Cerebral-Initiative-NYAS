import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "meta-llama/Llama-3.2-1B"  # base checkpoint
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.bfloat16, device_map="auto"
)

text = "The most famous Russian composer is" # input
from collections import defaultdict

TOP_K = 5          # branches per node
MAX_DEPTH = 10      # max tokens per word
MIN_PROB = 0.005   # prune branches below this cumulative probability
FIRST_K = 10       # how many possibilities to use on the main branch

# Build once: which token IDs continue a word (start with a letter, no leading space)
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

    # Chance the word ends here = total probability of all non-continuation tokens
    end_p = probs[~is_cont].sum().item()
    words[word] += prob * end_p
    print(f"{'  ' * depth}{word!r}  path={prob:.4f}  ends_here={prob * end_p:.4f}")

    if depth >= MAX_DEPTH:
        return

    # Only follow tokens that continue the word
    cont = probs.masked_fill(~is_cont, 0.0)
    top = torch.topk(cont, TOP_K)
    for p, i in zip(top.values, top.indices):
        child_prob = prob * p.item()
        if child_prob < MIN_PROB:
            continue
        new_ids = torch.cat([ids, i.view(1, 1)], dim=1)
        expand(new_ids, word + tok.decode(i), child_prob, depth + 1, words)

def word_tree(prompt):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    words = defaultdict(float)

    # First token: any top candidate is allowed to start the word
    top = torch.topk(next_probs(ids), FIRST_K)
    for p, i in zip(top.values, top.indices):
        new_ids = torch.cat([ids, i.view(1, 1)], dim=1)
        expand(new_ids, tok.decode(i), p.item(), 1, words)

    print("\nCompleted words:")
    for w, p in sorted(words.items(), key=lambda x: -x[1]):
        if p > 0.001:
            print(f"{w!r:>20}  {p:.4f}")

    total = sum(words.values())
    print(f"\nCovered: {total:.4f}   Missing: {1 - total:.4f}") # Covered does not have 0.001 filter so may be slightly higher
    # Words and probabilities in dict words        
    return words

words_predicted = word_tree(text) # find words predicted
