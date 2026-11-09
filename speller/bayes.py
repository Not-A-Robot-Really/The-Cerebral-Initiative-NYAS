"""Adaptive Bayesian early stopping for the 6x6 row/column P300 speller."""
import numpy as np
from scipy.stats import norm

N_ROWS, N_COLS = 6, 6
N_CHARS = N_ROWS * N_COLS
FPS = N_ROWS + N_COLS  # 12 flashes per sequence


def build_members():
    """M[code, i] = True if character i lights up when `code` flashes.
    Codes 1-6 = rows, 7-12 = columns."""
    M = np.zeros((FPS + 1, N_CHARS), bool)
    for i in range(N_CHARS):
        r, c = divmod(i, N_COLS)
        M[r + 1, i] = True
        M[N_ROWS + c + 1, i] = True
    return M


M = build_members()


def split_chars(scores, codes, chars):
    """List of (scores, codes) per character, in recorded flash order."""
    return [(scores[chars == c], codes[chars == c]) for c in np.unique(chars)]


def posterior(logp):
    p = np.exp(logp - logp.max())
    return p / p.sum()


def decode_char(scores, codes, dist, threshold=None, log_prior=None, min_seq=1):
    """Bayesian update over 36 characters.
    threshold=None -> use all sequences (baseline).
    Returns (chosen index, sequences used, final confidence)."""
    lt = norm.logpdf(scores, dist["target_mean"], dist["target_sd"])
    ln = norm.logpdf(scores, dist["nontarget_mean"], dist["nontarget_sd"])
    logp = np.zeros(N_CHARS) if log_prior is None else log_prior.copy()
    n_seq = len(scores) // FPS
    for s in range(n_seq):
        for k in range(s * FPS, (s + 1) * FPS):
            logp += np.where(M[codes[k]], lt[k], ln[k])
        if threshold is not None and s + 1 >= min_seq:
            post = posterior(logp)
            if post.max() >= threshold:
                return int(post.argmax()), s + 1, float(post.max())
    post = posterior(logp)
    return int(post.argmax()), n_seq, float(post.max())


def spell(char_flashes, targets, grid, dist, threshold=None, prior_fn=None, weight=1.0, min_seq=1):
    """Spell a phrase. prior_fn(text_so_far) -> 36 probabilities in grid order (optional, for LLM).
    Errors are kept (no backspace) and the LLM sees what was actually typed."""
    typed, rows = [], []
    for k, ((sc, cd), tgt) in enumerate(zip(char_flashes, targets)):
        log_prior = None
        if prior_fn is not None:
            text = "".join(" " if g == "sp" else g for g in typed)
            log_prior = weight * np.log(np.clip(prior_fn(text), 1e-12, None))
        i, ns, conf = decode_char(sc, cd, dist, threshold, log_prior, min_seq)
        typed.append(grid[i])
        rows.append({"pos": k, "target": tgt, "chosen": grid[i], "correct": grid[i] == tgt,
                     "n_seq": ns, "confidence": round(conf, 4)})
    return rows


def wolpaw_bits(P, N=N_CHARS):
    if P >= 1:
        return np.log2(N)
    if P <= 1 / N:
        return 0.0
    return np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))


def summarize(rows, seq_s=1.5, pause_s=3.5):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    seqs = sum(r["n_seq"] for r in rows)
    time_s = seqs * seq_s + n * pause_s
    acc = correct / n
    return {"n_chars": n, "n_correct": correct, "n_seq": seqs, "time_s": time_s,
            "accuracy": acc, "sec_per_char": time_s / n,
            "itr": wolpaw_bits(acc) * 60 / (time_s / n),
            "typed": "".join(" " if r["chosen"] == "sp" else r["chosen"] for r in rows)}
