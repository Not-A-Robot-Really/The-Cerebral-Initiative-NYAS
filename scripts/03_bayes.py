"""
Baseline vs Adaptive Bayesian Early Stopping on real LDA scores (2-fold).
Threshold tuned on each fold's TRAINING run (out-of-fold scores) only: pick the
threshold with the highest ITR that keeps accuracy >= 90% of baseline.

Outputs:
  results/bayes_results.csv   per subject/fold/condition
  results/bayes_chars.csv     per character (sequences used, confidence)
  results/bayes_summary.csv   final table (per-subject averages, pass/fail, Wilcoxon)
  figures/bayes_results.png
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so we can import speller/

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from speller.bayes import split_chars, spell, summarize, wolpaw_bits

cfg = yaml.safe_load(open("config.yaml"))
B = cfg["bayes"]
SEQ_S = cfg["flashes_per_sequence"] * (cfg["flash_ms"] + cfg["isi_ms"]) / 1000  # 1.5 s
PAUSE_S = (cfg["pre_char_ms"] + cfg["post_char_ms"]) / 1000                     # 3.5 s
RES, FIG = Path("results"), Path("figures")
to_targets = lambda text: ["sp" if c == " " else c for c in text]
summ = lambda rows: summarize(rows, SEQ_S, PAUSE_S)


def tune_threshold(fl, tgt, grid, dist):
    base = summ(spell(fl, tgt, grid, dist))
    best = None
    for thr in B["thresholds"]:
        s = summ(spell(fl, tgt, grid, dist, thr, min_seq=B["min_seq"]))
        if s["accuracy"] < 0.9 * base["accuracy"]:
            continue
        key = (s["itr"], s["accuracy"], thr)
        if best is None or key > best:
            best = key
    return best[2] if best else max(B["thresholds"])  # fallback: most conservative


rows, char_rows = [], []
for f in sorted(RES.glob("P*_fold-*_scores.npz")):
    sub, fold = f.stem.split("_")[0], f.stem.split("_")[1]
    d = np.load(f, allow_pickle=True)
    dist = json.load(open(RES / f"{sub}_{fold}_dist.json"))
    grid = [str(g) for g in d["grid"]]
    train_fl = split_chars(d["train_scores"], d["train_code"], d["train_char"])
    test_fl = split_chars(d["test_scores"], d["test_code"], d["test_char"])
    train_tgt = to_targets(str(d["train_text"]))
    test_tgt = to_targets(str(d["test_text"]))

    thr = tune_threshold(train_fl, train_tgt, grid, dist)
    line = []
    for cond, t in [("baseline", None), ("bayes", thr)]:
        chars = spell(test_fl, test_tgt, grid, dist, t, min_seq=B["min_seq"])
        s = summ(chars)
        rows.append({"sub": sub, "fold": fold, "condition": cond, "threshold": t, **s})
        char_rows += [{"sub": sub, "fold": fold, "condition": cond, **c} for c in chars]
        line.append(f"{cond}: {s['n_correct']}/{s['n_chars']} '{s['typed']}' "
                    f"{s['n_seq'] / s['n_chars']:.1f} seq/char")
    print(f"{sub} {fold} (thr={thr}): " + " | ".join(line))

df = pd.DataFrame(rows)
df.to_csv(RES / "bayes_results.csv", index=False)
pd.DataFrame(char_rows).to_csv(RES / "bayes_chars.csv", index=False)

# ---- Combine both folds per subject ----
g = df.groupby(["sub", "condition"])[["n_chars", "n_correct", "time_s", "n_seq"]].sum().reset_index()
g["accuracy"] = g.n_correct / g.n_chars
g["sec_per_char"] = g.time_s / g.n_chars
g["seq_per_char"] = g.n_seq / g.n_chars
g["itr"] = [wolpaw_bits(a) * 60 / t for a, t in zip(g.accuracy, g.sec_per_char)]
itr = g.pivot(index="sub", columns="condition", values="itr")
acc = g.pivot(index="sub", columns="condition", values="accuracy")
spc = g.pivot(index="sub", columns="condition", values="sec_per_char")
seq = g.pivot(index="sub", columns="condition", values="seq_per_char")

print("\nPer subject:")
print(pd.DataFrame({"acc_base": acc.baseline, "acc_bayes": acc.bayes,
                    "itr_base": itr.baseline.round(1), "itr_bayes": itr.bayes.round(1),
                    "seq_bayes": seq.bayes.round(1)}).to_string())

change = (itr.bayes.mean() - itr.baseline.mean()) / itr.baseline.mean() * 100
ratio = acc.bayes.mean() / acc.baseline.mean()
p = wilcoxon(itr.bayes, itr.baseline).pvalue if (itr.bayes != itr.baseline).any() else np.nan
summary = pd.DataFrame([
    {"condition": "baseline", "ITR_mean": itr.baseline.mean(), "ITR_sd": itr.baseline.std(),
     "accuracy_mean": acc.baseline.mean(), "sec_per_char": spc.baseline.mean(),
     "seq_per_char": seq.baseline.mean()},
    {"condition": "bayes", "ITR_mean": itr.bayes.mean(), "ITR_sd": itr.bayes.std(),
     "accuracy_mean": acc.bayes.mean(), "sec_per_char": spc.bayes.mean(),
     "seq_per_char": seq.bayes.mean(), "ITR_change_%": change,
     "accuracy_vs_baseline": ratio, "wilcoxon_p": p,
     "PASS": bool(change >= 10 and ratio >= 0.9)}]).round(3)
summary.to_csv(RES / "bayes_summary.csv", index=False)
print("\nSummary (n = %d):" % len(itr))
print(summary.to_string(index=False))

# ---- Figure ----
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
conds = ["baseline", "bayes"]
ax[0].bar(conds, [itr[c].mean() for c in conds], yerr=[itr[c].std() for c in conds], capsize=4)
ax[0].axhline(itr.baseline.mean() * 1.1, ls="--", color="k", label="+10% target")
ax[0].set_ylabel("ITR (bits/min)"); ax[0].legend()
ax[1].bar(conds, [acc[c].mean() * 100 for c in conds], yerr=[acc[c].std() * 100 for c in conds],
          capsize=4, color="tab:green")
ax[1].axhline(acc.baseline.mean() * 90, ls="--", color="k", label="90% of baseline")
ax[1].set_ylabel("Accuracy (%)"); ax[1].set_ylim(0, 105); ax[1].legend()
cr = pd.DataFrame(char_rows)
ax[2].hist(cr[cr.condition == "bayes"].n_seq, bins=np.arange(0.5, 11.5, 1), rwidth=0.85)
ax[2].set_xlabel("Sequences used per character"); ax[2].set_ylabel("Characters")
ax[2].set_title("Bayes: when it stopped")
fig.tight_layout(); fig.savefig(FIG / "bayes_results.png", dpi=150); plt.close(fig)
print("\nSaved results/bayes_*.csv and figures/bayes_results.png")
