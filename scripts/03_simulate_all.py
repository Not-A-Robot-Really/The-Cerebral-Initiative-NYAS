"""
All four conditions on 2-fold LDA scores:
  baseline : uniform prior, 10 sequences
  bayes    : uniform prior, early stopping
  llm      : Llama word-tree letter prior, 10 sequences
  combined : Llama prior + early stopping   <- hypothesis
A) dataset phrases: real recorded scores
B) held-out phrases: flashes simulated by resampling each fold's real test scores
Threshold / LLM weight tuned on the TRAINING run of each fold only
(highest ITR with accuracy >= 90% of training baseline).

Outputs: results/sim_results.csv, results/final_summary_{dataset,heldout}.csv,
         figures/results_{dataset,heldout}.png
Usage:   python scripts/03_simulate_all.py        (all)
         python scripts/03_simulate_all.py P01    (one subject)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from speller.bayes import M, split_chars, spell, summarize, wolpaw_bits
from speller.llm import prior_fn, save_cache, GRID as LLM_GRID

cfg = yaml.safe_load(open("config.yaml"))
TH, MIN_SEQ = cfg["bayes"]["thresholds"], cfg["bayes"]["min_seq"]
W = cfg["llm"]["weights"]
SEQ_S = cfg["flashes_per_sequence"] * (cfg["flash_ms"] + cfg["isi_ms"]) / 1000
PAUSE_S = (cfg["pre_char_ms"] + cfg["post_char_ms"]) / 1000
RES, FIG = Path("results"), Path("figures")
rng = np.random.default_rng(cfg["seed"])
CONDS = ["baseline", "bayes", "llm", "combined"]
ALPHA = 0.05 / 3
to_targets = lambda text: ["sp" if c == " " else c for c in text]
summ = lambda rows: summarize(rows, SEQ_S, PAUSE_S)

OPTIONS = {"bayes": [(t, None, 1.0) for t in TH],
           "llm": [(None, prior_fn, w) for w in W],
           "combined": [(t, prior_fn, w) for t in TH for w in W]}
FALLBACK = {"bayes": (max(TH), None, 1.0), "llm": (None, prior_fn, min(W)),
            "combined": (max(TH), prior_fn, min(W))}


def tune(fl, tgt, grid, dist):
    base_acc = summ(spell(fl, tgt, grid, dist))["accuracy"]
    params = {"baseline": (None, None, 1.0)}
    for cond, opts in OPTIONS.items():
        best = None
        for t, pf, w in opts:
            s = summ(spell(fl, tgt, grid, dist, t, pf, w, MIN_SEQ))
            if s["accuracy"] < 0.9 * base_acc:
                continue
            if best is None or (s["itr"], s["accuracy"]) > best[0]:
                best = ((s["itr"], s["accuracy"]), (t, pf, w))
        params[cond] = best[1] if best else FALLBACK[cond]
    return params


def sim_flashes(targets, grid, pool_t, pool_n):
    out = []
    for ch in targets:
        ti = grid.index(ch)
        codes = np.concatenate([rng.permutation(np.arange(1, 13)) for _ in range(cfg["n_sequences"])])
        is_t = M[codes, ti]
        sc = np.where(is_t, rng.choice(pool_t, len(codes)), rng.choice(pool_n, len(codes)))
        out.append((sc, codes))
    return out


def run(only=None):
    files = sorted(RES.glob("P*_fold-*_scores.npz"))
    if only:
        files = [f for f in files if f.name.startswith(only)]
    rows = []
    for f in files:
        sub, fold = f.stem.split("_")[0], f.stem.split("_")[1]
        d = np.load(f, allow_pickle=True)
        dist = json.load(open(RES / f"{sub}_{fold}_dist.json"))
        grid = [str(g) for g in d["grid"]]
        assert grid == LLM_GRID, "Grid order mismatch between data and LLM"
        tr_fl = split_chars(d["train_scores"], d["train_code"], d["train_char"])
        te_fl = split_chars(d["test_scores"], d["test_code"], d["test_char"])
        tr_tgt, te_tgt = to_targets(str(d["train_text"])), to_targets(str(d["test_text"]))
        params = tune(tr_fl, tr_tgt, grid, dist)

        def record(phrases, rep, phrase, fl, tgt):
            for cond in CONDS:
                t, pf, w = params[cond]
                s = summ(spell(fl, tgt, grid, dist, t, pf, w, MIN_SEQ))
                rows.append({"sub": sub, "fold": fold, "phrases": phrases, "rep": rep,
                             "phrase": phrase, "condition": cond, "threshold": t,
                             "llm_weight": w if pf else None, **s})

        record("dataset", 0, str(d["test_text"]), te_fl, te_tgt)
        pool_t, pool_n = d["test_scores"][d["test_y"] == 1], d["test_scores"][d["test_y"] == 0]
        for rep in range(cfg["heldout"]["reps"]):
            for ph in cfg["heldout"]["phrases"]:
                tg = to_targets(ph)
                record("heldout", rep, ph, sim_flashes(tg, grid, pool_t, pool_n), tg)
        save_cache()
        done = [r for r in rows if r["sub"] == sub and r["fold"] == fold and r["phrases"] == "dataset"]
        print(f"{sub} {fold}: " + " | ".join(
            f"{r['condition']} {r['n_correct']}/{r['n_chars']} {r['n_seq']/r['n_chars']:.1f}seq "
            f"'{r['typed']}'" for r in done))
    return pd.DataFrame(rows)


def analyze(df):
    for ph in ["dataset", "heldout"]:
        g = (df[df.phrases == ph].groupby(["sub", "condition"])
             [["n_chars", "n_correct", "time_s", "n_seq"]].sum().reset_index())
        g["accuracy"] = g.n_correct / g.n_chars
        g["sec_per_char"] = g.time_s / g.n_chars
        g["seq_per_char"] = g.n_seq / g.n_chars
        g["itr"] = [wolpaw_bits(a) * 60 / t for a, t in zip(g.accuracy, g.sec_per_char)]
        piv = lambda col: g.pivot(index="sub", columns="condition", values=col)[CONDS]
        itr, acc, spc, seq = piv("itr"), piv("accuracy"), piv("sec_per_char"), piv("seq_per_char")

        out = []
        for c in CONDS:
            change = (itr[c].mean() - itr.baseline.mean()) / itr.baseline.mean() * 100
            ratio = acc[c].mean() / acc.baseline.mean()
            p = np.nan
            if c != "baseline":
                try:
                    p = wilcoxon(itr[c], itr.baseline).pvalue
                except ValueError:
                    pass  # identical to baseline
            out.append({"condition": c, "ITR_mean": itr[c].mean(), "ITR_sd": itr[c].std(),
                        "ITR_change_%": change, "accuracy_mean": acc[c].mean(),
                        "accuracy_vs_baseline": ratio, "sec_per_char": spc[c].mean(),
                        "seq_per_char": seq[c].mean(), "wilcoxon_p": p,
                        "significant": None if np.isnan(p) else bool(p < ALPHA),
                        "PASS": None if c == "baseline" else bool(change >= 10 and ratio >= 0.9)})
        summary = pd.DataFrame(out).round(3)
        summary.to_csv(RES / f"final_summary_{ph}.csv", index=False)
        print(f"\n===== {ph.upper()} PHRASES (n = {len(itr)}) =====")
        print(summary.to_string(index=False))

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(CONDS, itr.mean(), yerr=itr.std(), capsize=4)
        ax[0].axhline(itr.baseline.mean() * 1.1, ls="--", color="k", label="+10% target")
        ax[0].set_ylabel("ITR (bits/min)"); ax[0].legend()
        ax[1].bar(CONDS, acc.mean() * 100, yerr=acc.std() * 100, capsize=4, color="tab:green")
        ax[1].axhline(acc.baseline.mean() * 90, ls="--", color="k", label="90% of baseline")
        ax[1].set_ylabel("Accuracy (%)"); ax[1].set_ylim(0, 105); ax[1].legend()
        fig.suptitle(f"{ph} phrases (n = {len(itr)})")
        fig.tight_layout(); fig.savefig(FIG / f"results_{ph}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    df = run(only)
    df.to_csv(RES / ("sim_results.csv" if not only else f"sim_results_{only}.csv"), index=False)
    if not only:
        analyze(df)
