"""
Orchestrates the full experiment in the order of the README "Research Plan".

Stages (each mapped to the plan below):
  A. Data + classifier  : filter EEG, cut epochs, train 1 LDA per participant, 2-fold CV
  B. Sanity checks      : 11 participants x 2 folds of scores, grid order, 120 flashes/char
  C. Simulations 1-4    : baseline, Bayes, LLM, combined (scripts/03_simulate_all.py)
                          or 1-2 only with --no-llm (scripts/03_bayes.py)
  D. Hypothesis verdict : ITR +10% and accuracy >= 90% of baseline, Wilcoxon (Bonferroni)

Usage (from the repo root):
  python scripts/run_all.py                 # use existing results/, run all 4 conditions
  python scripts/run_all.py --from-raw      # also redo preprocessing + LDA (needs data/ds005028)
  python scripts/run_all.py --no-llm        # conditions 1-2 only (no Llama / torch needed)
  python scripts/run_all.py --subject P01   # quick test on one participant (no final analysis)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SUBJECTS = [f"P{i:02d}" for i in range(1, 12)]   # Plan: "11 times, one for each participant (P_01 to P_11)"
FOLDS = ["fold-1", "fold-2"]                      # Plan: "2-fold cross-validation ... on their two runs"
ALPHA = 0.05 / 3                                  # 3 comparisons vs. control (Bonferroni)


def run(script, *args):
    """Run one pipeline script from the repo root; stop the whole pipeline if it fails."""
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit(f"FAILED: {script}")


# ---------------------------------------------------------------------------
# Stage A - Plan: "take the raw EEG dataset from the OpenNeuro dataset, filter it,
# cut out the epochs using a Python script, and put it into the LDA classifier,
# where it will output a score for each individual epoch."
# Plan: "The classifier is trained on one run and tested on the other. Then, the
# runs are swapped, so every character is tested by a classifier that never saw it."
# ---------------------------------------------------------------------------
def stage_a_classifier():
    if not (ROOT / "data" / "ds005028").exists():
        sys.exit("data/ds005028 missing. Download it first (see README: OpenNeuro Data Retrieval).")
    run("01_preprocess.py")   # filter (0.5-20 Hz, 60 Hz notch) + epoch -0.1..0.8 s per flash
    run("01b_plot_erp.py")    # visual check that the P300 is present (target vs non-target ERP)
    run("02_train_lda.py")    # 1 LDA per participant, 2 folds -> PXX_fold-K_scores.npz + _dist.json


# ---------------------------------------------------------------------------
# Stage B - checks that the classifier outputs match the plan before simulating.
# Plan: "6x6 grid, 62.5 ms flashes, 10 sequences" -> 12 flashes x 10 sequences = 120 per character.
# ---------------------------------------------------------------------------
def stage_b_checks(subjects):
    grid_expected = [chr(c) for c in range(65, 91)] + [str(i) for i in range(1, 10)] + ["sp"]
    for sub in subjects:
        for fold in FOLDS:
            f, dj = RES / f"{sub}_{fold}_scores.npz", RES / f"{sub}_{fold}_dist.json"
            if not f.exists() or not dj.exists():
                sys.exit(f"Missing {f.name} / {dj.name}: run with --from-raw")
            d = np.load(f, allow_pickle=True)
            assert [str(g) for g in d["grid"]] == grid_expected, f"{f.name}: grid order"
            for part in ("train", "test"):
                counts = np.bincount(d[f"{part}_char"])
                assert (counts == 120).all(), f"{f.name}: {part} flashes per char {set(counts)}"
                # every character has exactly 2 target flashes per sequence (its row + column)
                assert d[f"{part}_y"].sum() == 20 * len(counts), f"{f.name}: {part} target count"
            json.load(open(dj))  # distributions come from the training run only
    print(f"Stage B OK: {len(subjects)} participants x {len(FOLDS)} folds, 120 flashes/char")
    lda = RES / "lda_summary.json"
    if lda.exists():
        print("LDA AUC summary:", lda.read_text()[:300])


# ---------------------------------------------------------------------------
# Stage C - the four simulations, each run for every participant and fold.
#   1) Control: baseline speller, uniform prior, all 10 sequences
#   2) Adaptive Bayesian Early Stopping
#   3) LLM (Llama 3.2 1B) phrase prediction, all 10 sequences
#   4) Combined LLM + Bayesian early stopping  <- the hypothesis condition
# Settings (threshold, LLM weight) are tuned on each fold's training run only.
# Plan: "Selections not present in the recordings are simulated using each participant's
# measured ... score distributions from their test data" -> held-out phrases in 03_simulate_all.py.
# ---------------------------------------------------------------------------
def stage_c_simulate(subject, no_llm):
    if no_llm:
        run("03_bayes.py")                       # conditions 1 and 2 only
    else:
        run("03_simulate_all.py", *([subject] if subject else []))  # conditions 1-4


# ---------------------------------------------------------------------------
# Stage D - Plan: "A trial is considered to have a sufficient ITR if the ITR increased by
# at least 10% relative to the baseline" (Wolpaw ITR) and "Any test with an accuracy less
# than 90% of the accuracy of the control case will be considered failed, regardless of ITR."
# Plan: "The results from both folds are combined into one accuracy and one ITR per
# participant" -> done inside the 03_* scripts before these summaries are written.
# ---------------------------------------------------------------------------
def stage_d_verdict(no_llm):
    files = [RES / "bayes_summary.csv"] if no_llm else \
            [RES / "final_summary_dataset.csv", RES / "final_summary_heldout.csv"]
    for f in files:
        s = pd.read_csv(f)
        print(f"\n===== {f.name} =====")
        for _, r in s[s.condition != "baseline"].iterrows():
            itr_ok = r["ITR_change_%"] >= 10
            acc_ok = r["accuracy_vs_baseline"] >= 0.9
            sig = r["wilcoxon_p"] < ALPHA if pd.notna(r["wilcoxon_p"]) else False
            print(f"{r.condition:>9}: ITR {r['ITR_change_%']:+.1f}% ({'ok' if itr_ok else 'FAIL'}) | "
                  f"accuracy {r.accuracy_vs_baseline:.3f} of baseline ({'ok' if acc_ok else 'FAIL'}) | "
                  f"p={r.wilcoxon_p:.4f} ({'significant' if sig else 'n.s.'} at {ALPHA:.4f}) | "
                  f"{'PASS' if itr_ok and acc_ok else 'FAIL'}")
        if not no_llm:
            c = s[s.condition == "combined"].iloc[0]
            ok = c["ITR_change_%"] >= 10 and c["accuracy_vs_baseline"] >= 0.9
            print(f"HYPOTHESIS (combined LLM + Bayes): {'SUPPORTED' if ok else 'NOT SUPPORTED'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-raw", action="store_true", help="redo preprocessing + LDA (Stage A)")
    ap.add_argument("--no-llm", action="store_true", help="run conditions 1-2 only")
    ap.add_argument("--subject", help="single participant quick test, e.g. P01")
    a = ap.parse_args()

    if a.from_raw:
        stage_a_classifier()
    stage_b_checks([a.subject] if a.subject else SUBJECTS)
    stage_c_simulate(a.subject, a.no_llm)
    if a.subject and not a.no_llm:
        print("\nSingle-subject test done; run without --subject for the final analysis.")
    else:
        stage_d_verdict(a.no_llm)
