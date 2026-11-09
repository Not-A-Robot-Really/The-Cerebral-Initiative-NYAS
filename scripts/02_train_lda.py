"""
Part 9: One LDA per participant per fold (2-fold cross-validation on the rc runs).
  Fold 1: train on run with 'THE QUICK ', test on 'BROWN FOX '
  Fold 2: the reverse
Score distributions come from OUT-OF-FOLD scores on the TRAINING run only
(5-fold by character inside the training run), so they are honest.

Outputs per subject/fold:
  results/P01_fold-1_scores.npz   (test + training scores, codes, chars, text, grid)
  results/P01_fold-1_dist.json    (target / non-target score mean and sd)
  results/lda_summary.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

RES = Path("results")
run_num = lambda f: int(f.name.split("run-")[1].split("_")[0])
subs = sorted({f.name.split("_")[0] for f in RES.glob("P*_run-*_epochs.npz")})
summary = {}

for sub in subs:
    files = sorted(RES.glob(f"{sub}_run-*_epochs.npz"), key=run_num)  # lower run = THE QUICK
    runs = [np.load(f, allow_pickle=True) for f in files]

    for fold, (tr, te) in enumerate([(0, 1), (1, 0)], start=1):
        A, B = runs[tr], runs[te]

        # Honest training-run scores: predict each character with a model that didn't see it
        oof = cross_val_predict(LDA(solver="lsqr", shrinkage="auto"),
                                A["X"], A["y"], groups=A["char"],
                                cv=GroupKFold(n_splits=5), method="decision_function")

        # Final model on the whole training run, applied to the unseen test run
        lda = LDA(solver="lsqr", shrinkage="auto").fit(A["X"], A["y"])
        s_test = lda.decision_function(B["X"])

        yA = A["y"]
        dist = {"target_mean": float(oof[yA == 1].mean()), "target_sd": float(oof[yA == 1].std()),
                "nontarget_mean": float(oof[yA == 0].mean()), "nontarget_sd": float(oof[yA == 0].std())}

        tag = f"{sub}_fold-{fold}"
        np.savez(RES / f"{tag}_scores.npz",
                 test_scores=s_test, test_y=B["y"], test_code=B["code"],
                 test_char=B["char"], test_text=B["text_to_spell"],
                 train_scores=oof, train_y=yA, train_code=A["code"],
                 train_char=A["char"], train_text=A["text_to_spell"], grid=A["grid"])
        json.dump(dist, open(RES / f"{tag}_dist.json", "w"), indent=2)

        summary[tag] = {"train_AUC_oof": round(roc_auc_score(yA, oof), 3),
                        "test_AUC": round(roc_auc_score(B["y"], s_test), 3),
                        "train_text": str(A["text_to_spell"]), "test_text": str(B["text_to_spell"]),
                        **{k: round(v, 3) for k, v in dist.items()}}
        print(f"{tag}: train(oof) AUC {summary[tag]['train_AUC_oof']:.3f} | "
              f"test AUC {summary[tag]['test_AUC']:.3f}")

json.dump(summary, open(RES / "lda_summary.json", "w"), indent=2)
aucs = [v["test_AUC"] for v in summary.values()]
print(f"\nMean test AUC {np.mean(aucs):.3f}  (min {min(aucs):.3f}, max {max(aucs):.3f})")