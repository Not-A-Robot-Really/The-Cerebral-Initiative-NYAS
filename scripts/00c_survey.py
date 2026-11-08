"""
Survey every .dat file: paradigm, spelling mode, text, flash counts, target labels.
Saves docs/dataset_survey.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from BCI2kReader import BCI2kReader as b2k

ROOT = Path("data/ds005028")
rows = []

for p in sorted(ROOT.rglob("*_eeg.dat")):
    parts = dict(x.split("-", 1) for x in p.name.split("_") if "-" in x)
    with b2k.BCI2kReader(str(p)) as f:
        prm, st, sig = f.parameters, f.states, f.signals
        code = np.asarray(st["StimulusCode"]).ravel()
        typ = np.asarray(st["StimulusType"]).ravel()
        onsets = np.where((code[1:] != 0) & (code[:-1] == 0))[0] + 1
        fs = float(prm["SamplingRate"])
        ch = list(prm["ChannelNames"])
        pz = sig[ch.index("Pz")] if "Pz" in ch else sig[0]

        rows.append({
            "sub": parts.get("sub"),
            "session": parts.get("ses"),
            "task": parts.get("task", ""),
            "run": parts.get("run"),
            "paradigm": str(prm.get("FlashingParadigm", "")),
            "interpret_mode": str(prm.get("InterpretMode", "")),
            "text_to_spell": repr(str(prm.get("TextToSpell", ""))),
            "text_result": repr(str(prm.get("TextResult", "")))[:40],
            "n_flashes": len(onsets),
            "max_code": int(code.max()),
            "target_frac": round(float((typ[onsets] == 1).mean()), 3) if len(onsets) else None,
            "minutes": round(sig.shape[1] / fs / 60, 2),
            "pz_std": round(float(np.std(pz)), 2),
        })
    print("done", p.name)

df = pd.DataFrame(rows)
df.to_csv("docs/dataset_survey.csv", index=False)
pd.set_option("display.width", 250)
print("\n", df.to_string(index=False))