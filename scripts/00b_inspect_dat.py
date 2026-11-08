"""
Inspect BCI2000 .dat files (signals + flash events + parameters).
Saves to docs/inspect_dat_output.txt
"""
from pathlib import Path
import numpy as np
from BCI2kReader import BCI2kReader as b2k

ROOT = Path("data/ds005028")
SUB = "01"
out = open("docs/inspect_dat_output.txt", "w", encoding="utf-8")

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    out.write(text + "\n")

# Parameters most relevant to a P300 speller
KEY_PARAMS = ["SamplingRate", "ChannelNames", "SourceCh", "NumberOfSequences",
              "StimulusDuration", "ISIMinDuration", "ISIMaxDuration",
              "PreSequenceDuration", "PostSequenceDuration", "TextToSpell",
              "NumMatrixRows", "NumMatrixColumns", "TargetDefinitions",
              "InterpretMode", "SpatialFilterType", "TransmitChList"]

def inspect(path):
    log("\n" + "=" * 70)
    log("FILE:", path.name, f"({path.stat().st_size/1e6:.1f} MB)")
    log("=" * 70)
    with b2k.BCI2kReader(str(path)) as f:
        sig = f.signals
        log("Signal shape (channels, samples):", sig.shape)

        params = f.parameters
        log("\n--- KEY PARAMETERS ---")
        for k in KEY_PARAMS:
            if k in params:
                v = params[k]
                log(f"{k}: {str(v)[:500]}")
        fs = params.get("SamplingRate")
        log("\nAll parameter names:", sorted(params.keys()))

        states = f.states
        log("\n--- STATES (event channels) ---")
        log("State names:", sorted(states.keys()))
        for name in ["StimulusCode", "StimulusType", "StimulusBegin",
                     "PhaseInSequence", "SelectedTarget", "SelectedRow",
                     "SelectedColumn", "Running", "Recording"]:
            if name in states:
                arr = np.asarray(states[name]).ravel()
                vals, counts = np.unique(arr, return_counts=True)
                log(f"{name}: unique = {dict(zip(vals.tolist(), counts.tolist()))}")

        # Count flashes = number of times StimulusCode switches from 0 to non-zero
        if "StimulusCode" in states:
            code = np.asarray(states["StimulusCode"]).ravel()
            starts = np.where((code[1:] != 0) & (code[:-1] == 0))[0] + 1
            log("\nNumber of flash onsets:", len(starts))
            log("First 30 flash codes:", code[starts[:30]].tolist())
            log("First 10 onset sample indices:", starts[:10].tolist())
            if len(starts) > 1:
                gaps = np.diff(starts[:50])
                log("Samples between flash onsets (first 49):", gaps.tolist())
            if "StimulusType" in states:
                typ = np.asarray(states["StimulusType"]).ravel()
                log("Target fraction among flashes:",
                    round(float((typ[starts] == 1).mean()), 3))

eeg = ROOT / f"sub-{SUB}"
dats = sorted(eeg.rglob("*_eeg.dat"))
train_rc = [p for p in dats if "ses-train" in p.name and "task-rc" in p.name]
test = [p for p in dats if "ses-test" in p.name]
if train_rc: inspect(train_rc[0])
if test:     inspect(test[0])

out.close()
print("\nSaved to docs/inspect_dat_output.txt")