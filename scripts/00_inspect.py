"""
Inspect ds005028 before building the pipeline.
Prints dataset structure and saves everything to docs/inspect_output.txt
"""
import sys
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

ROOT = "data/ds005028"
out = open("docs/inspect_output.txt", "w", encoding="utf-8")

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    out.write(text + "\n")

# ---------- 1. Overall structure ----------
log("=" * 60)
log("DATASET STRUCTURE")
log("=" * 60)
log("Subjects:", get_entity_vals(ROOT, "subject"))
log("Sessions:", get_entity_vals(ROOT, "session"))
log("Tasks:   ", get_entity_vals(ROOT, "task"))
log("Runs:    ", get_entity_vals(ROOT, "run"))

# ---------- 2. Files for the first subject ----------
sub = get_entity_vals(ROOT, "subject")[0]
paths = BIDSPath(root=ROOT, subject=sub, datatype="eeg", suffix="eeg").match()
log("\n" + "=" * 60)
log(f"EEG FILES FOR SUBJECT {sub}  ({len(paths)} files)")
log("=" * 60)
for p in paths:
    log("  ", p.basename)

if not paths:
    log("No EEG files found. Check the folder structure in data/ds005028.")
    sys.exit()

# ---------- 3. Recording details ----------
bp = paths[0]
raw = read_raw_bids(bp, verbose=False)
log("\n" + "=" * 60)
log(f"RECORDING: {bp.basename}")
log("=" * 60)
log("Sampling rate:", raw.info["sfreq"], "Hz")
log("Number of channels:", len(raw.ch_names))
log("Channel names:", raw.ch_names)
log("Duration:", round(raw.times[-1], 1), "seconds")

# ---------- 4. Events table ----------
ev_path = bp.copy().update(suffix="events", extension=".tsv")
ev = pd.read_csv(ev_path.fpath, sep="\t")
log("\n" + "=" * 60)
log("EVENTS TABLE")
log("=" * 60)
log("Rows:", len(ev))
log("Columns:", ev.columns.tolist())
log("\nFirst 40 rows:")
log(ev.head(40).to_string())

log("\n" + "=" * 60)
log("UNIQUE VALUES PER COLUMN (columns with < 60 unique values)")
log("=" * 60)
for col in ev.columns:
    n = ev[col].nunique()
    if n < 60:
        log(f"\n[{col}]  ({n} unique):")
        log("  ", ev[col].value_counts().head(60).to_dict())
    else:
        log(f"\n[{col}]  ({n} unique values, e.g. {ev[col].head(5).tolist()})")

# ---------- 5. MNE's view of the events ----------
log("\n" + "=" * 60)
log("MNE ANNOTATIONS (first 20)")
log("=" * 60)
for a in raw.annotations[:20]:
    log(f"  onset={a['onset']:.3f}s  duration={a['duration']}  desc={a['description']}")

out.close()
print("\nSaved to docs/inspect_output.txt")