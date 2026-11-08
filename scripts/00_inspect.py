from pathlib import Path
import pandas as pd
import mne

ROOT = Path("data/ds005028")
SUB = "01"
out = open("docs/inspect_output.txt", "w", encoding="utf-8")

def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    out.write(text + "\n")

def inspect_file(edf_path):
    log("\n" + "=" * 70)
    log("FILE:", edf_path.name)
    log("=" * 70)

    # --- Recording ---
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    log("Sampling rate:", raw.info["sfreq"], "Hz")
    log("Channels (%d):" % len(raw.ch_names), raw.ch_names)
    log("Duration:", round(raw.times[-1], 1), "s")

    # --- Events table (same name, _eeg.edf -> _events.tsv) ---
    ev_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
    if not ev_path.exists():
        log("No events.tsv found at", ev_path.name)
    else:
        ev = pd.read_csv(ev_path, sep="\t")
        log("\nEvents rows:", len(ev))
        log("Events columns:", ev.columns.tolist())
        log("\nFirst 40 rows:")
        log(ev.head(40).to_string())
        log("\nUnique values per column:")
        for col in ev.columns:
            n = ev[col].nunique()
            if n < 60:
                log(f"  [{col}] ({n} unique):", ev[col].value_counts().head(60).to_dict())
            else:
                log(f"  [{col}] ({n} unique), e.g.", ev[col].head(5).tolist())

    # --- Annotations stored inside the EDF ---
    ann = raw.annotations
    log("\nEDF annotations:", len(ann))
    for a in ann[:20]:
        log(f"  onset={a['onset']:.3f}s  dur={a['duration']}  desc={a['description']}")

# ---------- Overview ----------
eeg_dir = ROOT / f"sub-{SUB}"
edfs = sorted(eeg_dir.rglob("*_eeg.edf"))
log("EDF files for subject", SUB)
for p in edfs:
    log("  ", p.relative_to(ROOT))

# ---------- Inspect one train file and one test file ----------
train = [p for p in edfs if "ses-train" in p.name and "task-rc" in p.name]
test = [p for p in edfs if "ses-test" in p.name]
if train: inspect_file(train[0])
if test:  inspect_file(test[0])

# ---------- Top-level docs that explain columns ----------
log("\n" + "=" * 70)
log("TOP-LEVEL JSON / TSV FILES")
log("=" * 70)
for f in sorted(ROOT.glob("*.json")) + sorted(ROOT.glob("*.tsv")):
    log("\n---", f.name, "---")
    log(f.read_text(encoding="utf-8", errors="ignore")[:3000])

out.close()
print("\nSaved to docs/inspect_output.txt")