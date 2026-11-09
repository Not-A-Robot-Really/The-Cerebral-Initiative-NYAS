"""
Part 7: Preprocess BCI2000 .dat files (ses-train, task-rc).
For each run: filter, epoch every flash, label targets, save features.

Usage:
    python scripts/01_preprocess.py          # all subjects
    python scripts/01_preprocess.py 01       # one subject (for testing)

Output: results/P{sub}_run-{run}_epochs.npz
"""
import sys
from pathlib import Path
import numpy as np
import yaml
import mne
from BCI2kReader import BCI2kReader as b2k

cfg = yaml.safe_load(open("config.yaml"))
ROOT = Path(cfg["bids_root"])
OUT = Path("results")
OUT.mkdir(exist_ok=True)
mne.set_log_level("WARNING")
ROWS, COLS = cfg["grid_rows"], cfg["grid_cols"]


# ---------- 1. Find the rc training runs for a subject ----------
def find_runs(sub):
    folder = ROOT / f"sub-{sub}" / f"ses-{cfg['session']}" / "eeg"
    pattern = f"sub-{sub}_ses-{cfg['session']}_task-{cfg['task']}_run-*_eeg.dat"
    return sorted(folder.glob(pattern))


# ---------- 2. Read one .dat file ----------
def read_dat(path):
    with b2k.BCI2kReader(str(path)) as f:
        sig = np.asarray(f.signals, dtype=float)          # (channels, samples)
        prm = f.parameters
        code = np.nan_to_num(np.asarray(f.states["StimulusCode"]).ravel()).astype(int)
        typ = np.nan_to_num(np.asarray(f.states["StimulusType"]).ravel()).astype(int)
    fs = float(str(prm["SamplingRate"]).replace("Hz", ""))
    chans = [str(c) for c in prm["ChannelNames"]]
    grid = [str(t[0]) for t in prm["TargetDefinitions"]]
    text = str(prm.get("TextToSpell", ""))
    return sig, fs, chans, grid, text, code, typ


# ---------- 3. Find flashes and group them into characters ----------
def find_flashes(code, typ, fs):
    prev = np.concatenate([[0], code[:-1]])
    onsets = np.where((code != 0) & (prev == 0))[0]       # sample where a flash starts
    flash_code = code[onsets]                              # which row/col flashed (1-12)
    is_target = (typ[onsets] == 1).astype(int)             # 1 = contained target
    gaps = np.diff(onsets, prepend=onsets[0])
    char_idx = np.cumsum(gaps > 1.0 * fs)                  # >1 s gap = new character
    return onsets, flash_code, is_target, char_idx


# ---------- 4. Rebuild the target text (checks our labels are right) ----------
def decode_targets(flash_code, is_target, char_idx, grid):
    """Try both BCI2000 conventions; the correct one will match TextToSpell."""
    out = {"A_cols_1to6": "", "B_rows_1to6": ""}
    for c in np.unique(char_idx):
        t = set(flash_code[(char_idx == c) & (is_target == 1)].tolist())
        low = [k for k in t if 1 <= k <= 6]
        high = [k for k in t if 7 <= k <= 12]
        if len(low) == 1 and len(high) == 1:
            # A: codes 1-6 = columns, 7-12 = rows
            out["A_cols_1to6"] += grid[(high[0] - 7) * COLS + (low[0] - 1)]
            # B: codes 1-6 = rows, 7-12 = columns
            out["B_rows_1to6"] += grid[(low[0] - 1) * COLS + (high[0] - 7)]
        else:
            out["A_cols_1to6"] += "?"
            out["B_rows_1to6"] += "?"
    return out


# ---------- 5. Filter and epoch ----------
def make_epochs(sig, fs, chans, onsets):
    # BCI2kReader gives microvolts; MNE expects volts
    if np.median(np.std(sig, axis=1)) > 1e-3:
        sig = sig * 1e-6

    raw = mne.io.RawArray(sig, mne.create_info(chans, fs, ch_types="eeg"))
    raw.set_eeg_reference("average")          # use all 32 channels for the average
    raw.notch_filter(cfg["notch"])
    raw.filter(cfg["l_freq"], cfg["h_freq"])  # filter CONTINUOUS data
    raw.pick(cfg["channels"])                 # then keep our 10 channels

    events = np.column_stack([onsets, np.zeros_like(onsets), np.ones_like(onsets)])
    decim = max(1, int(fs // cfg["decim_to_hz"]))
    ep = mne.Epochs(raw, events, event_id=1,
                    tmin=cfg["tmin"], tmax=cfg["tmax"],
                    baseline=(cfg["tmin"], 0), decim=decim,
                    preload=True, reject=None)
    return ep


# ---------- 6. Main loop ----------
def process_subject(sub):
    runs = find_runs(sub)
    print(f"\n=== Subject {sub}: {len(runs)} rc runs ===")
    for path in runs:
        run = path.name.split("run-")[1].split("_")[0]
        sig, fs, chans, grid, text, code, typ = read_dat(path)
        onsets, fcode, y, cidx = find_flashes(code, typ, fs)
        decoded = decode_targets(fcode, y, cidx, grid)

        ep = make_epochs(sig, fs, chans, onsets)
        keep = ep.selection                                   # epochs MNE kept
        data = ep.get_data()                                  # (flashes, channels, times)
        X = data[:, :, ep.times >= 0].reshape(len(data), -1)  # features: 0 to 0.8 s

        np.savez(OUT / f"P{sub}_run-{run}_epochs.npz",
                 X=X, y=y[keep], code=fcode[keep], char=cidx[keep],
                 erp=data, times=ep.times, chans=np.array(cfg["channels"]),
                 grid=np.array(grid), text_to_spell=np.array(text),
                 decoded_A=np.array(decoded["A_cols_1to6"]),
                 decoded_B=np.array(decoded["B_rows_1to6"]))

        per_char = np.bincount(cidx)
        print(f" run {run}: {len(onsets)} flashes, {len(per_char)} characters, "
              f"flashes/char = {sorted(set(per_char.tolist()))}")
        print(f"   kept {len(keep)} epochs | X shape {X.shape} | target rate {y[keep].mean():.3f}")
        print(f"   TextToSpell : {text!r}")
        print(f"   Decoded (A) : {decoded['A_cols_1to6']!r}")
        print(f"   Decoded (B) : {decoded['B_rows_1to6']!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        subs = sys.argv[1:]
    else:
        subs = sorted(p.name.split("-")[1] for p in ROOT.glob("sub-*") if p.is_dir())
    for s in subs:
        process_subject(s)
    print("\nDone.")