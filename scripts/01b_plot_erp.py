from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG = Path("results"), Path("figures")
FIG.mkdir(exist_ok=True)
SHOW = ["Cz", "Pz", "PO7"]
WIN = (250, 500)

subs = sorted({f.name.split("_")[0] for f in RES.glob("P*_run-*_epochs.npz")})
rows, grand_t, grand_n = [], [], []

for sub in subs:
    erps, ys = [], []
    for f in sorted(RES.glob(f"{sub}_run-*_epochs.npz")):
        d = np.load(f, allow_pickle=True)
        erps.append(d["erp"]); ys.append(d["y"])
        times = d["times"] * 1000
        chans = list(d["chans"])
    erp = np.concatenate(erps) * 1e6          # volts -> microvolts
    y = np.concatenate(ys)
    tgt, non = erp[y == 1].mean(0), erp[y == 0].mean(0)
    diff = tgt - non
    win = (times >= WIN[0]) & (times <= WIN[1])
    pz = chans.index("Pz")
    rows.append({"sub": sub,
                 "p300_diff_uV_Pz": round(float(diff[pz, win].mean()), 2),
                 "peak_latency_ms_Pz": int(times[win][np.argmax(diff[pz, win])]),
                 "noise_sd_uV": round(float(erp[y == 0].std()), 1),
                 "n_target": int(y.sum()), "n_nontarget": int((y == 0).sum())})
    grand_t.append(tgt); grand_n.append(non)

    fig, axes = plt.subplots(1, len(SHOW), figsize=(12, 3.5), sharey=True)
    for ax, ch in zip(axes, SHOW):
        i = chans.index(ch)
        ax.plot(times, tgt[i], color="tab:red", label="Target")
        ax.plot(times, non[i], color="tab:gray", label="Non-target")
        ax.axvspan(*WIN, color="tab:blue", alpha=0.1)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_title(ch); ax.set_xlabel("Time after flash (ms)")
    axes[0].set_ylabel("Amplitude (µV)"); axes[0].legend()
    fig.suptitle(f"{sub}: average ERP (both rc runs)")
    fig.tight_layout(); fig.savefig(FIG / f"{sub}_erp.png", dpi=150); plt.close(fig)
    print("saved", sub)

gt, gn = np.mean(grand_t, 0), np.mean(grand_n, 0)
i = chans.index("Pz")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(times, gt[i], color="tab:red", label="Target")
ax.plot(times, gn[i], color="tab:gray", label="Non-target")
ax.axvspan(*WIN, color="tab:blue", alpha=0.1, label="P300 window")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Time after flash (ms)"); ax.set_ylabel("Pz amplitude (µV)")
ax.set_title(f"Grand average ERP at Pz (n = {len(subs)})"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "grand_average_erp.png", dpi=150); plt.close(fig)

df = pd.DataFrame(rows)
df.to_csv(RES / "erp_summary.csv", index=False)
print("\n", df.to_string(index=False))