"""Every figure of the report.  Reads the CSV tables / training histories; computes nothing heavy."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import read_csv, table  # noqa: E402
from .settings import results_path  # noqa: E402

# method -> (label, colour, marker, line style)
LOOK = {
    "LS_linear":  ("LS + linear interp.",          "#7f7f7f", "s", "--"),
    "LS_cubic":   ("LS + cubic interp.",           "#bcbd22", "d", "--"),
    "LS_rbf":     ("LS + RBF interp. (net input)", "#8c564b", "v", "--"),
    "MMSE_est":   ("Estimated MMSE",               "#ff7f0e", "^", "-."),
    "ALMMSE":     ("Ideal ALMMSE",                 "#e377c2", "D", "-."),
    "MMSE_ideal": ("Ideal MMSE",                   "#000000", "o", "-"),
    "SR_12":      ("SRCNN only (12 dB)",           "#2ca02c", None, ":"),
    "SR_22":      ("SRCNN only (22 dB)",           "#d62728", None, ":"),
    "CN_12":      ("Deep low-SNR ChannelNet (12 dB)",  "#2ca02c", "*", "-"),
    "CN_22":      ("Deep high-SNR ChannelNet (22 dB)", "#d62728", "P", "-"),
    "CN_mix":     ("ChannelNet, multi-SNR 0-30 dB",    "#9467bd", "h", "-"),
}
SNR_FIG = ["LS_linear", "LS_rbf", "MMSE_est", "ALMMSE", "MMSE_ideal", "SR_12", "SR_22",
           "CN_12", "CN_22", "CN_mix"]


def _save(fig, cfg, name):
    p = results_path(cfg, "figures", name)
    fig.savefig(p, dpi=int(cfg["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return p


def _curve(ax, x, rows, key):
    if not rows or key not in rows[0]:
        return
    y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    ok = ~np.isnan(y)                     # e.g. no network trained for one pilot density
    if not ok.any():
        return
    lab, col, mk, ls = LOOK.get(key, (key, None, "o", "-"))
    ax.semilogy(np.asarray(x, dtype=float)[ok], y[ok], ls=ls, marker=mk, color=col, label=lab, ms=5, lw=1.4)


# ------------------------------------------------------------------------------- data
def pilot_patterns(cfg, masks: dict):
    fig, axs = plt.subplots(1, len(masks), figsize=(2.2 * len(masks), 4.6), sharey=True)
    for ax, (P, m) in zip(np.atleast_1d(axs), sorted(masks.items(), reverse=True)):
        ax.imshow(m, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1.2,
                  extent=[-0.5, m.shape[1] - 0.5, -0.5, m.shape[0] - 0.5])
        ax.set_title(f"{P} pilots")
        ax.set_xlabel("OFDM symbol")
    np.atleast_1d(axs)[0].set_ylabel("sub-carrier")
    fig.tight_layout()
    return _save(fig, cfg, "pilot_patterns.png")


def ground_truth(cfg, H, ch):
    """Mandatory plot 1 (paper Fig. 2/Fig. 1): real and imaginary channel images."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.8))
    for ax, img, nm in zip(axs, (H.real, H.imag), ("real part", "imaginary part")):
        im = ax.imshow(img, origin="lower", aspect="auto", cmap="viridis")
        ax.set(title=f"{ch}: {nm}", xlabel="OFDM symbol (time)", ylabel="sub-carrier (frequency)")
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout()
    return _save(fig, cfg, f"fig2_ground_truth_{ch}.png")


def lowres_example(cfg, truth, lowres, mask, ch, snr):
    fig, axs = plt.subplots(1, 3, figsize=(9, 4.4), sharey=True)
    vmin, vmax = truth.real.min(), truth.real.max()
    pk, pl = np.nonzero(mask)
    for ax, img, nm in zip(axs, (truth.real, np.where(mask, lowres.real, np.nan), lowres.real),
                           ("ground truth", f"LS pilots ({snr} dB)", "interpolated input")):
        ax.imshow(img, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set(title=nm, xlabel="OFDM symbol")
    axs[1].scatter(pl, pk, s=4, c="r")
    axs[0].set_ylabel("sub-carrier")
    fig.suptitle(f"{ch}: real part")
    fig.tight_layout()
    return _save(fig, cfg, f"lowres_example_{ch}.png")


# ------------------------------------------------------------------------------- training
def loss_curves(cfg, ch, P, t, hist_sr, hist_ir):
    """Mandatory plot 2: training/validation loss of both stages."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, h, nm in zip(axs, (hist_sr, hist_ir), ("SR stage (SRCNN)", "IR stage (DnCNN)")):
        if h:
            e = [r["epoch"] for r in h]
            ax.semilogy(e, [r["train"] for r in h], label="training")
            ax.semilogy(e, [r["val"] for r in h], label="validation")
            ax.legend()
        ax.set(title=nm, xlabel="epoch", ylabel="MSE (normalised)")
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{ch}, {P} pilots, trained at {t}{'' if t == 'mix' else ' dB'}")
    fig.tight_layout()
    return _save(fig, cfg, f"loss_curves/loss_{ch}_p{P}_snr{t}.png")


# ------------------------------------------------------------------------------- results
def mse_vs_snr(cfg, ch, P, fname, methods=SNR_FIG, title=None):
    f = table(cfg, f"mse_{ch}_P{P}.csv")
    if not f.exists():
        return None
    rows = read_csv(f)
    snr = [r["snr"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for m in methods:
        _curve(ax, snr, rows, m)
    ax.set(xlabel="SNR (dB)", ylabel="MSE", title=title or f"{ch}, {P} pilots")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    return _save(fig, cfg, fname)


def high_snr_pair(cfg, P):
    """Mandatory plot 4: deep high-SNR network vs baselines, VehA and SUI5 side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, ch in zip(axs, ("VehA", "SUI5")):
        f = table(cfg, f"mse_{ch}_P{P}.csv")
        if not f.exists():
            continue
        rows = read_csv(f)
        snr = [r["snr"] for r in rows]
        for m in ("LS_linear", "MMSE_est", "ALMMSE", "MMSE_ideal", "SR_22", "CN_22"):
            _curve(ax, snr, rows, m)
        ax.axvline(22, color="grey", lw=0.8, ls=":")
        ax.set(title=f"{ch}, {P} pilots", xlabel="SNR (dB)")
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel("MSE")
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    return _save(fig, cfg, "high_snr_VehA_vs_SUI5.png")


def pilot_sweep(cfg):
    """Mandatory plot 5 (paper Fig. 5): MSE vs number of pilots at a fixed SNR."""
    snr = int(cfg["snr"]["pilot_study_db"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, ch in zip(axs, ("VehA", "SUI5")):
        f = table(cfg, f"pilots_{ch}_snr{snr}.csv")
        if not f.exists():
            continue
        rows = read_csv(f)
        P = [int(r["pilots"]) for r in rows]
        for m in ("LS_linear", "MMSE_est", "ALMMSE", "MMSE_ideal", "SR_22", "CN_22"):
            _curve(ax, P, rows, m)
        ax.set(title=f"{ch}, SNR = {snr} dB", xlabel="number of pilots", ylabel="MSE")
        ax.set_xticks(sorted(set(P)))
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return _save(fig, cfg, f"fig5_mse_vs_pilots_snr{snr}.png")


def generalisation(cfg, P):
    """Gap (dB) of each ChannelNet to the ideal MMSE across SNR: shows the SNR-mismatch problem."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, ch in zip(axs, ("VehA", "SUI5")):
        f = table(cfg, f"mse_{ch}_P{P}.csv")
        if not f.exists():
            continue
        rows = read_csv(f)
        snr = np.array([r["snr"] for r in rows])
        ref = np.array([r["MMSE_ideal"] for r in rows])
        for m in ("CN_12", "CN_22", "CN_mix"):
            if m in rows[0]:
                lab, col, mk, _ = LOOK[m]
                ax.plot(snr, 10 * np.log10(np.array([r[m] for r in rows]) / ref), marker=mk, color=col, label=lab)
        ax.set(title=ch, xlabel="SNR (dB)")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("MSE / MSE(ideal MMSE)  [dB]")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, cfg, "snr_generalization.png")


def reconstruction(cfg, ch, P):
    f = table(cfg, f"example_{ch}_P{P}.npz")
    if not f.exists():
        return None
    d = np.load(f)
    fig, axs = plt.subplots(1, 4, figsize=(11, 4.2), sharey=True)
    lo, hi = d["truth"].real.min(), d["truth"].real.max()
    for ax, k, nm in zip(axs, ("truth", "lowres", "sr", "cn"),
                         ("ground truth", "RBF input", "SRCNN output", "ChannelNet output")):
        ax.imshow(d[k].real, origin="lower", aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
        ax.set(title=nm, xlabel="OFDM symbol")
    axs[0].set_ylabel("sub-carrier")
    fig.suptitle(f"{ch}, {P} pilots, real part")
    fig.tight_layout()
    return _save(fig, cfg, f"example_reconstruction_{ch}.png")
