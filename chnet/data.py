"""Data sets: ground-truth channels, splits, noisy LS pilots, and the image representation.

* 40 000 channel grids per model, generated once and cached in data/ (or read from Vienna .mat)
* split 32 000 / 4 000 / 4 000 (train / validation / test), as in the paper
* the noisy LS pilots of (channel, split, #pilots, SNR) always come from the same random stream
* a complex grid becomes two real single-channel images (real part, imaginary part), divided by
  one global scale (std of the real/imaginary entries of the training channels, ~0.707)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import channels, rng
from .ofdm_system import LTEGrid, ls_at_pilots, pattern

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------------------- channels
def _cache_file(cfg: dict, name: str) -> Path:
    src = cfg["channel"]["source"]
    return Path(cfg["dirs"]["data"]) / f"H_{name}_{src}_{cfg['data']['frames']}_s{cfg['seed']}.npy"


def _read_vienna(cfg: dict, name: str, grid: LTEGrid) -> np.ndarray:
    """Read the [N x 72 x 14] variable H written by matlab/make_vienna_dataset.m (any number of files)."""
    from .settings import ROOT
    parts = []
    for f in cfg["channel"]["vienna_files"][name]:
        f = Path(f) if Path(f).is_absolute() else ROOT / f
        try:
            from scipy.io import loadmat
            H = np.asarray(loadmat(f)["H"])
        except NotImplementedError:                       # -v7.3 file (HDF5)
            import h5py
            with h5py.File(f, "r") as fh:
                raw = fh["H"][()]
            H = np.transpose(raw["real"] + 1j * raw["imag"])
        if H.shape[1:] != grid.shape:
            H = np.moveaxis(H, -1, 0)                     # stored as 72 x 14 x N
        parts.append(H)
    H = np.concatenate(parts)
    return (H / np.sqrt(np.mean(np.abs(H) ** 2))).astype(np.complex64)   # E|H|^2 = 1


def all_channels(cfg: dict, name: str, log=print) -> np.ndarray:
    f = _cache_file(cfg, name)
    if f.exists():
        return np.load(f)
    grid = LTEGrid.from_cfg(cfg)
    if cfg["channel"]["source"] == "python":
        n = int(cfg["data"]["frames"])
        log(f"generating {n} {name} channels")
        H = channels.generate(name, n, grid, cfg, rng.stream(cfg, "channel", name))
    elif cfg["channel"]["source"] == "vienna":
        log(f"reading Vienna channels for {name}")
        H = _read_vienna(cfg, name, grid)
    else:
        raise ValueError(cfg["channel"]["source"])
    np.save(f, H)
    log(f"cached {H.shape} in {f}")
    return H


def split(cfg: dict, name: str, part: str, log=print) -> np.ndarray:
    """One of the three consecutive blocks train / val / test (frames are i.i.d.)."""
    H = all_channels(cfg, name, log)
    n_tr, n_va, n_te = (int(v) for v in cfg["data"]["split"])
    if n_tr + n_va + n_te > len(H):
        raise ValueError("split larger than the number of channel frames")
    blocks = {"train": H[:n_tr], "val": H[n_tr:n_tr + n_va], "test": H[n_tr + n_va:n_tr + n_va + n_te]}
    out = blocks[part]
    return out[: int(cfg["data"]["test_frames"])] if part == "test" else out


# ------------------------------------------------------------------------------- observations
def frame_snrs(cfg: dict, snr, n: int, gen: np.random.Generator):
    """A fixed SNR, or for 'mix' one SNR per frame drawn uniformly from cfg['snr']['mix_db']."""
    if snr == "mix":
        lo, hi = cfg["snr"]["mix_db"]
        return gen.uniform(lo, hi, n)
    return float(snr)


def pilots_ls(cfg: dict, H: np.ndarray, name: str, part: str, n_pilots: int, snr) -> np.ndarray:
    """Noisy LS pilot estimates (N, P); identical for every caller with the same arguments."""
    grid = LTEGrid.from_cfg(cfg)
    gen = rng.stream(cfg, "noise", name, part, int(n_pilots), str(snr))
    return ls_at_pilots(H, pattern(cfg, grid, n_pilots), frame_snrs(cfg, snr, len(H), gen), gen)


# ------------------------------------------------------------------------------- images
def scale_of(H_train: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(H_train) ** 2) / 2))


def to_images(H: np.ndarray, scale: float) -> np.ndarray:
    """(N, K, L) complex -> (2N, 1, K, L) float32 = [real parts of all frames ; imaginary parts]."""
    return (np.concatenate([H.real, H.imag])[:, None] / scale).astype(np.float32)


def from_images(img: np.ndarray, scale: float) -> np.ndarray:
    n = img.shape[0] // 2
    return ((img[:n, 0] + 1j * img[n:, 0]) * scale).astype(np.complex64)
