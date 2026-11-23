"""SISO-OFDM system on the LTE 1.4 MHz grid, lattice pilots and LS estimation.

Grid:  K = 72 active sub-carriers (k = -36..-1, +1..+36, DC unused), L = 14 OFDM symbols,
       15 kHz spacing, N_FFT = 128 (f_s = 1.92 MHz), normal CP of 10/9 samples (5.2/4.7 us).

Per resource element (paper eqs. (1)-(2); CP longer than the channel, channel constant within
one OFDM symbol):

        Y[k,l] = H[k,l] X[k,l] + Z[k,l],         Z ~ CN(0, sigma^2)                        (1)

With unit-energy symbols and E|H|^2 = 1:   SNR = 1 / sigma^2.
At a pilot position the least-squares estimate is

        H_LS[k_p,l_p] = Y[k_p,l_p] / X[k_p,l_p] = H[k_p,l_p] + Z / X                        (2)

which is unbiased with error variance sigma^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LTEGrid:
    K: int = 72
    L: int = 14
    spacing: float = 15e3
    nfft: int = 128
    cp: tuple = (10, 9, 9, 9, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9)

    @staticmethod
    def from_cfg(cfg: dict) -> "LTEGrid":
        o = cfg["ofdm"]
        return LTEGrid(int(o["subcarriers"]), int(o["symbols"]), float(o["spacing_hz"]),
                       int(o["nfft"]), tuple(int(c) for c in o["cp"]))

    # ---- frequency axis
    @property
    def k(self) -> np.ndarray:
        """Logical sub-carrier numbers, DC skipped."""
        half = self.K // 2
        return np.r_[np.arange(-half, 0), np.arange(1, half + 1)]

    @property
    def freqs(self) -> np.ndarray:
        return self.k * self.spacing

    @property
    def bins(self) -> np.ndarray:
        """FFT bin occupied by each active sub-carrier."""
        return self.k % self.nfft

    # ---- time axis
    @property
    def fs(self) -> float:
        return self.nfft * self.spacing

    @property
    def starts(self) -> np.ndarray:
        """Sample index at which each OFDM symbol (its CP) starts."""
        return np.r_[0, np.cumsum(np.array(self.cp) + self.nfft)[:-1]]

    @property
    def t_centre(self) -> np.ndarray:
        """Time (s) of the middle of the useful part of every symbol (fading sampled there)."""
        return (self.starts + np.array(self.cp) + self.nfft / 2) / self.fs

    @property
    def n_samples(self) -> int:
        return self.L * self.nfft + sum(self.cp)

    @property
    def shape(self):
        return (self.K, self.L)


# ------------------------------------------------------------------------------- pilots
def lattice(grid: LTEGrid, symbols, spacing, shifts) -> np.ndarray:
    """Boolean (K, L) mask of a lattice (diamond) pilot arrangement.

    On OFDM symbol ``symbols[i]`` the pilots occupy sub-carriers shifts[i], shifts[i]+spacing, ...
    Alternating shifts between pilot symbols give the staggered LTE-CRS-like diamond."""
    m = np.zeros(grid.shape, dtype=bool)
    for l, s in zip(symbols, shifts):
        m[int(s)::int(spacing), int(l)] = True
    return m


def pattern(cfg: dict, grid: LTEGrid, n_pilots: int) -> np.ndarray:
    spec = cfg["pilots"]["patterns"][int(n_pilots)]
    m = lattice(grid, spec["symbols"], spec["spacing"], spec["shifts"])
    if int(m.sum()) != int(n_pilots):
        raise ValueError(f"pattern '{n_pilots}' has {int(m.sum())} pilots")
    return m


def pilot_index(mask: np.ndarray) -> np.ndarray:
    """Positions of the pilots in the row-major vector vec(H)[k*L + l]."""
    return np.flatnonzero(mask.reshape(-1))


# ------------------------------------------------------------------------------- symbols
def qpsk(rng: np.random.Generator, shape) -> np.ndarray:
    """Unit-energy QPSK: (+-1 +- j)/sqrt(2)."""
    bits = rng.integers(0, 2, size=(2, *shape))
    return ((1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])) / np.sqrt(2.0)


def noise_var(snr_db):
    """sigma^2 = 10^(-SNR/10) for unit signal power."""
    return np.power(10.0, -np.asarray(snr_db, dtype=float) / 10.0)


# ------------------------------------------------------------------------------- OFDM link
def modulate(grid: LTEGrid, X: np.ndarray) -> np.ndarray:
    """Resource grid X (K, L) -> one time-domain sub-frame (unitary IFFT, cyclic prefix)."""
    out = []
    for l in range(grid.L):
        spec = np.zeros(grid.nfft, dtype=complex)
        spec[grid.bins] = X[:, l]
        sym = np.fft.ifft(spec) * np.sqrt(grid.nfft)
        out.append(np.r_[sym[-grid.cp[l]:], sym])
    return np.concatenate(out)


def demodulate(grid: LTEGrid, y: np.ndarray) -> np.ndarray:
    """Time-domain sub-frame -> resource grid Y (K, L) (CP removal, unitary FFT)."""
    Y = np.empty(grid.shape, dtype=complex)
    for l, (s0, cp) in enumerate(zip(grid.starts, grid.cp)):
        body = y[s0 + cp: s0 + cp + grid.nfft]
        Y[:, l] = (np.fft.fft(body) / np.sqrt(grid.nfft))[grid.bins]
    return Y


def time_domain_link(grid: LTEGrid, h_taps: np.ndarray, X: np.ndarray) -> np.ndarray:
    """IFFT + CP -> linear convolution with a static impulse response -> CP removal + FFT.
    Used by tests/test_ofdm_link.py to confirm that the per-sub-carrier model (1) is exact
    when the impulse response is shorter than the CP."""
    tx = modulate(grid, X)
    rx = np.convolve(tx, h_taps)[: tx.size]
    return demodulate(grid, rx)


# ------------------------------------------------------------------------------- LS at the pilots
def ls_at_pilots(H: np.ndarray, mask: np.ndarray, snr_db, rng: np.random.Generator) -> np.ndarray:
    """Send random QPSK pilots through H (N, K, L) and return the LS estimates (N, P), eq. (2).

    ``snr_db`` may be a scalar or one SNR per frame (multi-SNR training)."""
    n = H.shape[0]
    Hp = H.reshape(n, -1)[:, pilot_index(mask)]
    Xp = qpsk(rng, Hp.shape)
    s2 = np.broadcast_to(noise_var(snr_db), (n,))[:, None]
    Z = np.sqrt(s2 / 2.0) * (rng.standard_normal(Hp.shape) + 1j * rng.standard_normal(Hp.shape))
    Yp = Hp * Xp + Z                                       # eq. (1) at the pilots
    return (Yp / Xp).astype(np.complex64)                  # eq. (2)
