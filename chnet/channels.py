"""VehA and SUI5 channels on the 72 x 14 grid (WSSUS tapped-delay line, Rayleigh / Jakes fading).

Time-frequency response sampled at sub-carrier f_k and symbol centre t_l:

        H[k,l] = sum_p sqrt(P_p) g_p(t_l) exp(-j 2 pi f_k tau_p),       sum_p P_p = 1          (3)

g_p(t) is a unit-power complex Gaussian process with the Jakes (Clarke) autocorrelation
J0(2 pi f_D dt), produced with the statistically exact sum-of-sinusoids model of
Zheng & Xiao (IEEE Trans. Commun., 2003):

        g(t) = 1/sqrt(M) sum_{m=1..M} [ cos(w_D t cos a_m + phi_m) + j cos(w_D t sin a_m + psi_m) ]
        a_m  = (2 pi m - pi + theta) / (4 M),     theta, phi_m, psi_m ~ U[-pi, pi)          (4)

Each tap and each frame gets its own random theta / phi / psi, so frames are independent.

Power-delay profiles
  VehA (ITU-R M.1225) : 0, 310, 710, 1090, 1730, 2510 ns ; 0, -1, -9, -10, -15, -20 dB
  SUI5 (Erceg et al.) : 0, 4, 10 us ; 0, -5, -10 dB  (Rayleigh, K = 0)
Doppler: f_D = v f_c / c (50 km/h at 2.1 GHz -> 97.3 Hz), the set-up of the paper.
"""
from __future__ import annotations

import numpy as np

from .ofdm_system import LTEGrid

C0 = 299_792_458.0

PDP = {
    #        delays [s]                                         powers [dB]
    "VehA": (np.array([0, 310, 710, 1090, 1730, 2510]) * 1e-9, np.array([0, -1, -9, -10, -15, -20.0])),
    "SUI5": (np.array([0, 4, 10]) * 1e-6,                      np.array([0, -5, -10.0])),
}


def doppler_hz(cfg: dict) -> float:
    return cfg["channel"]["speed_kmh"] / 3.6 * float(cfg["channel"]["fc_hz"]) / C0


def taps(name: str):
    """(delays, linear powers normalised to sum 1)."""
    tau, p_db = PDP[name]
    p = 10 ** (p_db / 10)
    return tau.astype(float), p / p.sum()


def rms_delay(name: str) -> float:
    tau, p = taps(name)
    mean = np.sum(p * tau)
    return float(np.sqrt(np.sum(p * tau ** 2) - mean ** 2))


def summary(name: str, cfg: dict, grid: LTEGrid) -> dict:
    """Nominal channel parameters quoted in the report."""
    tau, _ = taps(name)
    trms, fd = rms_delay(name), doppler_hz(cfg)
    return {"taps": len(tau), "max_delay_us": tau.max() * 1e6, "rms_delay_us": trms * 1e6,
            "coherence_bw_kHz": 1 / (5 * trms) / 1e3, "doppler_Hz": fd,
            "coherence_time_ms": 0.423 / fd * 1e3, "cp_us": min(grid.cp) / grid.fs * 1e6}


def jakes(rng: np.random.Generator, shape, t: np.ndarray, fd: float, M: int) -> np.ndarray:
    """Unit-power Jakes fading, eq. (4). Returns complex array of shape (*shape, len(t))."""
    theta = rng.uniform(-np.pi, np.pi, size=(*shape, 1))
    phi = rng.uniform(-np.pi, np.pi, size=(*shape, M))
    psi = rng.uniform(-np.pi, np.pi, size=(*shape, M))
    a = (2 * np.pi * np.arange(1, M + 1) - np.pi + theta) / (4 * M)              # (*shape, M)
    wt = 2 * np.pi * fd * t                                                      # (T,)
    re = np.cos(np.cos(a)[..., None] * wt + phi[..., None]).sum(-2)
    im = np.cos(np.sin(a)[..., None] * wt + psi[..., None]).sum(-2)
    return (re + 1j * im) / np.sqrt(M)


def generate(name: str, n: int, grid: LTEGrid, cfg: dict, rng: np.random.Generator,
             chunk: int = 1000) -> np.ndarray:
    """n independent channel grids H (n, K, L), complex64, E|H|^2 = 1, eq. (3)."""
    tau, p = taps(name)
    steer = np.exp(-2j * np.pi * np.outer(grid.freqs, tau)) * np.sqrt(p)        # (K, P)
    fd, M = doppler_hz(cfg), int(cfg["channel"]["sinusoids"])
    H = np.empty((n, grid.K, grid.L), dtype=np.complex64)
    for s in range(0, n, chunk):
        b = min(chunk, n - s)
        g = jakes(rng, (b, len(tau)), grid.t_centre, fd, M)                     # (b, P, L)
        H[s:s + b] = np.einsum("kp,npl->nkl", steer, g)
    return H
