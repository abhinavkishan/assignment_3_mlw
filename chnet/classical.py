"""Conventional channel estimators (Task 2).

For a fixed pilot pattern every estimator below is linear in the LS pilot vector h_p (P x 1):

        vec(H_hat) = W h_p ,      W in C^{KL x P}   (vec = row-major, index k*L + l)

so each method is just a different matrix W, built once per (pattern, SNR).

1. LS + 2-D interpolation.  W = interpolation operator on the (sub-carrier, symbol) plane,
   obtained by interpolating the P unit vectors (interpolation is linear in the data):
   'linear'  Delaunay piecewise-linear,   'cubic'  Clough-Tocher,
   'rbf'     Gaussian radial basis functions exp(-(r/eps)^2), eps = mean node distance
             (the interpolation used in the ChannelNet reference code to build the input image).
   Points outside the convex hull of the pilots take the nearest pilot value.

2. LMMSE / Wiener (paper eq. (5)):
        W = R_hp (R_pp + sigma^2 I)^{-1},   R_hp = E[h h_p^H],  R_pp = E[h_p h_p^H]           (5)
   ideal MMSE     : R = sample covariance of the 32 000 true training channels (genie bound)
   estimated MMSE : R from 2 000 *received* frames: h_int = W_lin h_LS, R_hat = E[h_int h_int^H]
                    - sigma^2 W_lin W_lin^T (noise contribution removed). No genie knowledge.

3. Ideal ALMMSE (Simko, Mehlfuehrer, Wrulich, Rupp, WSA 2010):
        R_h ~ R_time (x) R_freq,  H_hat = B_freq H_int C_time^T                                (6)
   Eigen-decompositions R_freq = U_f diag(l_f) U_f^H, R_time = U_t diag(l_t) U_t^H keep N_freq / N_time
   eigenvalues; M = (l_t l_f^T) ./ (l_t l_f^T + sigma^2) is approximated by its best rank-one
   approximation s u v^T, giving   C_time = U_t diag(s u) U_t^H,  B_freq = U_f diag(v) U_f^H.
   Row-major vec:  vec(B X C^T) = (B (x) C) vec(X),  hence  W = (B (x) C) W_lin.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator, \
    RBFInterpolator

from .ofdm_system import LTEGrid, ls_at_pilots, noise_var, pilot_index


# ------------------------------------------------------------------------------- interpolation
def interp_operator(mask: np.ndarray, kind: str) -> np.ndarray:
    """Real (K*L, P) matrix that maps pilot values to the full grid."""
    K, L = mask.shape
    nodes = np.argwhere(mask).astype(float)                            # (P, 2), row-major order
    query = np.argwhere(np.ones((K, L), bool)).astype(float)            # (K*L, 2), row-major order
    basis = np.eye(len(nodes))
    if kind == "rbf":
        span = np.ptp(nodes, axis=0)
        span = span[span > 0]
        eps = float(np.prod(span) / len(nodes)) ** (1.0 / span.size)   # scipy.Rbf default width
        W = RBFInterpolator(nodes, basis, kernel="gaussian", epsilon=1.0 / eps)(query)
    elif kind in ("linear", "cubic"):
        cls = LinearNDInterpolator if kind == "linear" else CloughTocher2DInterpolator
        W = cls(nodes, basis)(query)
        outside = np.isnan(W).any(axis=1)
        W[outside] = NearestNDInterpolator(nodes, basis)(query[outside])
    elif kind == "nearest":
        W = NearestNDInterpolator(nodes, basis)(query)
    else:
        raise ValueError(f"unknown interpolation '{kind}'")
    return np.asarray(W, dtype=float)


def apply(W: np.ndarray, hp: np.ndarray, shape) -> np.ndarray:
    """Pilot vectors (N, P) -> grids (N, K, L)."""
    return (hp @ W.T).reshape(len(hp), *shape).astype(np.complex64)


# ------------------------------------------------------------------------------- statistics
def covariance(H: np.ndarray) -> np.ndarray:
    """E[vec(H) vec(H)^H] (zero-mean channel), shape (KL, KL)."""
    V = H.reshape(len(H), -1).astype(np.complex128)
    return V.T @ V.conj() / len(V)


def separable_covariances(H: np.ndarray):
    """Time (L x L) and frequency (K x K) correlation matrices used by the ALMMSE."""
    Hc = H.astype(np.complex128)
    n, K, L = Hc.shape
    R_freq = np.einsum("nal,nbl->ab", Hc, Hc.conj()) / (n * L)
    R_time = np.einsum("nka,nkb->ab", Hc, Hc.conj()) / (n * K)
    return R_time, R_freq


# ------------------------------------------------------------------------------- filters
def wiener(R: np.ndarray, pidx: np.ndarray, s2: float) -> np.ndarray:
    """eq. (5).  A tiny diagonal loading keeps R_pp + sigma^2 I well conditioned at high SNR."""
    R_hp = R[:, pidx]
    R_pp = R[np.ix_(pidx, pidx)]
    load = s2 + 1e-8 * np.real(np.trace(R_pp)) / len(pidx)
    A = R_pp + load * np.eye(len(pidx))
    return np.linalg.solve(A.T, R_hp.T).T                              # R_hp A^{-1}


def almmse(R_time, R_freq, W_lin, s2: float, n_time: int, n_freq: int) -> np.ndarray:
    """eq. (6): rank-one separable approximation of the LMMSE filter."""
    lt, Ut = np.linalg.eigh(R_time)
    lf, Uf = np.linalg.eigh(R_freq)
    it, jf = np.argsort(lt)[::-1][:n_time], np.argsort(lf)[::-1][:n_freq]
    lt, Ut = np.maximum(lt[it], 0), Ut[:, it]
    lf, Uf = np.maximum(lf[jf], 0), Uf[:, jf]
    G = np.outer(lt, lf)
    M = G / (G + s2)
    u, s, vh = np.linalg.svd(M)
    C_time = (Ut * (s[0] * u[:, 0])) @ Ut.conj().T
    B_freq = (Uf * vh[0]) @ Uf.conj().T
    return np.kron(B_freq, C_time) @ W_lin


# ------------------------------------------------------------------------------- bank
class ClassicalBank:
    """All conventional estimators for one channel model and one pilot pattern."""

    def __init__(self, cfg: dict, mask: np.ndarray, H_train: np.ndarray, cov_rng):
        """cov_rng(snr) must return the random stream used to simulate the 'estimated MMSE' frames."""
        self.cfg, self.mask = cfg, mask
        self.shape = LTEGrid.from_cfg(cfg).shape
        self.pidx = pilot_index(mask)
        kinds = list(dict.fromkeys(cfg["classical"]["interpolators"] + ["linear"]))
        self.W = {k: interp_operator(mask, k) for k in kinds}
        self.R = covariance(H_train)                                     # ideal (genie) statistics
        self.R_time, self.R_freq = separable_covariances(H_train)
        self.H_cov = H_train[: int(cfg["data"]["cov_frames"])]
        self.cov_rng = cov_rng

    def _estimated_R(self, snr: float) -> np.ndarray:
        hp = ls_at_pilots(self.H_cov, self.mask, snr, self.cov_rng(snr))
        Wl = self.W["linear"]
        R_int = covariance(apply(Wl, hp, self.shape))
        return R_int - float(noise_var(snr)) * (Wl @ Wl.T)

    def operators(self, snr: float) -> dict:
        """Name -> (KL, P) estimator matrix at this SNR."""
        s2 = float(noise_var(snr))
        e = self.cfg["classical"]["almmse_eigs"]
        ops = {f"LS_{k}": self.W[k] for k in self.cfg["classical"]["interpolators"]}
        ops["MMSE_est"] = wiener(self._estimated_R(snr), self.pidx, s2)
        ops["ALMMSE"] = almmse(self.R_time, self.R_freq, self.W["linear"], s2, int(e["time"]), int(e["freq"]))
        ops["MMSE_ideal"] = wiener(self.R, self.pidx, s2)
        return ops

    def estimate(self, hp: np.ndarray, snr: float) -> dict:
        return {name: apply(W, hp, self.shape) for name, W in self.operators(snr).items()}


def mse(H_hat: np.ndarray, H: np.ndarray) -> float:
    """Mean of |H_hat - H|^2 over all resource elements and frames (the paper's metric)."""
    return float(np.mean(np.abs(H_hat.astype(np.complex128) - H) ** 2))
