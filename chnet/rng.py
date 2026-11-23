"""Reproducibility.

All randomness comes from ``numpy.random.SeedSequence([master_seed, h(tag_1), h(tag_2), ...])``,
where ``h`` is a SHA-256 based hash of the tag. The same tags therefore always give the same stream,
on any machine and in any order of execution. Example: the noise added to the VehA test pilots at
22 dB is ``stream(cfg, "noise", "VehA", "test", 48, "22")`` - every estimator, classical or deep,
is evaluated on exactly these noisy observations.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def _h(tag) -> int:
    return int.from_bytes(hashlib.sha256(str(tag).encode()).digest()[:4], "little")


def stream(cfg: dict, *tags) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(cfg["seed"])] + [_h(t) for t in tags]))


def int_seed(cfg: dict, *tags) -> int:
    """A 31-bit integer seed for torch, derived like ``stream``."""
    return int(stream(cfg, "int-seed", *tags).integers(0, 2 ** 31 - 1))


def seed_torch(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch; optionally force deterministic cuDNN kernels."""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
