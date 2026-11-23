"""One training stage (used for both SR and IR).

Loss         : MSE between the network output and the normalised ground-truth image
               SR stage  C1 = 1/|T| sum_i || f_SR(X_i) - H_i ||^2                          (7)
               IR stage  C2 = 1/|T| sum_i || f_IR(f_SR(X_i)) - H_i ||^2, f_SR frozen        (8)
Optimiser    : Adam, learning rate 1e-3, batch 128 (paper)
Schedule     : learning rate halved when the validation loss stalls (ReduceLROnPlateau)
Stopping     : early stopping on the validation loss, best weights restored at the end
Precision    : optional fp16 autocast on CUDA
History      : list of {epoch, train, val, lr, sec} -> CSV -> loss-curve figures
"""
from __future__ import annotations

import copy
import csv
import math
import time

import numpy as np
import torch
from torch import nn


def device_of(cfg: dict) -> torch.device:
    want = cfg["train"]["device"]
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(want)


def _t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)


@torch.no_grad()
def run(model: nn.Module, x: np.ndarray, dev, batch: int = 2048, fp16: bool = False) -> np.ndarray:
    """Batched inference on a numpy array (N, 1, K, L)."""
    model.eval().to(dev)
    fp16 = fp16 and dev.type == "cuda"
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            y = model(_t(x[i:i + batch], dev))
        y = y[0] if isinstance(y, tuple) else y
        out.append(y.float().cpu().numpy())
    return np.concatenate(out)


class StageTrainer:
    def __init__(self, model: nn.Module, hp: dict, cfg: dict, dev, seed: int, name: str, log=print):
        self.model, self.hp, self.dev, self.name, self.log = model.to(dev), hp, dev, name, log
        self.batch = int(cfg["train"]["batch"])
        self.vbatch = int(cfg["train"]["infer_batch"])
        self.fp16 = bool(cfg["train"]["mixed_precision"]) and dev.type == "cuda"
        self.shuffle = torch.Generator().manual_seed(seed)
        self.opt = torch.optim.Adam(model.parameters(), lr=float(hp["lr"]))
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, factor=float(hp["lr_drop"]), patience=int(hp["lr_wait"]), min_lr=float(hp["lr_min"]))
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)
        except (AttributeError, TypeError):                       # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.loss = nn.MSELoss()

    def _epoch(self, x, y) -> float:
        self.model.train()
        order = torch.randperm(len(x), generator=self.shuffle).to(self.dev)
        total = 0.0
        for s in range(0, len(x), self.batch):
            idx = order[s:s + self.batch]
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.fp16):
                out = self.model(x[idx])
            loss = self.loss(out.float(), y[idx])
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            total += loss.item() * len(idx)
        return total / len(x)

    @torch.no_grad()
    def _validate(self, x, y) -> float:
        self.model.eval()
        total = 0.0
        for s in range(0, len(x), self.vbatch):
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.fp16):
                out = self.model(x[s:s + self.vbatch])
            total += self.loss(out.float(), y[s:s + self.vbatch]).item() * len(out)
        return total / len(x)

    def fit(self, x_tr, y_tr, x_va, y_va):
        # 72 x 14 images are small: the whole data set lives on the device
        x_tr, y_tr, x_va, y_va = (_t(a, self.dev) for a in (x_tr, y_tr, x_va, y_va))
        best, best_state, best_ep, wait, hist = math.inf, None, 0, 0, []
        self.log(f"[{self.name}] {len(x_tr)} train / {len(x_va)} val images, fp16={self.fp16}")
        for ep in range(1, int(self.hp["epochs"]) + 1):
            t0 = time.time()
            tr = self._epoch(x_tr, y_tr)
            va = self._validate(x_va, y_va)
            self.sched.step(va)
            lr = self.opt.param_groups[0]["lr"]
            hist.append({"epoch": ep, "train": tr, "val": va, "lr": lr, "sec": time.time() - t0})
            better = va < best * (1 - 1e-4)
            if better:
                best, best_ep, wait = va, ep, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
            self.log(f"[{self.name}] ep {ep:3d} train {tr:.4e} val {va:.4e} lr {lr:.1e}"
                     f"{'  <- best' if better else ''}")
            if wait >= int(self.hp["stop_after"]):
                self.log(f"[{self.name}] early stop at epoch {ep}")
                break
        self.model.load_state_dict(best_state)
        return hist, {"best_val": best, "best_epoch": best_ep, "epochs": len(hist),
                      "minutes": sum(h["sec"] for h in hist) / 60}


def save_history(hist: list, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
