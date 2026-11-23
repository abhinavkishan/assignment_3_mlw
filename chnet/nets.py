"""ChannelNet = SRCNN (super-resolution, Task 3) -> DnCNN (image restoration, Task 4).

Input : the low-resolution image, i.e. the LS pilot values interpolated onto the 72 x 14 grid
        (real and imaginary parts are separate 1-channel images)
Output: the full-resolution channel image.          H_hat = f_IR( f_SR( interp(H_LS) ) )

SRCNN (Dong et al., TPAMI 2016), 9-1-5 configuration
    9x9 conv, 64 maps, ReLU   patch extraction
    1x1 conv, 32 maps, ReLU   non-linear mapping
    5x5 conv,  1 map          reconstruction                                 8 129 parameters
DnCNN (Zhang et al., TIP 2017), depth 20, residual learning
    3x3 conv 64 + ReLU
    18 x [3x3 conv 64 (no bias) + BatchNorm + ReLU]
    3x3 conv 1  -> residual R(y);  output x_hat = y - R(y)                  ~0.67 M parameters
Zero 'same' padding keeps every feature map at 72 x 14.
"""
from __future__ import annotations

import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, c1=64, c2=32, k1=9, k2=1, k3=5):
        super().__init__()
        self.extract = nn.Conv2d(1, c1, k1, padding=k1 // 2)
        self.mapping = nn.Conv2d(c1, c2, k2, padding=k2 // 2)
        self.rebuild = nn.Conv2d(c2, 1, k3, padding=k3 // 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rebuild(self.act(self.mapping(self.act(self.extract(x)))))


class DnCNN(nn.Module):
    def __init__(self, depth=20, width=64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            for _ in range(depth - 2)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail.weight)          # R(y) = 0 at start: the network begins as identity

    def residual(self, y):
        return self.tail(self.blocks(self.head(y)))

    def forward(self, y):
        return y - self.residual(y)               # residual learning


class ChannelNet(nn.Module):
    def __init__(self, sr: SRCNN, ir: DnCNN):
        super().__init__()
        self.sr, self.ir = sr, ir

    def forward(self, x):
        s = self.sr(x)
        return self.ir(s), s                      # (final estimate, SR-only estimate)


def make_sr(cfg: dict) -> SRCNN:
    return SRCNN(**cfg["nets"]["srcnn"])


def make_ir(cfg: dict) -> DnCNN:
    return DnCNN(**cfg["nets"]["dncnn"])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------- earlier checkpoint layout
def convert_old_sr(state: dict) -> dict:
    """Checkpoints of the first version of this project stored SRCNN as one nn.Sequential 'body'."""
    names = {"body.0": "extract", "body.2": "mapping", "body.4": "rebuild"}
    return {names[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]: v for k, v in state.items()}


def convert_old_ir(state: dict) -> dict:
    """Old DnCNN 'body' indices -> head / blocks.i.{0 conv, 1 bn} / tail."""
    convs = sorted({int(k.split(".")[1]) for k in state if k.startswith("body.")})
    last = convs[-1]
    out = {}
    for k, v in state.items():
        _, idx, *rest = k.split(".")
        idx, rest = int(idx), ".".join(rest)
        if idx == 0:
            out[f"head.0.{rest}"] = v
        elif idx == last:
            out[f"tail.{rest}"] = v
        else:                                     # idx = 2 + 3 i (conv) or 3 + 3 i (batch-norm)
            i, r = divmod(idx - 2, 3)
            out[f"blocks.{i}.{r}.{rest}"] = v
    return out
