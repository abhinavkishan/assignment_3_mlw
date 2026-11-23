"""Evaluation (Task 6).  All methods at a given (channel, #pilots, SNR) see the *same* noisy pilots.

Tables written to results/tables/
    classical_<ch>_P<P>.csv   snr, LS_linear, LS_cubic, LS_rbf, MMSE_est, ALMMSE, MMSE_ideal
    nets_<ch>_P<P>.csv        snr, SR_12, CN_12, SR_22, CN_22, SR_mix, CN_mix, ...
    mse_<ch>_P<P>.csv         both merged  (MSE vs SNR, paper Figs. 3 and 4)
    pilots_<ch>_snr<S>.csv    #pilots x methods at one SNR (paper Fig. 5)
"""
from __future__ import annotations

import csv
import json

import numpy as np

from . import rng
from .classical import ClassicalBank, apply, interp_operator, mse
from .data import from_images, pilots_ls, split, to_images
from .ofdm_system import LTEGrid, pattern
from .settings import model_dir, results_path, runs, tag


# ------------------------------------------------------------------------------- csv
def write_csv(path, rows: list) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: float(v) if v not in ("", None) else np.nan for k, v in r.items()}
                for r in csv.DictReader(fh)]


def table(cfg, name):
    return results_path(cfg, "tables", name)


# ------------------------------------------------------------------------------- classical
def classical_sweep(cfg: dict, ch: str, P: int, log=print) -> list:
    mask = pattern(cfg, LTEGrid.from_cfg(cfg), P)
    H_te = split(cfg, ch, "test", log)
    bank = ClassicalBank(cfg, mask, split(cfg, ch, "train", log),
                         lambda s: rng.stream(cfg, "cov", ch, P, str(s)))
    rows = []
    for snr in cfg["snr"]["test_db"]:
        hp = pilots_ls(cfg, H_te, ch, "test", P, snr)
        row = {"snr": snr, **{k: mse(v, H_te) for k, v in bank.estimate(hp, snr).items()}}
        rows.append(row)
        log(f"{ch} P={P} {snr:>2} dB  " + "  ".join(f"{k}={v:.2e}" for k, v in row.items() if k != "snr"))
    write_csv(table(cfg, f"classical_{ch}_P{P}.csv"), rows)
    return rows


# ------------------------------------------------------------------------------- networks
def load_net(cfg: dict, ch: str, P: int, snr, dev):
    """Returns (ChannelNet, info dict) or None when the network has not been trained."""
    import torch

    from .nets import ChannelNet, make_ir, make_sr
    d = model_dir(cfg, ch, P, snr)
    files = [d / "sr.pt", d / "ir.pt", d / "info.json"]
    if not all(f.exists() for f in files):
        return None
    sr, ir = make_sr(cfg), make_ir(cfg)
    sr.load_state_dict(torch.load(files[0], map_location=dev, weights_only=True))
    ir.load_state_dict(torch.load(files[1], map_location=dev, weights_only=True))
    return ChannelNet(sr, ir).to(dev).eval(), json.loads(files[2].read_text())


def net_estimate(cfg, net, info, W_in, hp, dev):
    """LS pilots -> interpolated image -> SRCNN -> DnCNN.  Returns (H_sr, H_cn)."""
    import torch
    shape = LTEGrid.from_cfg(cfg).shape
    sc = float(info["scale"])
    x = to_images(apply(W_in, hp, shape), sc)
    bs, fp16 = int(cfg["train"]["infer_batch"]), bool(cfg["train"]["mixed_precision"]) and dev.type == "cuda"
    sr_out, cn_out = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i:i + bs]).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=fp16):
                cn, s = net(xb)
            sr_out.append(s.float().cpu().numpy())
            cn_out.append(cn.float().cpu().numpy())
    return from_images(np.concatenate(sr_out), sc), from_images(np.concatenate(cn_out), sc)


def network_sweep(cfg: dict, ch: str, P: int, dev, log=print) -> list:
    mask = pattern(cfg, LTEGrid.from_cfg(cfg), P)
    H_te = split(cfg, ch, "test", log)
    nets = {}
    for _, _, s in runs(cfg, ch, P):
        loaded = load_net(cfg, ch, P, s, dev)
        if loaded is None:
            log(f"no trained network for {ch} P={P} snr {tag(s)} - skipped")
        else:
            nets[tag(s)] = loaded
    if not nets:
        return []
    W_in = interp_operator(mask, cfg["classical"]["net_input"])
    rows, example = [], None
    for snr in cfg["snr"]["test_db"]:
        hp = pilots_ls(cfg, H_te, ch, "test", P, snr)
        row = {"snr": snr}
        for t, (net, info) in nets.items():
            H_sr, H_cn = net_estimate(cfg, net, info, W_in, hp, dev)
            row[f"SR_{t}"], row[f"CN_{t}"] = mse(H_sr, H_te), mse(H_cn, H_te)
            if snr == cfg["figures"]["example_snr"] and t == tag(snr):
                example = dict(truth=H_te[0], lowres=apply(W_in, hp[:1], mask.shape)[0],
                               sr=H_sr[0], cn=H_cn[0])
        rows.append(row)
        log(f"{ch} P={P} {snr:>2} dB  " + "  ".join(f"{k}={v:.2e}" for k, v in row.items() if k != "snr"))
    write_csv(table(cfg, f"nets_{ch}_P{P}.csv"), rows)
    if example:
        np.savez(table(cfg, f"example_{ch}_P{P}.npz"), **example)
    return rows


# ------------------------------------------------------------------------------- merging
def merge(cfg: dict, ch: str, P: int) -> list:
    c, n = table(cfg, f"classical_{ch}_P{P}.csv"), table(cfg, f"nets_{ch}_P{P}.csv")
    if not c.exists():
        return []
    dl = {r["snr"]: r for r in read_csv(n)} if n.exists() else {}
    rows = [{**r, **dl.get(r["snr"], {})} for r in read_csv(c)]
    write_csv(table(cfg, f"mse_{ch}_P{P}.csv"), rows)
    return rows


def pilot_study(cfg: dict, ch: str) -> list:
    snr = float(cfg["snr"]["pilot_study_db"])
    rows = []
    for P in sorted(cfg["pilots"]["patterns"], reverse=True):
        f = table(cfg, f"mse_{ch}_P{P}.csv")
        if f.exists():
            rows += [{"pilots": P, **{k: v for k, v in r.items() if k != "snr"}}
                     for r in read_csv(f) if abs(r["snr"] - snr) < 1e-9]
    if rows:
        write_csv(table(cfg, f"pilots_{ch}_snr{int(snr)}.csv"), rows)
    return rows
