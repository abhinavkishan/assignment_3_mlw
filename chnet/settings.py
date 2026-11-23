"""Configuration: YAML file + optional parent file (``extends:``) + command-line overrides."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = ROOT / "configs" / "experiment.yaml"


def _merge(base: dict, new: dict) -> dict:
    """Recursive dictionary merge; values of ``new`` win."""
    out = copy.deepcopy(base)
    for key, val in new.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(_read(path.parent / parent), cfg)
    return cfg


def _override(cfg: dict, expr: str) -> None:
    """Apply ``a.b.c=value``; the value is parsed as YAML (numbers, lists, ...)."""
    key, sep, raw = expr.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got '{expr}'")
    node = cfg
    *parents, leaf = key.strip().split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def load(path=None, overrides=()) -> dict:
    path = Path(path or DEFAULT_CFG).resolve()
    cfg = _read(path)
    for o in overrides:
        _override(cfg, o)
    cfg["pilots"]["patterns"] = {int(k): v for k, v in cfg["pilots"]["patterns"].items()}
    # absolute output folders
    for name, d in cfg["dirs"].items():
        d = Path(d)
        cfg["dirs"][name] = str(d if d.is_absolute() else ROOT / d)
        Path(cfg["dirs"][name]).mkdir(parents=True, exist_ok=True)
    for sub in ("figures", "figures/loss_curves", "tables", "logs"):
        (Path(cfg["dirs"]["results"]) / sub).mkdir(parents=True, exist_ok=True)
    cfg["_file"] = str(path)
    return cfg


def parser(doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=str(DEFAULT_CFG), help="YAML configuration file")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="override a configuration entry (repeatable)")
    return ap


def runs(cfg: dict, channel=None, pilots=None, snr=None):
    """List of (channel, pilots, snr_spec) for every network in cfg['runs'] (optionally filtered)."""
    out = []
    for r in cfg["runs"]:
        for s in r["snr"]:
            if channel and r["channel"] != channel:
                continue
            if pilots and int(r["pilots"]) != int(pilots):
                continue
            if snr is not None and str(s) != str(snr):
                continue
            out.append((r["channel"], int(r["pilots"]), s))
    return out


def tag(snr) -> str:
    """Short label of a training SNR: 12 -> '12', 'mix' -> 'mix'."""
    if isinstance(snr, str):
        return snr
    return str(int(snr)) if float(snr).is_integer() else str(snr)


def results_path(cfg: dict, *parts) -> Path:
    return Path(cfg["dirs"]["results"]).joinpath(*parts)


def model_dir(cfg: dict, channel: str, pilots: int, snr) -> Path:
    d = Path(cfg["dirs"]["models"]) / f"{channel}_P{pilots}_snr{tag(snr)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Log:
    """Print a time-stamped line and append it to a log file."""

    def __init__(self, cfg: dict, name: str):
        self.file = results_path(cfg, "logs", f"{name}.log")

    def __call__(self, *items):
        import time
        line = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(i) for i in items)
        print(line, flush=True)
        with open(self.file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
