"""Task 1 - build the ground-truth channel data sets (VehA, SUI5) and record their statistics.

    python scripts/s1_make_dataset.py --cfg configs/experiment.yaml

Writes data/H_<model>_<source>_<N>_s<seed>.npy  (N, 72, 14) complex64
       results/tables/channel_statistics.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from chnet import channels, settings  # noqa: E402
from chnet.data import all_channels, split  # noqa: E402
from chnet.ofdm_system import LTEGrid  # noqa: E402


def lag_corr(H, axis, lag):
    """Normalised correlation E[H(x) H*(x+lag)] / E|H|^2 along one axis (1 = freq, 2 = time)."""
    a = np.take(H, range(H.shape[axis] - lag), axis=axis)
    b = np.take(H, range(lag, H.shape[axis]), axis=axis)
    return float(np.abs(np.mean(a * b.conj())) / np.mean(np.abs(H) ** 2))


def main():
    args = settings.parser(__doc__).parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s1_make_dataset")
    grid = LTEGrid.from_cfg(cfg)
    log(f"grid {grid.shape}, fs {grid.fs / 1e6:.2f} MHz, sub-frame {grid.n_samples} samples")
    stats = {}
    for ch in cfg["channel"]["models"]:
        H = all_channels(cfg, ch, log)
        s = channels.summary(ch, cfg, grid)
        s.update(source=cfg["channel"]["source"], frames=len(H),
                 train=len(split(cfg, ch, "train")), val=len(split(cfg, ch, "val")),
                 test=len(split(cfg, ch, "test")), power=float(np.mean(np.abs(H) ** 2)),
                 freq_corr_6sc=lag_corr(H, 1, 6), time_corr_13sym=lag_corr(H, 2, 13))
        stats[ch] = s
        log(ch, json.dumps(s))
    settings.results_path(cfg, "tables", "channel_statistics.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
