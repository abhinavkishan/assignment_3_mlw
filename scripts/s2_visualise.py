"""Task 1 plots - pilot patterns, ground-truth real/imag images (paper Fig. 2), low-res input example.

    python scripts/s2_visualise.py --cfg configs/experiment.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chnet import figures, settings  # noqa: E402
from chnet.classical import apply, interp_operator  # noqa: E402
from chnet.data import pilots_ls, split  # noqa: E402
from chnet.ofdm_system import LTEGrid, pattern  # noqa: E402


def main():
    args = settings.parser(__doc__).parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s2_visualise")
    grid = LTEGrid.from_cfg(cfg)
    masks = {P: pattern(cfg, grid, P) for P in cfg["pilots"]["patterns"]}
    log(figures.pilot_patterns(cfg, masks))
    P, snr = int(cfg["pilots"]["main"]), cfg["figures"]["example_snr"]
    W = interp_operator(masks[P], cfg["classical"]["net_input"])
    for ch in cfg["channel"]["models"]:
        H = split(cfg, ch, "test", log)[:1]
        log(figures.ground_truth(cfg, H[0], ch))
        lowres = apply(W, pilots_ls(cfg, H, ch, "figure", P, snr), grid.shape)[0]
        log(figures.lowres_example(cfg, H[0], lowres, masks[P], ch, snr))


if __name__ == "__main__":
    main()
