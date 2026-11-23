"""All result figures (mandatory plots 2-5) from results/tables and results/logs.

    python scripts/s6_figures.py --cfg configs/experiment.yaml

results/figures/
    loss_curves/loss_<ch>_p<P>_snr<S>.png   SR + IR training / validation loss    (plot 2)
    fig3_VehA_mse_vs_snr.png                 paper Fig. 3                           (plot 3)
    fig4_SUI5_mse_vs_snr.png                 paper Fig. 4                           (plot 3)
    high_snr_VehA_vs_SUI5.png                deep high-SNR case, both channels      (plot 4)
    fig5_mse_vs_pilots_snr<S>.png            paper Fig. 5                           (plot 5)
    snr_generalization.png                   gap of the 12 / 22 / mix networks to the ideal MMSE
    example_reconstruction_<ch>.png          truth / input / SRCNN / ChannelNet images
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chnet import figures, settings  # noqa: E402
from chnet.evaluate import read_csv  # noqa: E402


def main():
    args = settings.parser(__doc__).parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s6_figures")
    P = int(cfg["pilots"]["main"])
    made = []
    for ch, p, s in settings.runs(cfg):
        t = settings.tag(s)
        h = [settings.results_path(cfg, "logs", f"history_{ch}_P{p}_snr{t}_{st}.csv") for st in ("sr", "ir")]
        if any(f.exists() for f in h):
            made.append(figures.loss_curves(cfg, ch, p, t, *[read_csv(f) if f.exists() else [] for f in h]))
    made.append(figures.mse_vs_snr(cfg, "VehA", P, "fig3_VehA_mse_vs_snr.png", title=f"VehA, {P} pilots (paper Fig. 3)"))
    made.append(figures.mse_vs_snr(cfg, "SUI5", P, "fig4_SUI5_mse_vs_snr.png", title=f"SUI5, {P} pilots (paper Fig. 4)"))
    made.append(figures.high_snr_pair(cfg, P))
    made.append(figures.pilot_sweep(cfg))
    made.append(figures.generalisation(cfg, P))
    for ch in cfg["channel"]["models"]:
        made.append(figures.reconstruction(cfg, ch, P))
    for m in filter(None, made):
        log(m)


if __name__ == "__main__":
    main()
