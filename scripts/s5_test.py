"""Task 6 - TEST the saved networks (no training) and build the result tables.

  * every trained ChannelNet (and its SRCNN-only output) over the test SNRs
  * merged with the classical results (computed here if s3_classical.py has not been run)
  * MSE-vs-#pilots table at cfg['snr']['pilot_study_db']

    python scripts/s5_test.py --cfg configs/experiment.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chnet import rng, settings  # noqa: E402
from chnet.evaluate import classical_sweep, merge, network_sweep, pilot_study, table  # noqa: E402
from chnet.training import device_of  # noqa: E402


def main():
    args = settings.parser(__doc__).parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s5_test")
    rng.seed_torch(int(cfg["seed"]), bool(cfg["train"]["deterministic"]))
    dev = device_of(cfg)
    for ch, P in dict.fromkeys((c, p) for c, p, _ in settings.runs(cfg)):
        log(f"--- networks: {ch}, {P} pilots")
        network_sweep(cfg, ch, P, dev, log)
    for ch in cfg["channel"]["models"]:
        for P in cfg["pilots"]["patterns"]:
            if not table(cfg, f"classical_{ch}_P{P}.csv").exists():
                classical_sweep(cfg, ch, P, log)
            merge(cfg, ch, P)
        for r in pilot_study(cfg, ch):
            log(ch, {k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
