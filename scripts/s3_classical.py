"""Task 2 - LS + 2-D interpolation, estimated / ideal MMSE and ideal ALMMSE over the test SNRs.

    python scripts/s3_classical.py                         # every channel x every pilot pattern
    python scripts/s3_classical.py --channel SUI5 --pilots 48

Writes results/tables/classical_<channel>_P<pilots>.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chnet import settings  # noqa: E402
from chnet.evaluate import classical_sweep  # noqa: E402


def main():
    ap = settings.parser(__doc__)
    ap.add_argument("--channel")
    ap.add_argument("--pilots", type=int)
    args = ap.parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s3_classical")
    for ch in [args.channel] if args.channel else cfg["channel"]["models"]:
        for P in [args.pilots] if args.pilots else sorted(cfg["pilots"]["patterns"], reverse=True):
            log(f"--- classical estimators: {ch}, {P} pilots")
            classical_sweep(cfg, ch, P, log)


if __name__ == "__main__":
    main()
