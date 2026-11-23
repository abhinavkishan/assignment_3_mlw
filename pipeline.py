"""Run the whole workflow in order (each step is a separate script and can also be run alone).

    python pipeline.py                                     # everything, configs/experiment.yaml
    python pipeline.py --cfg configs/smoke.yaml            # tiny CPU check of the full chain
    python pipeline.py --only test figures tables          # evaluate already-trained networks
    python pipeline.py --opt train.ir.epochs=20            # overrides go to every step
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    "check": "tests/test_ofdm_link.py",
    "data": "scripts/s1_make_dataset.py",
    "visual": "scripts/s2_visualise.py",
    "classical": "scripts/s3_classical.py",
    "train": "scripts/s4_train.py",
    "test": "scripts/s5_test.py",
    "figures": "scripts/s6_figures.py",
    "tables": "scripts/s7_latex_tables.py",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--only", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--opt", action="append", default=[])
    a = ap.parse_args()
    start = time.time()
    for step in a.only:
        cmd = [sys.executable, str(ROOT / STEPS[step])]
        if step != "check":
            cmd += ["--cfg", a.cfg] + [x for o in a.opt for x in ("--opt", o)]
        print(f"\n>>> {step}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        if subprocess.run(cmd, cwd=ROOT).returncode:
            sys.exit(f"step '{step}' failed")
        print(f">>> {step} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nfinished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
