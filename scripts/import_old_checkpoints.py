"""Re-use networks and results of the first version of this project (no retraining needed).

The first version stored   checkpoints/<ch>/p<P>/snr<S>/{sr.pt, ir.pt, meta.json}
with the layers inside one nn.Sequential called 'body'. This script renames the layers to the
SRCNN / DnCNN classes of chnet/nets.py and writes   models/<ch>_P<P>_snr<S>/{sr.pt, ir.pt, info.json}.
Old training histories (results/logs/<ch>_p<P>_snr<S>_{sr,ir}_history.csv) are copied too.
With --tables, the old MSE tables (results/metrics/mse_vs_snr_*.csv) are converted to the new
column names in results/tables/, so s6_figures.py / s7_latex_tables.py can redraw them.

    python scripts/import_old_checkpoints.py --old /path/to/old_repo             # weights + histories
    python scripts/import_old_checkpoints.py --old /path/to/old_repo --tables    # + MSE tables
    python scripts/import_old_checkpoints.py --old /path/to/old_repo --tables --no-weights
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chnet import settings  # noqa: E402
from chnet.evaluate import pilot_study, read_csv, table, write_csv  # noqa: E402

RENAME = {"LS-linear": "LS_linear", "LS-cubic": "LS_cubic", "LS-rbf": "LS_rbf", "MMSE-estimated": "MMSE_est",
          "ALMMSE-ideal": "ALMMSE", "MMSE-ideal": "MMSE_ideal"}


def new_name(col):
    if col in RENAME or col == "snr":
        return RENAME.get(col, col)
    m = re.fullmatch(r"(SR|ChannelNet)\[snr(.+)\]", col)
    if m:
        return ("SR_" if m.group(1) == "SR" else "CN_") + m.group(2)
    return None                                    # columns that are not used any more


def import_tables(cfg, old: Path):
    for f in sorted(old.glob("results/metrics/mse_vs_snr_*_p*.csv")):
        ch, P = re.fullmatch(r"mse_vs_snr_(\w+?)_p(\d+)\.csv", f.name).groups()
        rows = [{new_name(k): v for k, v in r.items() if new_name(k)} for r in read_csv(f)]
        write_csv(table(cfg, f"mse_{ch}_P{P}.csv"), rows)
        print("table", f.name, "->", f"mse_{ch}_P{P}.csv")
    for ch in cfg["channel"]["models"]:
        pilot_study(cfg, ch)


def copy_history(src: Path, dst: Path):
    if not src.exists():
        return
    with open(src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train", "val", "lr", "sec"])
        w.writeheader()
        for r in rows:
            w.writerow({"epoch": r["epoch"], "train": r["train_loss"], "val": r["val_loss"],
                        "lr": r["lr"], "sec": r["seconds"]})


def main():
    ap = settings.parser(__doc__)
    ap.add_argument("--old", required=True, help="root folder of the old project")
    ap.add_argument("--tables", action="store_true", help="also convert the old MSE tables")
    ap.add_argument("--no-weights", action="store_true", help="skip the network weights")
    args = ap.parse_args()
    cfg = settings.load(args.cfg, args.opt)
    old = Path(args.old)
    if args.tables:
        import_tables(cfg, old)
    if args.no_weights:
        for f in sorted(old.glob("results/logs/*_history.csv")):
            ch, P, t, st = re.fullmatch(r"(\w+?)_p(\d+)_snr(\w+)_(sr|ir)_history\.csv", f.name).groups()
            copy_history(f, settings.results_path(cfg, "logs", f"history_{ch}_P{P}_snr{t}_{st}.csv"))
        return
    import torch

    from chnet.nets import convert_old_ir, convert_old_sr, make_ir, make_sr
    for meta_f in sorted(old.glob("checkpoints/*/p*/snr*/meta.json")):
        d = meta_f.parent
        ch, P, t = d.parent.parent.name, int(d.parent.name[1:]), d.name[3:]
        snr = t if t == "mix" else int(t)
        meta = json.loads(meta_f.read_text())
        sr_old = torch.load(d / "sr.pt", map_location="cpu", weights_only=True)["state_dict"]
        ir_old = torch.load(d / "ir.pt", map_location="cpu", weights_only=True)["state_dict"]
        sr, ir = make_sr(cfg), make_ir(cfg)
        sr.load_state_dict(convert_old_sr(sr_old))           # strict: fails loudly on a mismatch
        ir.load_state_dict(convert_old_ir(ir_old))
        out = settings.model_dir(cfg, ch, P, snr)
        torch.save(sr.state_dict(), out / "sr.pt")
        torch.save(ir.state_dict(), out / "ir.pt")
        info = {"channel": ch, "pilots": P, "train_snr": snr, "scale": meta["scale"],
                "net_input": meta.get("sr_input_interp", "rbf"), "imported_from": str(d),
                "sr": meta.get("sr_result"), "ir": meta.get("ir_result")}
        (out / "info.json").write_text(json.dumps(info, indent=2))
        for st in ("sr", "ir"):
            copy_history(old / "results" / "logs" / f"{ch}_p{P}_snr{t}_{st}_history.csv",
                         settings.results_path(cfg, "logs", f"history_{ch}_P{P}_snr{t}_{st}.csv"))
        print("imported", d, "->", out)


if __name__ == "__main__":
    main()
