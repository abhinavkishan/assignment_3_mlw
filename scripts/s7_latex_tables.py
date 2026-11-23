"""Write the report tables as LaTeX (results/tables/tab_snr.tex, tab_pilots.tex) from the CSVs.
Paste them into report/report.tex (or \\input them) after a new run.

    python scripts/s7_latex_tables.py --cfg configs/experiment.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from chnet import settings  # noqa: E402
from chnet.evaluate import read_csv, table  # noqa: E402

SNR_COLS = [("LS_linear", "LS-lin"), ("MMSE_est", "Est.\\ MMSE"), ("ALMMSE", "Ideal ALMMSE"),
            ("MMSE_ideal", "Ideal MMSE"), ("SR_22", "SR-22"), ("CN_12", "CN-12"), ("CN_22", "CN-22"),
            ("CN_mix", "CN-mix")]
PIL_COLS = [c for c in SNR_COLS if c[0] not in ("CN_12", "CN_mix")]
GENIE = {"MMSE_ideal", "ALMMSE"}


def sci(v, bold=False):
    if v is None or np.isnan(v):
        return "--"
    m, e = f"{v:.2e}".split("e")
    num = f"{m}\\cdot10^{{{int(e)}}}"
    return f"$\\mathbf{{{num}}}$" if bold else f"${num}$"


def body(rows, first_key, first_fmt, cols):
    out = []
    for r in rows:
        vals = {k: r.get(k, np.nan) for k, _ in cols}
        fair = {k: v for k, v in vals.items() if k not in GENIE and not np.isnan(v)}
        best = min(fair, key=fair.get) if fair else None
        cells = [sci(v, k == best) for k, v in vals.items()]
        out.append(f"{first_fmt(r[first_key])} & " + " & ".join(cells) + " \\\\")
    return out


def main():
    args = settings.parser(__doc__).parse_args()
    cfg = settings.load(args.cfg, args.opt)
    P = int(cfg["pilots"]["main"])
    keep = [0, 6, 12, 20, 22, 30]
    lines = ["\\begin{tabular}{ll" + "c" * len(SNR_COLS) + "}", "\\toprule",
             "channel & SNR & " + " & ".join(h for _, h in SNR_COLS) + " \\\\", "\\midrule"]
    for ch in ("VehA", "SUI5"):
        f = table(cfg, f"mse_{ch}_P{P}.csv")
        if f.exists():
            rows = [r for r in read_csv(f) if int(r["snr"]) in keep]
            lines += [f"{ch} & " + ln for ln in body(rows, "snr", lambda s: f"{int(s)}", SNR_COLS)]
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    settings.results_path(cfg, "tables", "tab_snr.tex").write_text("\n".join(lines))

    snr = int(cfg["snr"]["pilot_study_db"])
    lines = ["\\begin{tabular}{ll" + "c" * len(PIL_COLS) + "}", "\\toprule",
             "channel & pilots & " + " & ".join(h for _, h in PIL_COLS) + " \\\\", "\\midrule"]
    for ch in ("VehA", "SUI5"):
        f = table(cfg, f"pilots_{ch}_snr{snr}.csv")
        if f.exists():
            lines += [f"{ch} & " + ln for ln in body(read_csv(f), "pilots", lambda p: f"{int(p)}", PIL_COLS)]
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    settings.results_path(cfg, "tables", "tab_pilots.tex").write_text("\n".join(lines))
    print("written tab_snr.tex and tab_pilots.tex in", settings.results_path(cfg, "tables"))


if __name__ == "__main__":
    main()
