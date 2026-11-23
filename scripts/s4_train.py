"""Tasks 3-5 - TRAIN ChannelNet (training only; testing is s5_test.py).

For every (channel, #pilots, training SNR) listed under `runs:` in the configuration:
  1. noisy LS pilots at the training SNR -> Gaussian-RBF interpolation -> low-resolution image X
  2. SR stage : SRCNN trained on  X -> H                       (loss C1, eq. (7))
  3. IR stage : SRCNN frozen, DnCNN trained on  f_SR(X) -> H    (loss C2, eq. (8))
Real and imaginary parts are separate images; all images are divided by one global scale.

    python scripts/s4_train.py                                       # all runs
    python scripts/s4_train.py --channel VehA --pilots 48 --snr 22   # one network
    python scripts/s4_train.py --stage ir --force                    # retrain only the DnCNNs

Already trained stages are skipped (so an interrupted session can be restarted) unless --force.
Writes models/<ch>_P<P>_snr<S>/{sr.pt, ir.pt, info.json}
       results/logs/history_<ch>_P<P>_snr<S>_{sr,ir}.csv
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from chnet import rng, settings  # noqa: E402
from chnet.classical import apply, interp_operator  # noqa: E402
from chnet.data import pilots_ls, scale_of, split, to_images  # noqa: E402
from chnet.nets import make_ir, make_sr, n_params  # noqa: E402
from chnet.ofdm_system import LTEGrid, pattern  # noqa: E402
from chnet.training import StageTrainer, device_of, run, save_history  # noqa: E402


def images(cfg, H, ch, part, P, snr, W, scale):
    """(input, target) image pairs for one split."""
    hp = pilots_ls(cfg, H, ch, part, P, snr)
    return to_images(apply(W, hp, H.shape[1:]), scale), to_images(H, scale)


def history_file(cfg, ch, P, snr, stage):
    return settings.results_path(cfg, "logs", f"history_{ch}_P{P}_snr{settings.tag(snr)}_{stage}.csv")


def train_network(cfg, ch, P, snr, stage, force, dev, log):
    d = settings.model_dir(cfg, ch, P, snr)
    f_sr, f_ir, f_info = d / "sr.pt", d / "ir.pt", d / "info.json"
    want_sr = stage in ("all", "sr") and (force or not f_sr.exists())
    want_ir = stage in ("all", "ir") and (force or want_sr or not f_ir.exists())
    if not (want_sr or want_ir):
        log(f"{d.name}: already trained (use --force to retrain)")
        return
    log(f"========== {d.name} ==========")
    seed = rng.int_seed(cfg, "weights", ch, P, settings.tag(snr))
    rng.seed_torch(seed, bool(cfg["train"]["deterministic"]))

    W = interp_operator(pattern(cfg, LTEGrid.from_cfg(cfg), P), cfg["classical"]["net_input"])
    H_tr, H_va = split(cfg, ch, "train", log), split(cfg, ch, "val", log)
    scale = scale_of(H_tr)
    x_tr, y_tr = images(cfg, H_tr, ch, "train", P, snr, W, scale)
    x_va, y_va = images(cfg, H_va, ch, "val", P, snr, W, scale)
    log(f"images: train {x_tr.shape}, val {x_va.shape}, scale {scale:.4f}")

    info = json.loads(f_info.read_text()) if f_info.exists() and not force else {}
    info.update(channel=ch, pilots=P, train_snr=snr, seed=seed, scale=scale,
                net_input=cfg["classical"]["net_input"], nets=cfg["nets"], train=cfg["train"],
                data=cfg["data"], config=cfg["_file"],
                env={"python": platform.python_version(), "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})

    # ---- stage 1: super-resolution
    sr = make_sr(cfg)
    if want_sr:
        hist, res = StageTrainer(sr, cfg["train"]["sr"], cfg, dev, seed, f"SR {d.name}", log).fit(
            x_tr, y_tr, x_va, y_va)
        save_history(hist, history_file(cfg, ch, P, snr, "sr"))
        torch.save(sr.state_dict(), f_sr)
        info["sr"] = {**res, "params": n_params(sr)}
    else:
        sr.load_state_dict(torch.load(f_sr, map_location=dev, weights_only=True))

    # ---- stage 2: restoration on top of the frozen SRCNN
    if want_ir:
        bs = int(cfg["train"]["infer_batch"])
        s_tr, s_va = run(sr, x_tr, dev, bs), run(sr, x_va, dev, bs)
        ir = make_ir(cfg)
        hist, res = StageTrainer(ir, cfg["train"]["ir"], cfg, dev, seed + 1, f"IR {d.name}", log).fit(
            s_tr, y_tr, s_va, y_va)
        save_history(hist, history_file(cfg, ch, P, snr, "ir"))
        torch.save(ir.state_dict(), f_ir)
        info["ir"] = {**res, "params": n_params(ir)}
    f_info.write_text(json.dumps(info, indent=2, default=str))
    log(f"saved {d}")


def main():
    ap = settings.parser(__doc__)
    ap.add_argument("--channel")
    ap.add_argument("--pilots", type=int)
    ap.add_argument("--snr", help="12, 22, mix, ...")
    ap.add_argument("--stage", default="all", choices=["all", "sr", "ir"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = settings.load(args.cfg, args.opt)
    log = settings.Log(cfg, "s4_train")
    dev = device_of(cfg)
    todo = settings.runs(cfg, args.channel, args.pilots, args.snr)
    log(f"device {dev}; {len(todo)} network(s): {todo}")
    t0 = time.time()
    for ch, P, snr in todo:
        train_network(cfg, ch, P, snr, args.stage, args.force, dev, log)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
