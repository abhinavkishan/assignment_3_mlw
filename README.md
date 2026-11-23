# ChannelNet: deep-learning OFDM channel estimation (COM 837, Assignment 3)

This project implements **ChannelNet** (M. Soltani, V. Pourahmadi, A. Mirzaei, H. Sheikhzadeh, *Deep Learning-Based
Channel Estimation*, arXiv:1810.05893v4). The 72 × 14 time-frequency channel of one LTE sub-frame is treated as an
image. ChannelNet reconstructs it from noisy LS estimates at a few lattice pilots using an **SRCNN**
(super-resolution) followed by a **DnCNN** (denoising). It is compared with LS + 2-D interpolation, the estimated
MMSE, the ideal ALMMSE and the ideal (genie) MMSE on the **VehA** and **SUI5** channels.

## 1. Layout

```
configs/
  experiment.yaml          main configuration (paper scale; every script reads it)
  smoke.yaml               tiny CPU run that checks the whole chain in a few minutes
  vienna.yaml              test the trained networks on Vienna-simulator channels
chnet/                     library
  settings.py              YAML loading, --opt overrides, output folders, logging
  rng.py                   all random streams derived from the master seed
  ofdm_system.py           LTE grid, QPSK, OFDM modulator/demodulator, lattice pilots, LS
  channels.py              VehA / SUI5 tapped-delay-line channels, Jakes fading (sum of sinusoids)
  data.py                  data sets, 32k/4k/4k split, noisy pilots, complex <-> 2 real images
  classical.py             LS + linear/cubic/RBF interpolation, estimated & ideal MMSE, ideal ALMMSE
  nets.py                  SRCNN, DnCNN, ChannelNet
  training.py              stage trainer (Adam, LR schedule, early stopping, fp16)
  evaluate.py              MSE-vs-SNR and MSE-vs-#pilots tables
  figures.py               every figure of the report
scripts/
  s1_make_dataset.py       Task 1  channel data + statistics
  s2_visualise.py          Task 1  pilot patterns, ground-truth images (paper Fig. 2)
  s3_classical.py          Task 2  conventional estimators
  s4_train.py              Tasks 3-5  TRAINING (SR stage, then IR stage)
  s5_test.py               Task 6  TESTING of saved networks + merged tables
  s6_figures.py            loss curves, paper Figs. 3/4/5, deep high-SNR case, generalisation
  s7_latex_tables.py       LaTeX tables for the report
  import_old_checkpoints.py  converts networks/results of the first project version
matlab/make_vienna_dataset.m   channels from the Vienna 5G Link-Level Simulator
tests/test_ofdm_link.py    checks of the signal model (time-domain OFDM link, Jakes statistics, ...)
pipeline.py                runs the steps in order
report/                    report.tex + figures/
```

Outputs:

```
data/                      cached channels (H_<model>_<source>_<N>_s<seed>.npy)
models/<ch>_P<P>_snr<S>/   sr.pt, ir.pt (state dicts) and info.json (settings, seed, scale, environment)
results/figures/           all plots; loss_curves/ holds the training curves of every network
results/tables/            CSV tables (classical_*, nets_*, mse_*, pilots_*), channel_statistics.json, *.tex
results/logs/              log of every step, training histories history_*.csv
```

## 2. Environment

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # install torch for your CUDA version from pytorch.org
```

A GPU is recommended for training (the original run used a Kaggle Tesla T4 and took about 25 min per network).
Everything else runs on a CPU.

## 3. How to run

```bash
python tests/test_ofdm_link.py                           # 1 s: signal-model checks
python pipeline.py --cfg configs/smoke.yaml              # few minutes on CPU: whole chain, tiny sizes

# full experiment
python pipeline.py --cfg configs/experiment.yaml         # check, data, visual, classical, train, test, figures, tables

# or step by step
python scripts/s1_make_dataset.py
python scripts/s2_visualise.py
python scripts/s3_classical.py
python scripts/s4_train.py                               # all networks in `runs:`
python scripts/s4_train.py --channel VehA --pilots 48 --snr 22   # one network
python scripts/s5_test.py                                # loads models/, never trains
python scripts/s6_figures.py
python scripts/s7_latex_tables.py
```

Any configuration value can be changed from the command line, e.g. `--opt train.ir.epochs=20` or
`--opt train.device=cpu`. Training is resumable: stages that already have a checkpoint are skipped
unless `--force` is given.

**Evaluate without retraining.** The networks of the first version of this project (layer names inside one
`nn.Sequential`) can be converted and then tested directly:

```bash
python scripts/import_old_checkpoints.py --old /path/to/first_version --tables
python pipeline.py --only data test figures tables
```

**Vienna simulator.** In MATLAB, run `cd matlab; make_vienna_dataset('<path>/5GLLSimulator', '../data', 5000)`,
then `python pipeline.py --cfg configs/vienna.yaml --only data classical test figures`.

## 4. Settings (as in the paper unless noted)

| item | value |
|---|---|
| grid | 72 sub-carriers × 14 OFDM symbols, 15 kHz, normal CP, N_FFT = 128 |
| channels | VehA (ITU-R M.1225), SUI5 (Erceg); 2.1 GHz, 50 km/h (f_D = 97.3 Hz), Jakes |
| data | 40 000 frames per channel, 32 000 / 4 000 / 4 000 train / val / test |
| pilots | lattice; 48 (LTE CRS diamond), plus 36 (baselines only), 24, 16, 8 |
| network input | LS pilots → Gaussian-RBF interpolation → 72×14 image (real and imaginary parts separately) |
| SRCNN | 9×9×64 → 1×1×32 → 5×5×1, ReLU |
| DnCNN | 20 layers, 64 maps, BN, residual learning |
| training | Adam 1e-3, batch 128, MSE; SR stage first, then DnCNN on the frozen SR output; LR halving on plateau, early stopping (max 150 / 40 epochs) |
| training SNRs | 12 dB (deep low-SNR), 22 dB (deep high-SNR), plus a multi-SNR network (U[0, 30] dB) |
| test SNRs | 0 … 30 dB in steps of 2 dB; pilot study at 22 dB |
| seed | 2026; all random streams derived from it (`chnet/rng.py`), cuDNN deterministic |

## 5. Reproducibility

* One master seed. Channels, noise, covariance frames, weight initialisation and batch order each use their own stream,
  identified by a tag (for example `("noise", "VehA", "test", 48, "22")`). Every estimator is therefore evaluated
  on exactly the same noisy pilots.
* Every `models/*/info.json` records the configuration file, seed, normalisation scale, training summary and
  software versions.
* `results/logs/` keeps the console output of every step.
