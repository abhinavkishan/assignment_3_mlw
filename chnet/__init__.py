"""chnet - ChannelNet OFDM channel estimation (COM 837, Assignment 3).

Module map
----------
settings     YAML configuration, command-line overrides, output folders
rng          reproducible random streams derived from one master seed
ofdm_system  LTE grid, QPSK, OFDM modulator/demodulator, lattice pilots, LS estimation
channels     VehA / SUI5 tapped-delay-line channels with Jakes fading
data         channel data sets, splits, noisy pilot observations, complex <-> image conversion
classical    LS + 2-D interpolation, ideal / estimated LMMSE, ideal ALMMSE
nets         SRCNN, DnCNN and the ChannelNet cascade          (needs torch)
training     one-stage trainer with early stopping              (needs torch)
evaluate     MSE-vs-SNR and MSE-vs-#pilots tables
figures      all plots of the report
"""
__version__ = "2.0"
