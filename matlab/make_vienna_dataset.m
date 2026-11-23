function make_vienna_dataset(simDir, outDir, nFrames, seed, models, tag)
%MAKE_VIENNA_DATASET  72 x 14 ground-truth channel grids from the Vienna 5G Link-Level Simulator.
%
%   make_vienna_dataset(simDir, outDir, nFrames, seed)
%   make_vienna_dataset(simDir, outDir, 10000, 11, {'SUI5'}, '_part1')   % one chunk of SUI5
%
%   simDir  : folder of the 5G Link-Level Simulator (the one that contains +Channel/FastFading.m)
%   outDir  : output folder (default ../data)  -> vienna_<model><tag>.mat with variable H [N x 72 x 14]
%   nFrames : independent sub-frames per model (default 40000, as in the paper)
%   seed    : random seed (default 2026)
%   models  : subset of {'VehA','SUI5'} (default both)
%   tag     : text appended to the file name, to build one model in several parallel chunks
%             (use a different seed per chunk and list all files in configs/*.yaml -> vienna_files)
%
%   Set-up of Soltani et al. (2019): SISO, LTE grid of 72 sub-carriers x 14 OFDM symbols, 15 kHz,
%   normal CP, 2.1 GHz carrier, 50 km/h (f_D = 97.3 Hz), Vehicular-A and SUI5 channels.
%
%   The simulator's Channel.FastFading object (WSSUS tapped-delay line, Jakes Doppler spectrum)
%   produces the channel; GetTransferFunction returns the perfect frequency response at given time
%   samples and FFT bins. We read it at the centre of the useful part of every OFDM symbol, on the
%   72 active bins (logical sub-carriers -36..-1, 1..36). A 15.36 MHz sampling rate (N_FFT = 1024)
%   keeps the VehA delays accurate to 65 ns. SUI5 is not built into the simulator, so it is passed
%   as a user-defined power-delay profile on the sampling grid (0/4/10 us, 0/-5/-10 dB).
%
%   The Python side (chnet/data.py) normalises H to unit average power when it loads the file.

if nargin < 1 || isempty(simDir),  simDir  = fullfile('..', '..', 'supporting', '5GLLSimulator'); end
if nargin < 2 || isempty(outDir),  outDir  = fullfile('..', 'data'); end
if nargin < 3 || isempty(nFrames), nFrames = 40000; end
if nargin < 4 || isempty(seed),    seed    = 2026; end
if nargin < 5 || isempty(models),  models  = {'VehA', 'SUI5'}; end
if nargin < 6,                     tag     = ''; end
if ischar(models), models = {models}; end

addpath(simDir);
if ~exist(outDir, 'dir'), mkdir(outDir); end
rng(seed, 'twister');

sys = systemParameters();
for i = 1:numel(models)
    name = models{i};
    pdp  = powerDelayProfile(name, sys.fs);
    ch   = buildFading(sys, pdp);
    fprintf('%s: %d frames, f_D = %.1f Hz, rms delay %.2f us\n', ...
        name, nFrames, sys.fD, ch.GetRmsDelaySpread * 1e6);

    H = complex(zeros(nFrames, sys.K, sys.L, 'single'));
    t0 = tic;
    for n = 1:nFrames
        ch.NewRealization(n);                                         % independent sub-frame
        H(n, :, :) = single(ch.GetTransferFunction(sys.tSample, sys.nfft, sys.bins));   % 72 x 14
        if mod(n, 5000) == 0
            fprintf('   %6d / %d   %.0f s\n', n, nFrames, toc(t0));
        end
    end

    meta = sys;
    meta.model = name;
    meta.seed = seed;
    meta.rmsDelaySpread = ch.GetRmsDelaySpread;
    meta.meanPower = mean(abs(H(:)).^2);
    f = fullfile(outDir, ['vienna_' name tag '.mat']);
    save(f, 'H', 'meta', '-v7');
    fprintf('   saved %s (mean |H|^2 = %.3f)\n', f, meta.meanPower);
end
end

% -------------------------------------------------------------------------------------------
function sys = systemParameters()
% LTE 1.4 MHz numerology simulated at 15.36 MHz
sys.df    = 15e3;                                         % sub-carrier spacing
sys.nfft  = 1024;
sys.fs    = sys.df * sys.nfft;                            % 15.36 MHz
sys.cp    = [80 72 72 72 72 72 72 80 72 72 72 72 72 72];  % normal CP (samples)
sys.K     = 72;
sys.L     = 14;
sys.fc    = 2.1e9;
sys.v     = 50 / 3.6;                                     % m/s
sys.fD    = sys.v * sys.fc / 299792458;                   % ~97.3 Hz
sys.nSamp = sum(sys.cp) + sys.L * sys.nfft;               % 15360 samples = 1 ms
sys.nPaths = 50;                                          % sinusoids per tap of the fading process
sys.subSampling = 32;                                     % time sub-sampling of the fading process
start = [0, cumsum(sys.cp(1:end-1) + sys.nfft)];
sys.tSample = start + sys.cp + sys.nfft / 2 + 1;          % 1-based centre sample of each symbol
k = [-sys.K/2:-1, 1:sys.K/2];
sys.bins = mod(k, sys.nfft) + 1;                          % 1-based FFT bins of the active carriers
end

function pdp = powerDelayProfile(name, fs)
switch name
    case 'VehA'
        pdp = 'VehicularA';                               % built into the simulator
    case 'SUI5'
        d = [0 4 10] * 1e-6;  p = 10 .^ ([0 -5 -10] / 10);
        pdp = zeros(1, round(d(end) * fs) + 1);
        pdp(round(d * fs) + 1) = p;                       % taps on the sampling grid
    otherwise
        error('unknown channel model %s', name);
end
end

function ch = buildFading(sys, pdp)
% Argument order of Channel.FastFading in the Vienna 5G LLS
noSpatial.spatialChannelModel = false;
ch = Channel.FastFading( ...
    sys.fs, pdp, sys.nSamp, sys.fD, 'Jakes', sys.nPaths, ...
    false, ...                  % no correlation between consecutive realisations
    'none', 0, 0, ...           % no antenna correlation (SISO)
    1, 1, ...                   % 1 transmit, 1 receive antenna
    false, ...                  % no warning about delay rounding
    0, 1, ...                   % K-factor 0 -> Rayleigh
    noSpatial, [1 1], [1 1], ...
    sys.fc, sys.subSampling);
end
