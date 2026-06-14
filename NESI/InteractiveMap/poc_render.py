"""
Proof-of-concept: render a single PaCMAP point's 10-min EEG segment as a
combined PNG (bipolar EEG on top, 4-region log spectrogram on bottom,
shared time axis). Borrows montage + spectrogram math from BDSP's
ilae-skill-certification-test-multi/scripts/eeg_bank_viewer.py.

Output PNGs are written next to the input .mat files.
"""
from __future__ import annotations
import os

# Quiet OpenMP duplicate-runtime crash on macOS + py3.14
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal as sig

# Morgoth's 9-color jet (verbatim from BDSP eeg_bank_viewer.py:707), 0-255.
_MORGOTH_JET_HEX = [
    (0, 0, 127), (0, 0, 255), (0, 127, 255), (0, 255, 255),
    (127, 255, 127), (255, 255, 0), (255, 127, 0),
    (255, 0, 0), (127, 0, 0),
]
MORGOTH_JET = LinearSegmentedColormap.from_list(
    'morgoth_jet',
    [(np.array(c) / 255.0) for c in _MORGOTH_JET_HEX],
)

POC_DIR = Path(__file__).resolve().parent / "poc"

# 19-channel order used by all BDSP yama .mat files.
CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
               'Fz', 'Cz', 'Pz',
               'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

# Bipolar display order — 4 chains (LL, RL, LP, RP) + 2 central, with NaN
# separator rows between chains (like a standard EEG report layout).
BIPOLAR_DISPLAY = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    None,
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    None,
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    None,
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    None,
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]

# 18-channel "clean" bipolar for spectrogram region averages (no separators).
# Order matches BDSP compute_features._fcn_bipolar.
BIPOLAR_CLEAN_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),       # LL 0..3
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),       # RL 4..7
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),       # LP 8..11
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),       # RP 12..15
    ('Fz', 'Cz'), ('Cz', 'Pz'),                                     # central 16,17
]
REGION_CHANS = {'LL': [0, 1, 2, 3], 'RL': [4, 5, 6, 7],
                'LP': [8, 9, 10, 11], 'RP': [12, 13, 14, 15]}


def _h5_char_array_to_str(arr) -> str:
    """uint16 char-array (MATLAB string) -> Python str."""
    return ''.join(chr(int(c)) for c in np.asarray(arr).flatten())


def _is_hdf5(path: Path) -> bool:
    # MATLAB v7.3 files start with a 128-byte ASCII header ("MATLAB 7.3 MAT-f…")
    # before the HDF5 payload; v5 files use "MATLAB 5.0 MAT-f…" with no HDF5.
    with open(path, 'rb') as fh:
        return b'MATLAB 7.3' in fh.read(64)


def load_mat_segment(path: Path) -> dict:
    """Read a yama 10-min EEG .mat (either v7.3 HDF5 or v5 legacy).
    Returns dict with data (n_ch, n_samples), fs, channels, metadata."""
    if _is_hdf5(path):
        with h5py.File(path, 'r') as f:
            # v7.3: /data is (n_samples, n_ch) — transpose to channels first.
            data = np.asarray(f['/data'][()]).T.astype(np.float64)
            fs = int(np.asarray(f['/Fs'][()]).item())
            channels = []
            for ref in np.asarray(f['/channels'][()]).flatten():
                channels.append(_h5_char_array_to_str(f[ref][()]))
            meta = {}
            for key in ('BDSPPatientID', 'StudyID', 'EvalDTS',
                        'Snippet_StartDTS', 'Snippet_EndDTS',
                        'SourceEEG_StartDTS'):
                if key in f:
                    meta[key] = _h5_char_array_to_str(f[key][()])
        return {'data': data, 'fs': fs, 'channels': channels, 'meta': meta}

    # v5 legacy .mat — scipy.io.loadmat returns plain numpy.
    from scipy.io import loadmat
    d = loadmat(path, squeeze_me=True)
    data = np.asarray(d['data']).astype(np.float64)
    if data.shape[0] != 19 and data.shape[1] == 19:
        data = data.T                          # normalize to (n_ch, n_samples)
    fs = int(np.asarray(d['Fs']).item())
    channels = [str(c) for c in np.asarray(d['channels']).flatten()]
    meta = {k: str(v) for k, v in d.items()
            if not k.startswith('__') and k not in ('data', 'Fs', 'channels')}
    return {'data': data, 'fs': fs, 'channels': channels, 'meta': meta}


def apply_filters(data: np.ndarray, fs: float,
                   lo: float, hi: float,
                   notch: float = 60.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth bandpass + IIR notch (BDSP convention)."""
    hi = min(hi, fs / 2 - 1)
    sos_bp = sig.butter(N=order, Wn=[lo, hi], btype='band',
                          fs=fs, output='sos')
    b, a = sig.iirnotch(w0=notch, Q=30.0, fs=fs)
    sos_nt = sig.tf2sos(b, a)
    out = data.copy()
    for i in range(out.shape[0]):
        out[i] = sig.sosfiltfilt(sos_bp, out[i])
        out[i] = sig.sosfiltfilt(sos_nt, out[i])
    return out


def build_bipolar_display(data: np.ndarray, channels: list[str]):
    """Returns (rows, names) for the display montage with NaN separator rows."""
    ch_to_idx = {nm: i for i, nm in enumerate(channels) if i < data.shape[0]}
    n = data.shape[1]
    blank = np.full(n, np.nan)
    rows, names = [], []
    for entry in BIPOLAR_DISPLAY:
        if entry is None:
            rows.append(blank.copy()); names.append('')
            continue
        a, b = entry
        if a in ch_to_idx and b in ch_to_idx:
            rows.append(data[ch_to_idx[a]] - data[ch_to_idx[b]])
            names.append(f"{a}-{b}")
        else:
            rows.append(blank.copy()); names.append(f"{a}-{b} (missing)")
    return np.asarray(rows), names


def build_bipolar_clean(data: np.ndarray, channels: list[str]) -> np.ndarray:
    """(18, n_samples) bipolar for the regional spectrogram averages."""
    ch_to_idx = {nm: i for i, nm in enumerate(channels) if i < data.shape[0]}
    n = data.shape[1]
    bp = np.zeros((len(BIPOLAR_CLEAN_PAIRS), n), dtype=data.dtype)
    for k, (a, b) in enumerate(BIPOLAR_CLEAN_PAIRS):
        if a in ch_to_idx and b in ch_to_idx:
            bp[k] = data[ch_to_idx[a]] - data[ch_to_idx[b]]
    return bp


def compute_mt_spectrogram_one(x: np.ndarray, fs: float,
                                window_s: float = 4.0, step_s: float = 1.0,
                                NW: float = 3.0, K: int = 5,
                                fpass: tuple[float, float] = (0.0, 55.0)):
    """Multitaper spectrogram, lifted from BDSP precompute_spectrograms.py
    (matches morgoth's Chronux `mtspecgram_jj`).

    Per window: linear detrend → multiply by K DPSS tapers
    (time-half-bandwidth NW) → |FFT|² / fs (density scaling) → mean across
    tapers. ~K× variance reduction vs single-taper STFT. NFFT = next power
    of 2 ≥ window samples (Chronux convention).
    """
    Nwin = int(window_s * fs)
    Nstep = int(step_s * fs)
    if Nwin > len(x):
        return None, None, None
    NFFT = 1 << int(np.ceil(np.log2(Nwin)))
    tapers = sig.windows.dpss(Nwin, NW=NW, Kmax=K)            # (K, Nwin)
    starts = np.arange(0, len(x) - Nwin + 1, Nstep)
    if len(starts) == 0:
        return None, None, None
    freqs_all = np.fft.rfftfreq(NFFT, d=1.0 / fs)
    fmask = (freqs_all >= fpass[0]) & (freqs_all <= fpass[1])
    freqs = freqs_all[fmask]
    S = np.empty((len(freqs), len(starts)), dtype=np.float64)
    for i, ws in enumerate(starts):
        seg = sig.detrend(x[ws:ws + Nwin], type='linear')
        F = np.fft.rfft(seg[None, :] * tapers, n=NFFT, axis=-1)
        psd = (np.abs(F) ** 2) / fs
        S[:, i] = psd[:, fmask].mean(axis=0)
    times = (starts + Nwin // 2) / fs
    return S, freqs, times


def compute_regional_spectrograms(bp_clean: np.ndarray, fs: float,
                                    fmin: float = 0.5, fmax: float = 25.0):
    """Multitaper PSD per channel, mean across the 4 bipolar channels per
    region (LL/RL/LP/RP). Compute range = 0-55 Hz (BDSP default); slice to
    fmin..fmax for display."""
    out = {}
    freqs = stimes = None
    for region, idxs in REGION_CHANS.items():
        acc = None
        for ch in idxs:
            S, f, t = compute_mt_spectrogram_one(bp_clean[ch], fs)
            if freqs is None:
                freqs, stimes = f, t
            acc = S if acc is None else acc + S
        out[region] = acc / len(idxs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return {k: v[mask] for k, v in out.items()}, freqs[mask], stimes


EEG_WINDOW_S = 15.0       # short clip shown above the spec
EEG_GAIN_UV_DIV = 100.0   # BDSP default: fixed µV per row-spacing unit


def render_segment_png(mat_path: Path, out_path: Path, *, title: str):
    """Produce a combined PNG:
       top    = EEG_WINDOW_S of bipolar EEG, centered on the segment midpoint
       bottom = full 10-min, 4-region log spectrogram
       Time axes do NOT align (intentional — EEG is zoomed in)."""
    seg = load_mat_segment(mat_path)
    data, fs, channels = seg['data'], seg['fs'], seg['channels']
    duration_s = data.shape[1] / fs
    print(f"  loaded: {data.shape[0]}ch x {data.shape[1]} samples "
          f"@ {fs} Hz ({duration_s/60:.1f} min)")

    # Two filter passes: wide band (0.5-70 Hz) for spectrogram input so the
    # 0.5-25 Hz display range is well below cutoff; narrow band (0.5-30 Hz)
    # for the EEG display panel (standard clinical filter).
    data_spec = apply_filters(data, fs, lo=0.5, hi=70.0, order=3)
    bp_clean = build_bipolar_clean(data_spec, channels)
    specs, freqs, stimes = compute_regional_spectrograms(bp_clean, fs)

    data_disp = apply_filters(data, fs, lo=0.5, hi=30.0, order=3)
    rows, names = build_bipolar_display(data_disp, channels)

    # ── 15-sec EEG slice from segment midpoint ──
    win_samps = int(EEG_WINDOW_S * fs)
    mid = data.shape[1] // 2
    lo = max(0, mid - win_samps // 2)
    hi = min(data.shape[1], lo + win_samps)
    rows_clip = rows[:, lo:hi]
    t_clip = (np.arange(rows_clip.shape[1]) + lo) / fs
    eeg_start_s = lo / fs

    # Fixed gain (BDSP convention): each row spacing = EEG_GAIN_UV_DIV µV.
    # Values are clipped to ±3*gain to keep extreme artifact from overlapping
    # neighbouring channels.
    z = 1.0 / EEG_GAIN_UV_DIV
    clip = 3.0 * EEG_GAIN_UV_DIV

    fig = plt.figure(figsize=(16, 9), dpi=130)
    # 1 row × 2 cols: spectrograms in a narrow left strip (~25% of width),
    # EEG fills the rest. Matches the BDSP/morgoth review layout.
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.0, 3.0], wspace=0.06,
        left=0.05, right=0.985, top=0.93, bottom=0.07,
    )

    # ── LEFT: 4 stacked region spectrograms (LL, RL, LP, RP) ──
    inner = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=gs[0], hspace=0.05,
    )
    region_order = ['LL', 'RL', 'LP', 'RP']     # BDSP convention, top→bottom
    # Match BDSP: 10*log10(S + float32_eps), fixed dB range [-10, 25],
    # morgoth's 9-color jet. No autoscale.
    eps = float(np.finfo(np.float32).eps)
    log_specs = {k: 10.0 * np.log10(v + eps) for k, v in specs.items()}
    vmin, vmax = -10.0, 25.0
    for i, region in enumerate(region_order):
        ax = fig.add_subplot(inner[i])
        ax.imshow(log_specs[region], origin='lower', aspect='auto',
                   extent=[stimes[0] / 60.0, stimes[-1] / 60.0,
                           freqs[0], freqs[-1]],
                   cmap=MORGOTH_JET, vmin=vmin, vmax=vmax,
                   interpolation='bilinear')
        ax.set_ylabel(f"{region}\nHz", rotation=0, ha='right', va='center',
                       fontsize=9)
        ax.set_yticks([5, 15, 25])
        ax.tick_params(labelsize=7)
        if i < len(region_order) - 1:
            ax.set_xticks([])
        else:
            ax.set_xticks([0, 2, 4, 6, 8, 10])
            ax.tick_params(labelsize=7)
            ax.set_xlabel("time (min)", fontsize=8)

    # ── RIGHT: 15-sec EEG window (fixed gain = EEG_GAIN_UV_DIV µV/div) ──
    ax_eeg = fig.add_subplot(gs[1])
    n_rows = rows_clip.shape[0]
    row_spacing = 1.0
    for i in range(n_rows):
        y = (n_rows - 1 - i) * row_spacing
        ch = np.nan_to_num(rows_clip[i], nan=0.0)
        trace = z * np.clip(ch, -clip, clip)
        ax_eeg.plot(t_clip, y + trace, color='black', lw=0.4, antialiased=True)

    # BDSP-style L-shaped scale bar: 1 s × EEG_GAIN_UV_DIV µV, below the
    # bottom trace at the right edge. One row-spacing unit on y equals
    # EEG_GAIN_UV_DIV µV (because z = 1/gain).
    sb_x = eeg_start_s + EEG_WINDOW_S - 1.6
    sb_y = -1.15
    sb_kw = dict(color='black', lw=1.8, solid_capstyle='butt')
    ax_eeg.plot([sb_x, sb_x + 1.0], [sb_y, sb_y], **sb_kw)         # horizontal
    ax_eeg.plot([sb_x, sb_x], [sb_y, sb_y + 1.0], **sb_kw)         # vertical
    ax_eeg.text(sb_x + 0.5, sb_y - 0.08, "1 s",
                  ha='center', va='top', fontsize=8)
    ax_eeg.text(sb_x - 0.08, sb_y + 0.5, f"{int(EEG_GAIN_UV_DIV)} µV",
                  ha='right', va='center', fontsize=8)

    ax_eeg.set_ylim(-1.4, (n_rows - 1) * row_spacing + 0.6)
    ax_eeg.set_yticks([(n_rows - 1 - i) * row_spacing for i in range(n_rows)])
    ax_eeg.set_yticklabels(names, fontsize=8)
    ax_eeg.set_xlim(eeg_start_s, eeg_start_s + EEG_WINDOW_S)
    eeg_ticks = np.arange(eeg_start_s, eeg_start_s + EEG_WINDOW_S + 0.01, 1.0)
    ax_eeg.set_xticks(eeg_ticks)
    ax_eeg.set_xticklabels([f"{int(t)}" for t in eeg_ticks], fontsize=8)
    ax_eeg.set_xlabel(
        f"time in segment (s)  —  {EEG_WINDOW_S:.0f}-s window centered "
        f"at {duration_s/2:.0f} s   |   gain = {EEG_GAIN_UV_DIV:.0f} µV / division",
        fontsize=9,
    )
    ax_eeg.set_title(title, fontsize=11, fontweight='bold', loc='left')
    for spine in ('top', 'right'):
        ax_eeg.spines[spine].set_visible(False)

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out_path}")


def main():
    samples = [
        (POC_DIR / "cams_sample.mat",
         POC_DIR / "cams_sample.png",
         "CAMS — sub-S0001111802138_SID-EGGS048_cam_s_10min"),
        (POC_DIR / "rass_sample.mat",
         POC_DIR / "rass_sample.png",
         "RASS — sub-S0001111449034_ses-1_eeg_10min_seg8"),
    ]
    for mat, png, title in samples:
        if not mat.exists():
            print(f"missing: {mat}")
            continue
        print(f"\nrendering: {mat.name}")
        render_segment_png(mat, png, title=title)


if __name__ == '__main__':
    main()
