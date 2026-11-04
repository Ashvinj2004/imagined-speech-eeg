"""Before/after power-spectrum figure: shows what the fixed preprocessing removes.

    python scripts/plot_psd.py                      # Track 3, subject 1
    python scripts/plot_psd.py --dataset nieto --subject 2

A power spectrum (PSD, power spectral density) shows how much signal power sits at each
frequency. Mains hum appears as a sharp spike (60 Hz for Track 3); slow drift as a rise near 0 Hz.
Saves results/figures/psd_<dataset>_sub<N>.png (not committed to git) and prints the power at the
mains frequency before and after.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from bci.data import load
from bci.preprocess import default_config, preprocess

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"


def mean_psd(X: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per trial and channel, averaged over both; returns (freqs, power in µV²/Hz)."""
    freqs, p = signal.welch(X, fs=fs, nperseg=min(X.shape[-1], int(fs)), axis=-1)
    return freqs, p.mean(axis=(0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="track3", choices=["track3", "nieto"])
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()

    kwargs = {"subjects": [args.subject]}
    if args.dataset == "nieto":
        kwargs["sessions"] = (1,)
    raw = load(args.dataset, **kwargs)
    cfg = default_config(raw)
    clean = preprocess(raw, cfg)

    f0, p0 = mean_psd(raw.X, raw.sfreq)
    f1, p1 = mean_psd(clean.X, clean.sfreq)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(f0, p0, label="before", color="#999999")
    ax.semilogy(f1, p1, label="after", color="#1f6feb")
    for edge in (cfg.l_freq, cfg.h_freq):
        if edge:
            ax.axvline(edge, color="#1f6feb", lw=0.8, ls=":")
    if cfg.notch_hz:
        ax.axvline(cfg.notch_hz, color="#d1242f", lw=0.8, ls="--", label=f"mains {cfg.notch_hz:g} Hz")
    ax.set(xlabel="Frequency (Hz)", ylabel="Power (µV²/Hz, log scale)", xlim=(0, raw.sfreq / 2),
           title=f"{raw.name}, subject {args.subject}: average power spectrum, {raw.n_trials} trials")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"psd_{args.dataset}_sub{args.subject:02d}.png"
    fig.savefig(path, dpi=120)

    print(f"preprocessing: {cfg}")
    for f in sorted({10.0, 50.0, 60.0, 80.0}):
        if f < raw.sfreq / 2:
            b, a = (10 * np.log10(p[np.argmin(abs(fr - f))]) for fr, p in ((f0, p0), (f1, p1)))
            print(f"  {f:5.1f} Hz: {b:7.2f} dB before, {a:7.2f} dB after ({a - b:+.1f} dB)")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
