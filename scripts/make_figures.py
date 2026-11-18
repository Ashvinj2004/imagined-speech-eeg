"""Build the presentation figures from results/LOG.md, so every bar traces to a logged run.

    python scripts/make_figures.py

Writes results/figures/fig1_protocols.png ... fig4_summary.png (gitignored; regenerate any time).
Each figure draws the chance threshold for its own test size, never 1/k alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tables import read_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"

BLUE, GREY, RED, DARK = "#1f6feb", "#9aa4b1", "#d1242f", "#24292f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "figure.dpi": 130})


def pick(df, **kw):
    """Latest row matching all the given column values, or None."""
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= df[k] == v
    hit = df[m].sort_values("date")
    return hit.iloc[-1] if len(hit) else None


def bars(ax, labels, accs, thresholds, title, ylabel="Accuracy"):
    x = np.arange(len(labels))
    above = [a > t for a, t in zip(accs, thresholds)]
    ax.bar(x, accs, color=[BLUE if ok else GREY for ok in above], width=0.62, zorder=2)
    for i, (a, t) in enumerate(zip(accs, thresholds)):
        ax.hlines(t, i - 0.36, i + 0.36, color=RED, lw=2, zorder=3)
        # lift the value label clear of the threshold line when the bar top sits near it
        pad = 0.028 if abs(a - t) < 0.03 else 0.012
        ax.text(i, a + pad, f"{a:.3f}", ha="center", fontsize=10, zorder=4)
    ax.set_xticks(x, labels)
    ax.set(ylabel=ylabel, title=title, ylim=(0, max(max(accs), max(thresholds)) * 1.35))
    ax.grid(axis="y", alpha=0.25, zorder=0)


def fig1_protocols(df):
    """The central finding: a within-person effect that vanishes across people and datasets."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})

    t_w = pick(df, dataset="bci2020_track3", model="riemann", protocol="predefined", variant="")
    t_l = pick(df, dataset="bci2020_track3", model="riemann", protocol="loso", variant="")
    bars(a1, ["Same person\n(official test)", "New person\n(LOSO)"],
         [t_w["acc"], t_l["acc"]], [t_w["threshold"], t_l["threshold"]],
         "Track 3: 5 imagined phrases, 15 people")
    a1.annotate("", xy=(0.92, t_l["acc"] + 0.035), xytext=(0.08, t_w["acc"] + 0.035),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6,
                                connectionstyle="arc3,rad=-0.3"))
    a1.text(0.5, t_w["acc"] + 0.105, f"-{(t_w['acc'] - t_l['acc']) * 100:.0f} points",
            ha="center", fontsize=11.5, fontweight="bold", color=DARK)

    rows = [pick(df, dataset="nieto2022-inner", model="riemann", protocol=p, variant="")
            for p in ("within_subject", "cross_session", "loso")]
    bars(a2, ["Same person", "Another day", "New person"], [r["acc"] for r in rows],
         [r["threshold"] for r in rows], "Nieto: 4 imagined words, 10 people", ylabel="")
    a2.text(1, 0.055, "nothing above chance", ha="center", fontsize=11, color=RED, fontweight="bold")

    fig.suptitle("A real effect within a person; nothing that transfers", fontsize=13, y=1.0)
    fig.text(0.5, -0.03, "Red line = accuracy that luck alone reaches 5% of the time, for that exact test size. "
             "Grey bars do not beat it.", ha="center", fontsize=9.5, color=DARK)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_protocols.png", bbox_inches="tight")
    plt.close(fig)


def fig2_bands(df):
    """The headline: accuracy climbs into the muscle-exposed band."""
    bands = [("band1-8", "1-8 Hz"), ("band8-13", "8-13 Hz"), ("band13-30", "13-30 Hz"), ("band30-40", "30-40 Hz")]
    rows = [pick(df, dataset="bci2020_track3", protocol="predefined_val", variant=v) for v, _ in bands]
    full = pick(df, dataset="bci2020_track3", model="riemann", protocol="predefined_val", variant="", band="0.5-30.0 Hz")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(rows))
    accs = [r["acc"] for r in rows]
    ax.bar(x, accs, color=[BLUE] * 3 + [RED], width=0.6, zorder=2)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.008, f"{a:.3f}", ha="center", fontsize=10.5, zorder=4)
    ax.axhline(full["acc"], color=DARK, ls="--", lw=1.4, zorder=3)
    ax.text(-0.45, full["acc"] + 0.012, f"all 0.5-30 Hz together: {full['acc']:.3f}",
            ha="left", fontsize=9.5, color=DARK)
    ax.hlines(rows[0]["threshold"], -0.5, 3.5, color=RED, lw=1.6, ls=":", zorder=3)
    ax.text(-0.45, rows[0]["threshold"] - 0.028, "chance", fontsize=9.5, color=RED)
    ax.set_xticks(x, [lbl for _, lbl in bands])
    ax.set(ylabel="Accuracy", xlabel="Frequency band used (all else identical)", ylim=(0, 0.60),
           title="Track 3: the signal is strongest where muscle activity lives")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.annotate("highest band alone beats\nthe full band", xy=(3, accs[3] + 0.025), xytext=(1.6, 0.545),
                fontsize=10.5, color=RED, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5,
                                connectionstyle="arc3,rad=-0.15"))
    fig.text(0.5, -0.04, "30-40 Hz is where jaw and face muscle activity appears. Track 3 records no muscle "
             "channel, so we cannot rule it out.", ha="center", fontsize=9.5, color=DARK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bands.png", bbox_inches="tight")
    plt.close(fig)


def fig3_per_subject():
    """Every person, so the judges see the spread behind the average."""
    import pandas as pd

    runs = sorted((ROOT / "results" / "runs").glob("*track3_riemann_predefined"))
    loso = sorted((ROOT / "results" / "runs").glob("*track3_riemann_loso"))
    if not runs or not loso:
        print("skipping fig3: need a Track 3 predefined and loso run folder", file=sys.stderr)
        return
    w = pd.read_csv(runs[-1] / "folds.csv")
    l = pd.read_csv(loso[-1] / "folds.csv")

    fig, ax = plt.subplots(figsize=(10, 4.3))
    x = np.arange(len(w))
    ax.bar(x - 0.2, w["acc"], width=0.38, color=BLUE, label="same person (official test)", zorder=2)
    ax.bar(x + 0.2, l.sort_values("subject")["acc"].to_numpy(), width=0.38, color=GREY,
           label="new person (LOSO)", zorder=2)
    ax.hlines(w["chance_threshold"].iloc[0], -0.6, len(w) - 0.4, color=RED, lw=1.8,
              label=f"chance for 50 trials ({w['chance_threshold'].iloc[0]:.0%})", zorder=3)
    n_w = int((w["acc"] > w["chance_threshold"]).sum())
    n_l = int((l["acc"] > l["chance_threshold"]).sum())
    ax.set_xticks(x, [str(s) for s in w["subject"]])
    ax.set(xlabel="Participant", ylabel="Accuracy", ylim=(0, 0.75),
           title=f"Track 3, every participant: {n_w} of {len(w)} beat chance on their own data, "
                 f"{n_l} of {len(l)} as a new person")
    ax.legend(fontsize=9.5, loc="upper right", ncol=1)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_per_subject.png", bbox_inches="tight")
    plt.close(fig)


def fig4_summary(df):
    """What we tried to improve it with, and what each did."""
    base = pick(df, dataset="bci2020_track3", model="riemann", protocol="predefined_val", variant="", band="0.5-30.0 Hz")
    ea = pick(df, dataset="bci2020_track3", model="riemann_ea", protocol="predefined_val", variant="ea")
    cr = pick(df, dataset="bci2020_track3", model="riemann", protocol="predefined_val", variant="crops")
    items = [("Baseline", base), ("+ Euclidean\nAlignment", ea), ("+ Cropped\ntraining", cr)]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(items))
    accs = [r["acc"] for _, r in items]
    ax.bar(x, accs, color=[DARK, GREY, GREY], width=0.55, zorder=2)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.008, f"{a:.3f}", ha="center", fontsize=10.5)
        if i:
            d = a - accs[0]
            ax.text(i, a / 2, f"{d:+.3f}", ha="center", fontsize=11, color="white", fontweight="bold")
    ax.axhline(accs[0], color=DARK, ls="--", lw=1.2, zorder=1)
    ax.set_xticks(x, [n for n, _ in items])
    ax.set(ylabel="Accuracy (same person)", ylim=(0, 0.52),
           title="Neither standard improvement helped")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.text(0.5, -0.04, "Alignment is designed to help across people; it left cross-person accuracy at chance.",
             ha="center", fontsize=9.5, color=DARK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    df = read_log()
    if df.empty:
        print("results/LOG.md has no rows", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_protocols(df)
    fig2_bands(df)
    fig3_per_subject()
    fig4_summary(df)
    for p in sorted(OUT.glob("fig*.png")):
        print(f"  {p.name}  {p.stat().st_size // 1024} KB")
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
