"""Standard figures and the results log, so every result is reported the same way.

- Confusion matrix: rows = true word, columns = predicted word; shows which words get mixed up.
- Per-subject chart: each person's accuracy against the chance threshold for their test size,
  because a pooled average can hide large differences between people.
- `append_log_row`: one row per experiment in results/LOG.md (RULES.md rule 6).
"""

from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from bci.eval.chance_level import chance_threshold

ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = ROOT / "results" / "LOG.md"
LOG_COLUMNS = ["date", "commit", "dataset", "config", "protocol", "seed", "n_test", "acc", "chance threshold"]


def plot_confusion(preds: pd.DataFrame, class_names: list[str], title: str, path: Path) -> np.ndarray:
    """Row-normalised confusion matrix (each row sums to 1). Returns the raw count matrix."""
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=range(len(class_names)))
    frac = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{frac[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def plot_per_subject(preds: pd.DataFrame, n_classes: int, title: str, path: Path, alpha: float = 0.05) -> pd.DataFrame:
    """Bar per subject with that subject's own chance threshold marked. Returns the per-subject table."""
    g = preds.assign(correct=preds["y_true"] == preds["y_pred"]).groupby("subject")["correct"]
    tab = pd.DataFrame({"n_test": g.size(), "acc": g.mean()})
    tab["chance_threshold"] = [chance_threshold(int(n), n_classes, alpha) for n in tab["n_test"]]
    tab["above_chance"] = tab["acc"] > tab["chance_threshold"]
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(tab) + 2), 4))
    x = np.arange(len(tab))
    ax.bar(x, tab["acc"], color=np.where(tab["above_chance"], "#1f6feb", "#9aa4b1"))
    ax.hlines(tab["chance_threshold"], x - 0.4, x + 0.4, color="#d1242f", lw=1.5, label="chance threshold (p<0.05)")
    ax.axhline(1 / n_classes, color="#555", lw=0.8, ls=":", label="1/k")
    ax.set_xticks(x, [str(s) for s in tab.index])
    ax.set(xlabel="Subject", ylabel="Accuracy", ylim=(0, 1), title=title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return tab


def plot_null(observed: float, null: np.ndarray, p: float, title: str, path: Path) -> None:
    """Histogram of shuffled-label accuracies with the real accuracy marked."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(null, bins=30, color="#9aa4b1")
    ax.axvline(observed, color="#1f6feb", lw=2, label=f"real labels: {observed:.3f}")
    ax.set(xlabel="Pooled accuracy", ylabel="Shuffles", title=f"{title}\n{len(null)} shuffles, p = {p:.4f}")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def git_commit() -> str:
    """Short commit hash, with '-dirty' when the working tree has uncommitted changes to tracked files."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_log_row(dataset: str, config: str, protocol: str, seed: int, n_test: int, acc: float,
                   threshold: float, path: Path = LOG_PATH, commit: str | None = None) -> str:
    """Append one experiment row to results/LOG.md and return it."""
    cells = [dt.date.today().isoformat(), commit or git_commit(), dataset, config.replace("|", "/"), protocol,
             str(seed), str(n_test), f"{acc:.3f}", f"{threshold:.3f}"]
    row = "| " + " | ".join(cells) + " |"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("| " + " | ".join(LOG_COLUMNS) + " |\n|" + "---|" * len(LOG_COLUMNS) + "\n", encoding="utf-8")
    with open(path, "a", encoding="utf-8") as f:
        f.write(row + "\n")
    return row
