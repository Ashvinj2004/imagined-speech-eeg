"""Export real per-trial predictions to one JSON for the live demo page.

    python scripts/export_demo_data.py

Reads the latest matching run folders in results/runs/ and writes
results/figures/demo_data.json: for each scenario, the true and predicted label of every held-out
trial, in the order the harness produced them, plus the chance threshold for that test size.

The demo page replays these logged predictions. It does not run a model in the browser, and the
page says so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bci.eval.chance_level import chance_threshold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures" / "demo_data.json"

TRACK3 = ["Hello", "Help me", "Stop", "Thank you", "Yes"]
NIETO = ["Up", "Down", "Right", "Left"]

SCENARIOS = [
    ("track3_own", "*track3_riemann_predefined", "predefined", TRACK3,
     "Track 3 — the person's own data", "Trained on this person's 350 training trials, tested on their 50 held-out trials."),
    ("track3_new", "*track3_riemann_loso", "loso", TRACK3,
     "Track 3 — a brand-new person", "Trained on the other 14 people, tested on someone the model has never seen."),
    ("nieto_own", "*nieto_riemann_within*", "within_subject", NIETO,
     "Nieto — the person's own data", "Four imagined direction words, 5-fold cross-validation within each person."),
    ("nieto_day", "*nieto_riemann_within*", "cross_session", NIETO,
     "Nieto — the same person, another day", "Trained on two lab visits, tested on a third recorded on a different day."),
]


def latest(pattern: str) -> Path | None:
    """Newest run matching the pattern that is a BASELINE run: no ablation variant and no crops.

    Several ablation runs share a folder-name pattern with the baseline, so the folder name alone
    would silently pick up a band or crops variant and contradict the reported numbers."""
    hits = []
    for p in sorted(RUNS.glob(pattern)):
        cfg_path, preds = p / "config.json", p / "predictions.csv"
        if not (cfg_path.is_file() and preds.is_file()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if cfg.get("variant") or cfg.get("crop"):
            continue
        hits.append(p)
    return hits[-1] if hits else None


def main() -> int:
    out = {"scenarios": [], "note": "Real per-trial predictions from logged runs; the page replays them."}
    for key, pattern, protocol, classes, title, subtitle in SCENARIOS:
        run = latest(pattern)
        if run is None:
            print(f"skipping {key}: no run matching {pattern}", file=sys.stderr)
            continue
        preds = pd.read_csv(run / "predictions.csv")
        preds = preds[preds["protocol"] == protocol]
        if preds.empty:
            print(f"skipping {key}: no {protocol} rows in {run.name}", file=sys.stderr)
            continue
        by_subject = {}
        for subj, g in preds.groupby("subject"):
            by_subject[str(int(subj))] = {"t": [int(v) for v in g["y_true"]], "p": [int(v) for v in g["y_pred"]]}
        n = len(preds)
        correct = int((preds["y_true"] == preds["y_pred"]).sum())
        per_subject_n = len(next(iter(by_subject.values()))["t"])
        out["scenarios"].append({
            "key": key, "title": title, "subtitle": subtitle, "classes": classes,
            "protocol": protocol, "run": run.name,
            "pooled": {"n": n, "correct": correct, "acc": correct / n,
                       "threshold": chance_threshold(n, len(classes))},
            "subject_threshold": chance_threshold(per_subject_n, len(classes)),
            "one_over_k": 1 / len(classes),
            "subjects": by_subject,
        })
        print(f"{key}: {len(by_subject)} people, {n} trials, pooled {correct / n:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"written {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
