"""Turn results/LOG.md into the report's tables, so every table traces back to log rows.

    python scripts/build_tables.py              # prints the tables and writes results/tables/*.md

For each (dataset, model, protocol) the latest log row wins (later rows supersede reruns), so the
tables reflect the newest configuration of each experiment. Rows whose config is a frequency-band
or ablation variant are listed in a separate ablation table.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "LOG.md"
OUT = ROOT / "results" / "tables"

PROTOCOL_ORDER = ["predefined_val", "predefined", "within_subject", "cross_session", "loso"]
PROTOCOL_LABEL = {
    "predefined_val": "official train -> val",
    "predefined": "official test",
    "within_subject": "within-subject (5-fold)",
    "cross_session": "cross-session",
    "loso": "LOSO",
}
MODEL_LABEL = {"majority": "Majority class", "riemann": "Riemannian TS+LR", "riemann_ea": "Riemannian TS+LR + EA",
               "eegnet": "EEGNet", "shallow": "ShallowConvNet"}


def read_log(path: Path = LOG) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 20"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 9:
            continue
        date, commit, dataset, config, protocol, seed, n_test, acc, thr = cells
        model = config.split(";")[0].strip()
        bp = re.search(r"bp=([\d.]+)-([\d.]+)", config)
        perm = re.search(r"perm_p=([\d.]+) \(n=(\d+)\)", config)
        rows.append({
            "date": date, "commit": commit, "dataset": dataset, "model": model, "protocol": protocol,
            "seed": int(seed), "n_test": int(n_test), "acc": float(acc), "threshold": float(thr),
            "band": f"{bp.group(1)}-{bp.group(2)} Hz" if bp else "",
            "perm_p": float(perm.group(1)) if perm else None, "n_perm": int(perm.group(2)) if perm else None,
            "variant": re.search(r"variant=(\S+)", config).group(1) if "variant=" in config else "",
            "config": config,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["above"] = df["acc"] > df["threshold"]
    return df


def fmt_acc(r) -> str:
    s = f"{r['acc']:.3f}"
    if r["perm_p"] is not None and not pd.isna(r["perm_p"]):
        s += f" (p={r['perm_p']:.3f}, {int(r['n_perm'])} shuffles)"
    return s + ("" if r["above"] else " *")


def main_table(df: pd.DataFrame) -> str:
    """Rows: dataset × model; columns: protocols. Latest row per cell."""
    base = df[df["variant"] == ""].sort_values("date").groupby(["dataset", "model", "protocol"]).tail(1)
    out = []
    for dataset, g in base.groupby("dataset", sort=True):
        protos = [p for p in PROTOCOL_ORDER if p in set(g["protocol"])]
        out.append(f"### {dataset}\n")
        out.append("| Model | " + " | ".join(PROTOCOL_LABEL[p] for p in protos) + " |")
        out.append("|---|" + "---|" * len(protos))
        for model in [m for m in MODEL_LABEL if m in set(g["model"])]:
            cells = []
            for p in protos:
                r = g[(g["model"] == model) & (g["protocol"] == p)]
                cells.append(fmt_acc(r.iloc[0]) if len(r) else "-")
            out.append(f"| {MODEL_LABEL[model]} | " + " | ".join(cells) + " |")
        thr = {p: g[g["protocol"] == p].iloc[0] for p in protos}
        out.append("\nChance threshold (n_test): " + "; ".join(
            f"{PROTOCOL_LABEL[p]} {thr[p]['threshold']:.3f} (n={thr[p]['n_test']})" for p in protos))
        out.append("\n\\* = not above the chance threshold.\n")
    return "\n".join(out)


def ablation_table(df: pd.DataFrame) -> str:
    abl = df[df["variant"] != ""].sort_values("date").groupby(["dataset", "model", "protocol", "variant"]).tail(1)
    if abl.empty:
        return "_No ablation rows yet._\n"
    out = ["| Dataset | Model | Variant | Band | " + " | ".join(PROTOCOL_LABEL[p] for p in PROTOCOL_ORDER) + " |",
           "|---|---|---|---|" + "---|" * len(PROTOCOL_ORDER)]
    # order bands by their low edge, not alphabetically (band1-8 before band8-13 before band13-30)
    def key(row) -> tuple:
        lo = re.match(r"band([\d.]+)-", row["variant"])
        return (row["dataset"], float(lo.group(1)) if lo else 1e9, row["variant"])

    abl = abl.assign(_k=abl.apply(key, axis=1)).sort_values("_k")
    for (dataset, model, variant), g in abl.groupby(["dataset", "model", "variant"], sort=False):
        cells = []
        for p in PROTOCOL_ORDER:
            r = g[g["protocol"] == p]
            cells.append(fmt_acc(r.iloc[0]) if len(r) else "-")
        out.append(f"| {dataset} | {MODEL_LABEL.get(model, model)} | {variant} | {g['band'].iloc[0]} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    df = read_log()
    if df.empty:
        print("no rows in results/LOG.md", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    main_md = "# Main results (from results/LOG.md)\n\n" + main_table(df)
    abl_md = "# Ablations (from results/LOG.md)\n\n" + ablation_table(df)
    (OUT / "main.md").write_text(main_md, encoding="utf-8")
    (OUT / "ablations.md").write_text(abl_md, encoding="utf-8")
    print(main_md)
    print(abl_md)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
