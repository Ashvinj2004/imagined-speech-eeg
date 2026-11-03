"""Run one model on one dataset under one or more protocols, save figures, and log the result.

    python scripts/run_experiment.py --dataset track3 --model riemann --protocol predefined_val
    python scripts/run_experiment.py --dataset track3 --model eegnet --protocol predefined_val --subjects 1 2 3
    python scripts/run_experiment.py --dataset track3 --model riemann --protocol predefined --final --n-perm 1000

Safety locks:
- Protocols that use a dataset's held-out TEST split (`predefined`) refuse to run without --final.
  Choose settings with `predefined_val` (train -> validation) first; touch test once per final config.
- Every real-data run appends a row to results/LOG.md (RULES.md rule 6). Synthetic runs never do.
- A pooled accuracy above 90% with 3+ classes is not logged (RULES.md rule 4): investigate first.

Outputs go to results/runs/<timestamp>_<dataset>_<model>_<protocol>/ (not committed): per-fold and
per-trial CSVs, the summary, a confusion matrix, a per-subject chart, and the shuffle-test histogram.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from bci.data import load
from bci.eval.chance_level import permutation_pvalue
from bci.eval.harness import LEAKAGE_SUSPICION_ACC, evaluate, permutation_null, summarize
from bci.eval.report import ROOT, append_log_row, git_commit, plot_confusion, plot_null, plot_per_subject
from bci.eval.splits import TEST_SET_PROTOCOLS
from bci.models import eegnet, majority, riemann_ts_lr, shallow_convnet
from bci.preprocess import default_config, preprocess

DEEP = {"eegnet", "shallow"}


def make_factory(name: str, seed: int, max_epochs: int | None):
    extra = {} if max_epochs is None else {"max_epochs": max_epochs}
    return {
        "majority": lambda: majority(),
        "riemann": lambda: riemann_ts_lr(seed=seed),
        "eegnet": lambda: eegnet(seed=seed, **extra),
        "shallow": lambda: shallow_convnet(seed=seed, **extra),
    }[name]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["track3", "nieto", "synthetic"])
    ap.add_argument("--model", required=True, choices=["majority", "riemann", "eegnet", "shallow"])
    ap.add_argument("--protocol", nargs="+", required=True)
    ap.add_argument("--subjects", nargs="*", type=int, help="default: all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=0, help="label shuffles for the permutation test (0 = skip)")
    ap.add_argument("--resample", type=float, default=None,
                    help="target sampling rate in Hz; default 128 for neural networks, unchanged otherwise")
    ap.add_argument("--l-freq", type=float, default=None, help="override the band-pass low edge (Hz)")
    ap.add_argument("--h-freq", type=float, default=None, help="override the band-pass high edge (Hz)")
    ap.add_argument("--max-epochs", type=int, default=None, help="neural networks only")
    ap.add_argument("--final", action="store_true", help="allow protocols that use the held-out test split")
    args = ap.parse_args(argv)

    touched = TEST_SET_PROTOCOLS.intersection(args.protocol)
    if touched and not args.final:
        print(f"refusing: {sorted(touched)} uses the held-out TEST split. Tune with predefined_val, then rerun "
              "the chosen configuration once with --final.", file=sys.stderr)
        return 2

    kwargs = {"subjects": args.subjects} if args.subjects and args.dataset != "synthetic" else {}
    raw = load(args.dataset, **kwargs)
    cfg = default_config(raw)
    resample = args.resample if args.resample is not None else (128.0 if args.model in DEEP else None)
    if resample is not None:
        cfg = replace(cfg, resample_hz=resample)
    if args.l_freq is not None:
        cfg = replace(cfg, l_freq=args.l_freq)
    if args.h_freq is not None:
        cfg = replace(cfg, h_freq=args.h_freq)
    ds = preprocess(raw, cfg)
    print(ds.summary())
    print(f"preprocessing: {cfg}")

    factory = make_factory(args.model, args.seed, args.max_epochs)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "results" / "runs" / f"{stamp}_{args.dataset}_{args.model}_{'+'.join(args.protocol)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    folds, preds = evaluate(factory, ds, args.protocol, seed=args.seed, return_predictions=True)
    summary = summarize(folds, n_classes=ds.n_classes)
    folds.to_csv(run_dir / "folds.csv", index=False)
    preds.to_csv(run_dir / "predictions.csv", index=False)

    status = 0
    for _, row in summary.iterrows():
        proto = row["protocol"]
        p_pred = preds[preds["protocol"] == proto]
        label = f"{raw.name} | {args.model} | {proto}"
        plot_confusion(p_pred, ds.class_names, label, run_dir / f"confusion_{proto}.png")
        plot_per_subject(p_pred, ds.n_classes, label, run_dir / f"per_subject_{proto}.png")

        perm_note = ""
        if args.n_perm:
            null = permutation_null(factory, ds, proto, n_perm=args.n_perm, seed=args.seed)
            p = permutation_pvalue(row["acc"], null)
            summary.loc[summary["protocol"] == proto, "perm_p"] = p
            np.save(run_dir / f"null_{proto}.npy", null)
            plot_null(row["acc"], null, p, label, run_dir / f"null_{proto}.png")
            perm_note = f" perm_p={p:.4f} (n={args.n_perm})"

        config = (f"{args.model}; notch={cfg.notch_hz} bp={cfg.l_freq}-{cfg.h_freq} reref={cfg.reref} "
                  f"fs={ds.sfreq:g}" + (f" max_epochs={args.max_epochs}" if args.max_epochs else "") + perm_note)
        if args.dataset == "synthetic":
            print(f"[not logged: synthetic] {proto}: acc={row['acc']:.3f}")
        elif ds.n_classes >= 3 and row["acc"] > LEAKAGE_SUSPICION_ACC:
            print(f"[NOT LOGGED] {proto}: acc={row['acc']:.3f} > {LEAKAGE_SUSPICION_ACC}: investigate leakage first",
                  file=sys.stderr)
            status = 3
        else:
            print("logged:", append_log_row(raw.name, config, proto, args.seed, int(row["n_test"]), row["acc"],
                                            row["chance_threshold"], commit=commit))

    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "preprocess": asdict(cfg), "commit": commit},
                                                    indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"outputs: {run_dir}")
    return status


if __name__ == "__main__":
    sys.exit(main())
