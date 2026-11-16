"""Demo: classify one held-out trial at a time from a saved public recording.

    python scripts/demo.py                      # Track 3, subject 1, official test split
    python scripts/demo.py --subject 7 --delay 0.5
    python scripts/demo.py --dataset nieto --subject 2

Trains the Riemannian model on the subject's training trials (plus validation for Track 3), then
walks through their held-out trials one by one, printing the true phrase, the prediction and a
running accuracy next to the chance threshold. For Nieto (no official split) the last session is
held out, so the demo is a cross-session test.

This is a replay of saved data, not a live recording.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from bci.data import load
from bci.eval.chance_level import chance_threshold
from bci.models import riemann_ts_lr
from bci.preprocess import default_config, preprocess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="track3", choices=["track3", "nieto"])
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between trials (0 = as fast as possible)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = load(args.dataset, subjects=[args.subject])
    ds = preprocess(raw, default_config(raw))
    if args.dataset == "track3":
        train = np.isin(ds.split, ["train", "val"])
        test = ds.split == "test"
        held = "official test split"
    else:
        last = ds.session.max()
        train, test = ds.session != last, ds.session == last
        held = f"session {last} (cross-session)"
    rng = np.random.default_rng(args.seed)
    test_idx = rng.permutation(np.flatnonzero(test))

    model = riemann_ts_lr(seed=args.seed).fit(ds.X[train], ds.y[train])
    n, k = len(test_idx), ds.n_classes
    thr = chance_threshold(n, k)
    print(f"{raw.name}, subject {args.subject}: trained on {int(train.sum())} trials, replaying {n} from the {held}")
    print(f"classes: {', '.join(ds.class_names)} | chance threshold for n={n}: {thr:.1%} (1/k = {1 / k:.0%})\n")

    correct = 0
    for i, t in enumerate(test_idx, 1):
        pred = int(model.predict(ds.X[t : t + 1])[0])
        hit = pred == ds.y[t]
        correct += hit
        print(f"trial {i:3d}  true: {ds.class_names[ds.y[t]]:<10} predicted: {ds.class_names[pred]:<10} "
              f"{'ok ' if hit else 'x  '}  running accuracy {correct / i:.1%}", flush=True)
        if args.delay:
            time.sleep(args.delay)

    acc = correct / n
    verdict = "above" if acc > thr else "not above"
    print(f"\nfinal: {correct}/{n} = {acc:.1%}, {verdict} the chance threshold of {thr:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
