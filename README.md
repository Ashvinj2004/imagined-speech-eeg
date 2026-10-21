# imagined-speech-eeg

Decoding imagined (inner) speech from non-invasive EEG on public datasets: a rigorous, leakage-safe benchmark.

Imagined speech = words said silently in the head, with no movement. We aim to *honestly measure a
modest effect*: 55–70% on 2–5 classes is success; ~95% almost always means data leakage.
Read [`CLAUDE.md`](CLAUDE.md) (project rules) and [`docs/LEAKAGE_CHECKLIST.md`](docs/LEAKAGE_CHECKLIST.md) before contributing.

## Setup

Python 3.11. On Windows (Git Bash shown; in PowerShell use `.venv\Scripts\Activate.ps1`):

```bash
py -3.11 -m venv .venv            # or: python3.11 -m venv .venv
source .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch; skip if you have a CUDA GPU and want the GPU build
pip install -r requirements.txt   # or, to reproduce exactly: pip install -r requirements.lock
pip install -e .
python -m pytest -q
python scripts/smoke_test.py
python scripts/check_env.py      # compare this output across all four laptops
```

**Windows Smart App Control / Application Control:** if imports fail with
`DLL load failed ... An Application Control policy has blocked this file`, Windows is blocking a
compiled package. We pin `pillow==11.3.0` in `requirements.lock` because newer builds were blocked on
one team machine. `torch` may also be blocked; Sprint 1 does not need it (only `braindecode` models will).

## Repo layout

```
CLAUDE.md                  project rules for every session (humans and AI)
src/bci/
  data/base.py             EpochsDataset (data contract) + loader registry (@register, load)
  data/synthetic.py        "synthetic" dataset: plumbing tests only, never reported
  data/track3.py           "track3": BCI Competition 2020 Track 3 (5 imagined phrases, 15 subjects)
  data/nieto.py            "nieto": Nieto et al. 2022 inner speech (4 directions, 10 subjects x 3 sessions)
  eval/chance_level.py     binomial chance threshold, binomial / permutation p-values
  eval/splits.py           within_subject, cross_session, loso, predefined + leakage checker
  eval/harness.py          evaluate(), summarize(), permutation_null()
  models/                  model factories: majority(), riemann_ts_lr()
scripts/download_data.py   fetch Track 3 (~0.6 GB) and Nieto (~6.7 GB) into data/
scripts/smoke_test.py      end-to-end check on synthetic data
scripts/check_env.py       package versions + dataset shape (Sprint 1 gate)
notebooks/                 exploration only; reported numbers come from the harness
.github/workflows/         CI: env check, pytest and smoke test on every PR
tests/                     pytest suite
docs/LEAKAGE_CHECKLIST.md  rules -> enforcement table, team sign-off
docs/DATASETS.md           what each dataset contains, where it comes from, and its traps
results/LOG.md             one row per real experiment
```

## Data contract

Every loader returns a `bci.data.EpochsDataset`:

| field | type | meaning |
|-------|------|---------|
| `X` | float32 `(trials, channels, samples)` | one epoch (fixed-length EEG window around a cue) per trial |
| `y` | int `(trials,)`, values `0..k-1` | class label; `class_names[i]` names label `i` |
| `subject`, `session` | `(trials,)` | who and which recording session (cap put on again = new session) |
| `trial_id` | `(trials,)`, unique per original trial | crops/windows of one trial **share** their parent's id |
| `split` | optional `(trials,)` of `"train"/"val"/"test"` | dataset authors' predefined split |
| `sfreq`, `ch_names`, `class_names`, `name` | | sampling rate (Hz), electrode names, label names, dataset name |

Shapes and label ranges are validated on construction. Register a new loader with
`@register("name")` and load it with `bci.data.load("name", **kwargs)`.

## Usage

```python
from bci.data import load
from bci.eval.harness import evaluate, summarize
from bci.models import riemann_ts_lr
ds = load("synthetic", seed=0)
print(summarize(evaluate(riemann_ts_lr, ds, ["within_subject", "cross_session", "loso"])))
```
