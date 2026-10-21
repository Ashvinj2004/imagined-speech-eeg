# CLAUDE.md: rules for every session in this repo

Project: decoding **imagined speech** (words said silently in the head, no movement)
from **non-invasive EEG**, using **public datasets only**. 4-person undergraduate capstone.

Goal: a methodologically rigorous study that honestly measures a modest effect.
**55-70% on 2-5 classes is success. 95% almost always means data leakage.**

## The rules (non-negotiable)

1. **No data leakage.**
   - Split by *original trial*: crops/windows cut from one trial never land in both train and test.
     Every crop inherits its parent's `trial_id`; splitters group by `trial_id`.
   - Cross-subject claims use **leave-one-subject-out (LOSO)** only.
     Cross-session claims use **held-out sessions** only.
   - Anything that learns from data (z-scoring, ICA, PCA, Euclidean Alignment stats, feature
     selection, class balancing, early stopping) is **fit on the training fold only, inside the
     model pipeline** (an sklearn `Pipeline` or equivalent built by the model factory).
   - Fixed, data-independent filters (notch, band-pass, re-reference, resample) may run before splitting.
2. **Honest chance level.** Never compare to 1/k. Use the Combrisson & Jerbi (2015) binomial
   threshold for the exact test-set size (`bci.eval.chance_level.chance_threshold`), plus
   label-permutation tests (`bci.eval.harness.permutation_null`) for headline numbers.
3. **Report within-subject, cross-session and LOSO separately.** The drop between them is a finding,
   not something to hide.
4. **Any accuracy > 90% on 3+ classes is probable leakage.** Stop and investigate. Do not commit it as a result.
5. **Never write** "mind reading", sentence/continuous decoding, or plug-and-play cross-subject
   claims anywhere (code, docs, slides, commit messages).
6. **Every experiment appends a row to `results/LOG.md`**:
   date | commit | dataset | config | protocol | seed | n_test | acc | chance threshold.
   Synthetic-data runs are plumbing checks and are never logged or reported.
7. **The team knows Python/ML but is new to EEG.** Explain each EEG/signal-processing term in a
   one-line comment the first time it appears in a file.
8. **Workflow:** one branch per feature, PRs only (no direct pushes to `main`), tests green before
   every PR, fixed seeds everywhere (pass `seed`/`random_state` explicitly; no global RNG state).

## Commands

```bash
.venv/Scripts/python -m pytest -q          # unit tests (Windows; use .venv/bin/python on Linux/macOS)
.venv/Scripts/python scripts/smoke_test.py # end-to-end plumbing check on synthetic data
```

## Code map

- `src/bci/data/base.py`: `EpochsDataset` (the data contract) and the loader registry (`@register`, `load`).
- `src/bci/data/synthetic.py`: `"synthetic"` dataset for plumbing tests only. Never report its numbers.
- `src/bci/data/track3.py`, `src/bci/data/nieto.py`: real-data loaders (`"track3"`, `"nieto"`); data comes from
  `scripts/download_data.py` into the gitignored `data/`. Dataset facts and traps: `docs/DATASETS.md`.
  Never unpickle downloaded files with plain `pickle`/`np.load(allow_pickle=True)`; see `nieto._ArrayOnlyUnpickler`.
- `src/bci/eval/chance_level.py`: binomial chance threshold, binomial and permutation p-values.
- `src/bci/eval/splits.py`: `within_subject`, `cross_session`, `loso`, `predefined` splitters; all folds pass `check_no_leakage`.
- `src/bci/eval/harness.py`: `evaluate(make_model, ds, protocols)`, `summarize(df)`, `permutation_null(...)`.
- `src/bci/models/`: model **factories** (zero-arg callables returning a fresh unfitted estimator).
- `docs/LEAKAGE_CHECKLIST.md`: rule → enforcement table; each team member signs off.

## Conventions for new code

- New datasets: a loader decorated with `@register("name")` that returns a validated `EpochsDataset`
  (X float32 `(trials, channels, samples)`, y ints `0..k-1`, per-trial `subject`/`session`/`trial_id`).
- New models: a factory in `bci.models` returning an sklearn-compatible estimator; every learned
  preprocessing step lives inside it so the harness fits it per fold.
- Never evaluate outside `bci.eval.harness`; never hand-roll a train/test split.
