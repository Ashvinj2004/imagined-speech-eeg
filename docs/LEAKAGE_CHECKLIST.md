# Leakage & honesty checklist

Every team member reads this, checks the code paths named below, and signs at the bottom.
Re-sign whenever a rule or its enforcement changes.

| # | Rule | How we enforce it in this code |
|---|------|--------------------------------|
| 1 | **No data leakage.** Split by original trial (crops inherit the parent `trial_id`); cross-subject = LOSO only; cross-session = held-out sessions only; anything learned from data is fit on the training fold only, inside the model pipeline. Fixed filters may run before splitting. | `EpochsDataset.trial_id` is required. `bci.eval.splits` groups by `trial_id` (`StratifiedGroupKFold`) and every fold passes `check_no_leakage`, which raises `LeakageError` if a `trial_id`, a subject (LOSO) or a (subject, session) pair (cross-session) is on both sides. `evaluate()` takes a model **factory** and builds a fresh pipeline per fold, so scalers, covariance means, etc. are fit on train only. Tests: `tests/test_crops.py`, `tests/test_leakage_checker.py`. |
| 2 | **Honest chance level.** Never compare to 1/k; use the Combrisson & Jerbi (2015) binomial threshold for the exact n_test, plus permutation tests for headline numbers. | `bci.eval.chance_level.chance_threshold` / `binomial_pvalue` are in every row of `evaluate()` and `summarize()`; `bci.eval.harness.permutation_null` + `permutation_pvalue` for headline numbers. Tests: `tests/test_chance_level.py` (threshold > 1/k, convergence, ~alpha false-positive rate). |
| 3 | **Report within-subject, cross-session and LOSO separately**; the drop is a finding. | `evaluate()` runs protocols separately and `summarize()` reports one row per protocol (never a single blended number). |
| 4 | **> 90% on 3+ classes = probable leakage.** Stop, investigate, don't commit it as a result. | `summarize()` prints a loud stderr banner and raises a `UserWarning` (test: `test_summarize_warns_on_suspicious_accuracy`). Reviewers reject PRs that log such a number without an investigation note. |
| 5 | **No hype claims:** never "mind reading", sentence/continuous decoding, or plug-and-play cross-subject claims. | Rule in `CLAUDE.md`; PR review checks code, docs, slides and commit messages. |
| 6 | **Log every experiment** in `results/LOG.md` (date, commit, dataset, config, protocol, seed, n_test, acc, chance threshold). | Table template in `results/LOG.md`; PR review checks a new result has a row. Synthetic runs are never logged. |
| 7 | **Explain EEG terms** in a one-line comment the first time they appear. | Rule in `CLAUDE.md`; PR review. |
| 8 | **Branch per feature, PRs only, tests green before every PR, fixed seeds everywhere.** | `python -m pytest -q` and `python scripts/smoke_test.py` must pass before opening a PR; all generators/splitters/models take an explicit `seed`. |

## Sign-off

I have read the rules above and checked how they are enforced.

| Initials | Date |
|----------|------|
| ____ | ____ |
| ____ | ____ |
| ____ | ____ |
| ____ | ____ |
