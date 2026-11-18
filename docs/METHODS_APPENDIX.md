# Methods appendix

Exact settings behind every number in `docs/PAPER.md`. Dataset provenance is in
`docs/DATASET_CARDS.md`; per-run rows are in `results/LOG.md`.

## Software

Python 3.11.9. Pinned in `requirements.lock`; the versions that produced the reported results:
numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, scikit-learn 1.9.1, mne 1.13.2, pyriemann 0.12,
torch 2.11.0+cpu, torchaudio 2.11.0, braindecode 1.8.1, moabb 1.7.2, pillow 11.3.0.

`torch` and `torchaudio` must share a version; braindecode imports torchaudio, whose last release is
2.11.0. All runs were on CPU (no CUDA GPU available).

**Windows Smart App Control.** Two packages are pinned below their latest release because Smart App
Control blocks their unsigned compiled libraries: **pillow 11.3.0** and **matplotlib 3.9.2**. The
matplotlib pin is not optional for this project: `pyriemann` imports `matplotlib.pyplot` at module
level, so a blocked matplotlib makes the Riemannian model — the project's main classifier —
unimportable. Symptom: `ImportError: DLL load failed while importing ft2font`.

**Reproducibility of the neural-network runs.** Seeds fix initialisation, the inner train/validation
split and batch order, but multithreaded CPU reductions in torch sum in a non-deterministic order.
The resulting float differences can flip a borderline prediction, so EEGNet and ShallowConvNet
accuracies reproduce to within roughly one trial rather than bit-for-bit, unless `torch.set_num_threads(1)`
is set (which the determinism test does). The Riemannian results are unaffected.

## Preprocessing (`src/bci/preprocess.py`)

Fixed and data-independent, so it may run before splitting; applied to each trial separately, in
chunks of 256 trials (memory only; the output does not depend on the chunk size).

| Step | Setting |
|---|---|
| Re-reference | Common average (Track 3). None for Nieto: the authors already re-referenced |
| Notch | 60 Hz, quality factor 30, IIR, zero-phase (`filtfilt`). Track 3 only; Nieto is already notched at 50 Hz by its authors |
| Band-pass | Butterworth order 4, second-order sections, zero-phase (`sosfiltfilt`). 0.5–30 Hz for the Riemannian headline runs, 0.5–40 Hz for the networks and Nieto, single bands for the ablation |
| Edge padding | Odd-symmetric, `min(n_samples - 1, 3 × sfreq)` samples |
| Resample | None (Riemannian, 256 Hz) or polyphase to 128 Hz (networks), always after a band-pass below the new Nyquist frequency |

Zero-phase filtering (forward then backward) avoids the time shift a causal filter would introduce.
Filtering per trial matters for Track 3: its public copy stores trials end to end, so filtering the
concatenated recording would smear neighbouring trials into one another.

## Models

| Model | Definition |
|---|---|
| Majority class | `sklearn.dummy.DummyClassifier(strategy="most_frequent")` |
| Riemannian TS+LR | `Covariances(estimator="oas")` → `TangentSpace(metric="riemann")` → `LogisticRegression(C=1.0, max_iter=1000)`, one sklearn `Pipeline` |
| Riemannian + EA | Per-subject mean covariance R from the training fold; each trial whitened as R^(−1/2) C R^(−1/2); then tangent space + logistic regression. For an unseen participant, R comes from their own unlabelled test trials (unsupervised calibration), recorded in `unseen_subjects_` |
| EEGNet | `braindecode.models.EEGNet`, defaults, 128 Hz input |
| ShallowConvNet | `braindecode.models.ShallowFBCSPNet`, `final_conv_length="auto"`, 128 Hz input |

Network training: AdamW; learning rate 1e-3 (EEGNet) or 6.25e-4 (ShallowConvNet); weight decay 0;
batch size 32; at most 150 passes; early stopping with patience 30 on an inner validation split
(20% of the training fold, stratified); the best state by inner validation loss is restored.
Per-channel mean/standard-deviation scaling is computed from the inner training trials only.
Everything learned from data lives inside `fit`, so the harness fits it per fold.

## Protocols (`src/bci/eval/splits.py`)

| Protocol | Definition |
|---|---|
| `predefined_val` | Per participant: train on the official train split, test on the official validation split. Used to choose settings; never reads the test split |
| `predefined` | Per participant: train on train + validation, test on the official test split. One run per final configuration |
| `within_subject` | Per participant: 5-fold `StratifiedGroupKFold`, grouped by `trial_id`, shuffled with the run's seed |
| `cross_session` | Per participant: leave one session out; errors on single-session data |
| `loso` | Leave one participant out |

Every fold passes `check_no_leakage`, which raises if a `trial_id` appears on both sides, if a
participant appears on both sides under LOSO, or if a (participant, session) pair appears on both
sides under cross-session.

## Cropped training (`src/bci/data/crops.py`)

Sliding windows of 1.0 s every 0.5 s within each trial (5 windows per 3.1 s Track 3 trial, 4 per
2.5 s Nieto trial). Each crop inherits its parent's label, participant, session and `trial_id`, so
the splitters keep all crops of a trial on one side. At test time the crops of a trial are combined
by majority vote and the trial is counted once, so `n_test` stays the number of real trials and
chance thresholds are computed on trials rather than on correlated windows.

## Statistics (`src/bci/eval/chance_level.py`)

- **Chance threshold:** `binom.ppf(1 − α, n, 1/k) / n` with α = 0.05 (Combrisson & Jerbi, 2015),
  computed for the exact test size — pooled for pooled accuracies, per participant for
  per-participant claims.
- **Binomial p-value:** `binom.sf(n_correct − 1, n, 1/k)`.
- **Permutation null:** labels shuffled within participant, the whole evaluation rerun;
  p = (count(null ≥ observed) + 1) / (n_shuffles + 1). Run for the Track 3 headline result
  (100 shuffles). Not run for chance-level results, where the binomial p-value already answers the
  question and each shuffle costs a full re-evaluation.

## Seeds and logging

Every generator, splitter and model takes an explicit seed (0 for all reported runs); no global RNG
state is used. `scripts/run_experiment.py` appends one row per run to `results/LOG.md` with the
date, commit (marked `-dirty` if tracked files had uncommitted changes), dataset, configuration,
protocol, seed, test size, accuracy and chance threshold. Synthetic-data runs are never logged, and
a pooled accuracy above 0.90 with 3+ classes is refused and flagged for a leakage investigation.
