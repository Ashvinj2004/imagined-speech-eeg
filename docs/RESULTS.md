# Results

Raw rows: `results/LOG.md`. Each number below comes from `scripts/run_experiment.py` at the commit shown
in the log. Chance thresholds are the Combrisson & Jerbi binomial thresholds (alpha = 0.05).

## Sprint 2: BCI Competition 2020 Track 3, official split (within-subject)

5 imagined phrases, 15 subjects, one model per subject. Preprocessing: average reference, 60 Hz notch,
band-pass as listed, applied per trial. Neural networks run at 128 Hz.

Settings were chosen on train → validation (`predefined_val`); each chosen configuration was then run
**once** on the official test split (`predefined`: train + validation → test).

| Model | Band-pass | Validation (n=750) | **Test (n=750)** | Subjects above own threshold (test) | Shuffle test (test) |
|---|---|---|---|---|---|
| Majority class | 0.5–40 Hz | 0.200 | **0.200** | 0/15 | |
| EEGNet | 0.5–40 Hz | 0.233 | **0.257** | | |
| ShallowConvNet | 0.5–40 Hz | 0.240 | **0.285** | | |
| Riemannian tangent space + LR | 0.5–40 Hz | 0.405 | not run | | |
| Riemannian tangent space + LR | 0.5–30 Hz | 0.415 | **0.404** | 12/15 (range 0.26–0.52) | p = 0.0099 (100 shuffles; null mean 0.196, max 0.237) |

Pooled chance threshold for n = 750: **0.224** (1/k = 0.200). Per-subject threshold for n = 50: 0.300.

### What we conclude

- **The pipeline reproduces the published pattern.** The majority baseline sits exactly at 1/k, and
  ShallowConvNet on validation (0.240) matches the 0.244 a recent leakage-controlled re-analysis reports
  for the same network (arXiv 2610.20954). Neural networks are weak with ~300 training trials per person.
- **The Riemannian baseline is clearly above chance:** 0.404 on test, 303/750 correct, binomial
  p ≈ 2e-37, no shuffled run within 0.16 of it. Validation and test agree (0.415 vs 0.404), so the
  validation split did not give an optimistic picture.
- It is **higher than the ~0.31–0.33** that re-analysis reports for band-power models on this corpus.
  Plausible reasons: a different feature set (full covariance vs band power) and a different protocol
  (official split vs their 5-fold and held-out-block designs). Not yet explained; see open questions.
- Cutting the band at 30 Hz instead of 40 Hz did not lower accuracy (0.415 vs 0.405 on validation), so the
  30–40 Hz range, where facial muscle activity is strongest, is not what drives the score.
- The competition's 82.6% is not a reproducible target: the review does not say which split it was
  measured on, and later re-analyses did not reproduce it.

### Open questions (for Sprints 3–4)

1. **Muscle contribution below 30 Hz.** Track 3 has no muscle (EMG) channel. A frequency-band ablation
   (e.g. 1–8, 8–13, 13–30 Hz separately) will show where the information sits.
2. **Phrase length.** "Help me" and "Thank you" were recognised best (0.48 each on validation), "Stop"
   worst (0.33). The re-analysis found phrases with different syllable counts separate more easily, so the
   model may partly detect speech duration or rhythm rather than the words themselves.
3. **Repeated cues.** Each cue was imagined four times in a row and the public files do not say which
   trials share a cue. If the organisers' split keeps repetitions of one cue on both sides, test accuracy
   may be optimistic. We cannot check this from the public data.
4. **Cross-subject (LOSO)** has not been run yet; the re-analyses report chance level.
