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

## Sprint 3: all three protocols, both datasets

### Track 3, cross-subject (LOSO)

Riemannian (0.5–30 Hz), train on 14 people, test on the 15th: **0.199** pooled over 6,000 trials
(chance threshold 0.208, binomial p = 0.59). Per-subject range 0.172–0.228; 0/15 above their own
threshold (0.233 for n = 400). The same model and data gave 0.404 within-subject.

No shuffle test was run here: the result is at chance, so shuffled labels would score the same;
the binomial p already answers the question.

### Nieto, inner speech (4 words, n = 2,236)

| Model | Within-subject (5-fold) | Cross-session | LOSO |
|---|---|---|---|
| Majority class | 0.249 | 0.250 | 0.250 |
| Riemannian (0.5–40 Hz) | 0.224 | 0.263 | 0.247 |
| ShallowConvNet (0.5–40 Hz, 128 Hz) | 0.235 | – | – |

Chance threshold 0.265 for n = 2,236 (1/k = 0.250). Nothing is above it. The within-subject value
below 1/k is noise, not an artefact: predictions are spread evenly over the four classes
(587/529/558/562), per-person accuracies run 0.19–0.31, and single folds (≈45 trials) swing from
0.06 to 0.44. Cross-session, the only such measurement in the project, is also at chance.

### What we conclude from Sprint 3

- **The Track 3 effect is person-specific.** The 0.404 → 0.199 drop is the central finding: covariance
  features carry no information that transfers to a new person with this method.
- **The effect does not reappear on a second dataset.** On Nieto's inner-speech trials the same model is
  at chance in every protocol, consistent with the dataset authors' own report and with the 2026
  re-analysis. Possible reasons, not yet separated: a harder task (four short direction words vs five
  phrases of different length), a different montage (128 channels, BioSemi), the authors' broader
  preprocessing, or simply no usable signal in this kind of covariance feature.
- Together with the phrase-length pattern on Track 3, this strengthens the suspicion that part of the
  Track 3 within-subject accuracy reflects phrase duration/rhythm rather than word identity.

## Sprint 4: ablations

### Frequency bands (Riemannian, one band at a time)

| Band | Track 3 train → val (n=750, thr 0.224) | Track 3 LOSO (n=6000, thr 0.208) | Nieto within (thr 0.265) | Nieto cross-session | Nieto LOSO |
|---|---|---|---|---|---|
| 1–8 Hz | 0.311 | 0.208 | 0.233 | 0.246 | 0.268 |
| 8–13 Hz | 0.291 | 0.197 | 0.251 | 0.275 | 0.230 |
| 13–30 Hz | 0.403 | 0.201 | 0.234 | 0.270 | 0.259 |
| 30–40 Hz | **0.443** | 0.198 | 0.209 | 0.264 | 0.256 |
| 0.5–30 Hz (reference) | 0.415 | 0.199 | | | |
| 0.5–40 Hz (reference) | 0.405 | | 0.224 | 0.263 | 0.247 |

**What this shows.**
- On Track 3, accuracy *rises* with frequency: the 30–40 Hz band alone (0.443) beats the full 0.5–40 Hz
  band (0.405) and the 0.5–30 Hz band (0.415). 13–30 Hz alone recovers almost everything (0.403);
  the slow bands carry less (0.31, 0.29).
- This is the signature the plan warned about. Cortical speech-related activity is usually reported in
  the lower bands, while 30–40 Hz is where facial and jaw muscle activity (EMG) is strongest and Track 3
  has no muscle channel to rule it out. The 2026 re-analysis saw the same rising profile and drew the
  same conclusion. **We therefore do not read the Track 3 within-subject accuracy as evidence of
  decoded imagined-speech content.** It is evidence of a reliable, person-specific signal that is
  time-locked to the task and strongest in the muscle-exposed band.
- Every band is at chance across people (LOSO), so whatever the signal is, it is not shared across people.
- On Nieto, no band reaches the threshold in any protocol. The two cells just above it (8–13 Hz
  cross-session 0.275, 1–8 Hz LOSO 0.268) are 2 of 12 cells within 0.01 of a 5% threshold, i.e. what
  chance produces; without a shuffle test we treat them as noise.

### Euclidean Alignment

Per-subject covariance whitening, fitted on the training fold; for an unseen participant the
reference comes from their own unlabelled trials (unsupervised calibration).

| Dataset | Protocol | Without EA | With EA | Threshold |
|---|---|---|---|---|
| Track 3 | train → val | 0.415 | 0.415 | 0.224 |
| Track 3 | LOSO | 0.199 | 0.204 | 0.208 |
| Nieto | within-subject | 0.224 | 0.224 | 0.265 |
| Nieto | cross-session | 0.263 | 0.263 | 0.265 |
| Nieto | LOSO | 0.247 | 0.237 | 0.265 |

**EA does not help here.** Within subject it changes nothing, which is expected: whitening by a
subject's own mean is close to a no-op when every trial comes from that same subject. Cross-subject
it moves Track 3 from 0.199 to 0.204, still below the 0.208 threshold, and moves Nieto's LOSO down.
The published claim that EA buys several points of cross-subject accuracy does not reproduce on
imagined speech; it was developed on motor imagery, where a shared cross-subject pattern exists to
align. Here there is no shared signal to align, so aligning the coordinate frames cannot create one.

### Cropped training

1 s windows every 0.5 s (5 crops per 3.1 s Track 3 trial, 4 per 2.5 s Nieto trial). All crops of a
trial stay on one side of every split, and at test time a trial's crops are combined by majority
vote and counted once, so `n_test` is unchanged (750 / 6,000 / 2,236).

| Dataset | Protocol | Baseline | +Crops | Threshold |
|---|---|---|---|---|
| Track 3 | train → val | 0.415 | **0.387** | 0.224 |
| Track 3 | LOSO | 0.199 | 0.200 | 0.208 |
| Nieto | within-subject | 0.224 | 0.239 | 0.265 |
| Nieto | cross-session | 0.263 | 0.271 | 0.265 |
| Nieto | LOSO | 0.247 | 0.242 | 0.265 |

**Cropping costs accuracy on Track 3 and changes nothing on Nieto.** Five times the training
examples did not compensate for shortening each one from 3.1 s to 1 s: within-subject accuracy fell
by 2.8 points, and cross-subject stayed at chance.

This fits the rest of the evidence. If the Track 3 effect tracks the *duration or rhythm* of an
articulatory act, as the phrase-length pattern and the 30–40 Hz profile both suggest, then a 1 s
window is the wrong unit: it cuts up exactly the longer-timescale structure the classifier was
using. A feature that genuinely distinguished word identity moment to moment would survive cropping
better than this.

Cost note: the Track 3 crops run (30,000 windows, LOSO over 15 folds) took ~10 h on one laptop CPU,
by far the most expensive run in the project, for a result that reduced accuracy.

### Open questions

1. **Muscle contribution.** Answered as far as public data allows: the information concentrates in
   30–40 Hz, the most EMG-exposed band, and Track 3 has no EMG channel. Only a dataset with a muscle
   channel (e.g. own recordings with EMG) can settle it.
2. **Phrase length.** "Help me" and "Thank you" were recognised best (0.48 each on validation), "Stop"
   worst (0.33). The re-analysis found phrases with different syllable counts separate more easily, so the
   model may partly detect speech duration or rhythm rather than the words themselves.
3. **Repeated cues.** Each cue was imagined four times in a row and the public files do not say which
   trials share a cue. If the organisers' split keeps repetitions of one cue on both sides, test accuracy
   may be optimistic. We cannot check this from the public data.
4. **Cross-subject (LOSO)** has not been run yet; the re-analyses report chance level.
