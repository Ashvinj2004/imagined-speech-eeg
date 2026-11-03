# Datasets

Download both with `python scripts/download_data.py` (files land in `data/`, which is gitignored).
Load them with `bci.data.load("track3")` / `bci.data.load("nieto")`.

**Decision (2026-10-25):** the primary task is Track 3's five phrases. Nieto inner speech is the
second dataset, used to check that results hold elsewhere and for cross-session evaluation.

## BCI Competition 2020 Track 3 (`track3`), primary target and reproduction anchor

| | |
|---|---|
| Task | Imagined speech, 5 classes: Hello, Help me, Stop, Thank you, Yes |
| Subjects / sessions | 15 / 1 recording session each (so **no cross-session protocol**) |
| Channels, rate | 64 EEG channels, 256 Hz |
| Splits (per subject) | train 60 trials/class, validation 10/class, test 10/class (= BIDS run-00 / run-01 / run-02) |
| Source | Organisers' OSF project https://osf.io/pq7vb/. Its Track 3 folder now holds only a readme, so we use the BIDS/EDF conversion on NEMAR, [nm000113](https://nemar.org/dataset/nm000113) (converted by P. Guetschel; test labels from the official answer sheet) |
| Paper | Jeong et al. (2022), "2020 International brain–computer interface competition: A review", *Front. Hum. Neurosci.* 16:898300 |

**Paradigm** (from the Jeong et al. 2022 review)
- Per cue: an auditory cue of the phrase (2 s), then a fixation cross (0.8–1.2 s); imagery starts when the
  cross disappears. Each cue is followed by **four** cross-then-imagery phases of about 2 s, then 3 s of rest.
- 70 trials per class were recorded (60 train, 10 validation, 10 test). Recorded with a BrainAmp amplifier,
  64 electrodes in the 10-20 layout, ground at Fpz, reference at FCz, impedance below 15 kΩ.
- Each epoch in the public data is 795 samples (3.11 s at 256 Hz).
- Published competition results: best 82.6%, second 75.5% (methods not named in the review).
- Mains frequency: **60 Hz** (recorded in Korea). Confirmed by a clear 60 Hz spike in the raw spectrum
  (`scripts/plot_psd.py`), which our default preprocessing removes.

**Gotchas**
- **Repeated cues may be correlated.** The four imagery phases after one cue are probably separate trials,
  and repetitions of the same cue are likely more alike than independent trials. The public files do not
  mark which trials share a cue, and the stored order does not reveal it (labels look shuffled for some
  subjects, e.g. sub-01 and sub-05, and come in long same-label runs for others, e.g. sub-12). So our own
  within-subject cross-validation on this dataset may be **optimistic**; the organisers' predefined split
  is the reference result. Treat within-subject CV numbers on Track 3 as an upper bound.
- The test split is tiny: 50 trials per subject. The chance threshold for n=50 and k=5 is 30%, not 20%, and per-subject numbers are noisy. Report pooled numbers alongside the per-subject spread.
- The test labels came out after the competition. Never tune on `split == "test"`: tune on train/val, then touch test once per final configuration.
- The NEMAR copy concatenated the original epochs into continuous EDF files; the loader cuts them back out using the annotation onsets and durations. Units were converted to Volts by the converter; the loader returns microvolts.
- One session means the "cross-session" row of our tables is not available for this dataset. Within-subject (predefined split) and LOSO are.

## Nieto et al. 2022, "Thinking out loud" (`nieto`), second dataset

| | |
|---|---|
| Task | 4 directions in Spanish: Arriba/Up, Abajo/Down, Derecha/Right, Izquierda/Left |
| Conditions | pronounced (spoken aloud), **inner** (imagined, our target), visualized |
| Subjects / sessions | 10 / 3 sessions ("blocks") each, so the cross-session protocol is possible |
| Channels, rate | 128 EEG channels (BioSemi 128 montage) + 8 EXG, recorded at 1024 Hz; authors' epochs are decimated ×4 to 256 Hz |
| Epochs | −0.5 s to 4 s around the cue; the authors recommend the 1.0–3.5 s window (the "action interval") |
| Trials | 5,640 in total; **2,236 inner-speech trials** (559 per class), 180–240 per subject. Sessions are uneven: some third sessions are half-length (sub-01, sub-03, sub-06, sub-08), and sub-03's sessions 1 and 3 have an unusual condition mix |
| Source | OpenNeuro [ds003626](https://openneuro.org/datasets/ds003626); code: https://github.com/N-Nieto/Inner_Speech_Dataset |
| Paper | Nieto et al. (2022), *Scientific Data* 9:52 |

**Gotchas**
- The authors' derivatives are already re-referenced, notch-filtered at 50 Hz and band-passed 0.5–100 Hz. Those are fixed filters, which our rules allow before splitting.
- Per-trial condition labels (pronounced / inner / visualized) live in `*_events.dat`, not in the `.fif` files. The authors' own code reads that file with pickle; we do **not** unpickle downloaded files (pickle can execute code). See `bci/data/nieto.py` for how the loader reads it.
- Always filter to the inner-speech condition for headline results. Pronounced trials contain muscle activity and would inflate accuracy.
