# Dataset cards

One card per dataset, in the style of a datasheet: exactly what we used, from where, under which
licence, and what we changed. Operational traps are in `docs/DATASETS.md`.

## BCI Competition 2020, Track 3: imagined speech

| Field | Value |
|---|---|
| Name | 2020 International BCI Competition, Track #3 "Imagined speech classification" |
| Creators | Korea University (competition co-chairs S.-W. Lee, K.-R. Müller, J. del R. Millán) |
| Reference | Jeong et al. (2022), "2020 International brain–computer interface competition: A review", *Frontiers in Human Neuroscience* 16:898300 |
| Original distribution | OSF project pq7vb (Track 3 folder now holds only a readme) |
| Copy we use | NEMAR nm000113, DOI 10.82901/nemar.nm000113, BIDS/EDF conversion by P. Guetschel; licence as stated on NEMAR (CC-BY-4.0 at time of download) |
| Downloaded | 2026-10-22 via `scripts/download_data.py track3` (nemar-py), 586 MB |
| Participants | 15, aged 20–30 per the review; no other demographics released |
| Task | Five imagined phrases: Hello, Help me, Stop, Thank you, Yes. Auditory cue, then imagery; four imagery repetitions per cue |
| Recording | 64 EEG channels, 10-20 layout, BrainAmp, reference FCz, ground Fpz, 256 Hz |
| Trials | 70 per class per participant: 60 train / 10 validation / 10 test (official split, test labels from the organisers' answer sheet) |
| Sessions | One per participant |
| Epoch | 795 samples = 3.105 s, as distributed |
| Our processing | Epochs cut back out of the concatenated EDF by annotation; Volts → µV; average reference; 60 Hz notch; 0.5–30 Hz band-pass (0.5–40 Hz for neural nets, resampled to 128 Hz); all per trial |
| Known issues | Repetitions of one cue are not identifiable, so our own cross-validation on this set may be optimistic; the official split is the reference. No muscle channel was recorded |
| Used for | Primary target: within-subject (official split) and LOSO results, band and alignment ablations |

## Nieto et al. 2022, "Thinking out loud": inner speech

| Field | Value |
|---|---|
| Name | Inner Speech Dataset (OpenNeuro ds003626) |
| Creators | N. Nieto, V. Peterson, H. L. Rufiner, J. E. Kamienkowski, R. Spies (sinc(i), Argentina) |
| Reference | Nieto et al. (2022), "Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition", *Scientific Data* 9:52 |
| Licence | CC0 (OpenNeuro default), see dataset_description.json |
| Downloaded | 2026-10-24 via `scripts/download_data.py nieto`, authors' derivatives only (6.3 GB) |
| Participants | 10 |
| Task | Four Spanish direction words (Arriba/Up, Abajo/Down, Derecha/Right, Izquierda/Left) under three conditions: pronounced, inner (imagined), visualized |
| Recording | BioSemi ActiveTwo, 128 EEG + 8 EXG channels, 1024 Hz |
| Sessions | 3 per participant ("blocks"), separate days |
| Trials used | 2,236 inner-speech trials (559 per class; 180–240 per participant); pronounced and visualized trials excluded |
| Epoch | Authors' epochs: −0.5 to 4 s around the cue, decimated to 256 Hz; we keep the 1.0–3.5 s window (the authors' recommended action interval) |
| Authors' processing (kept) | Re-referenced, 50 Hz notch, 0.5–100 Hz band-pass, ICA not applied |
| Our processing | Additional 0.5–40 Hz band-pass per trial (0.5–30 Hz in some ablations); no re-reference (already done) |
| Known issues | Labels ship as pickle files; we read them with an allow-list unpickler and cross-check against the EEG trigger codes. Some third sessions are half-length (sub-01, 03, 06, 08) |
| Used for | Second dataset: within-subject, cross-session and LOSO results; the only cross-session measurement in the project |

## What we did not use

- The Track 3 training/validation `.mat` files in their original form (not publicly available any more).
- Nieto's raw BDF recordings and EXG channels (we start from the authors' derivatives).
- Nieto's pronounced and visualized conditions (they contain overt muscle activity or a different task).
