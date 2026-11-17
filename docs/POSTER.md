# Poster outline (A0, landscape, three columns)

Numbers come from `results/tables/main.md` and `results/tables/ablations.md`; regenerate before printing.

## Title band
**Can we decode imagined speech from scalp EEG? A leakage-controlled answer on two public datasets**
Team names · course · date · repository QR code

## Column 1: the question and the trap
- **What we did:** classify 5 imagined phrases (Track 3) and 4 imagined words (Nieto) from EEG.
- **Why most published numbers can't be trusted:** three leakage routes (same trial on both sides,
  wrong protocol for the claim, preprocessing fitted on all data) and comparison against 1/k.
  One diagram: a trial cut into crops, with crops landing on both sides of a split (crossed out).
- **Our rules:** every split checked by code; chance = binomial threshold for the real test size;
  shuffle test for headline numbers; three protocols reported separately; every run logged.

## Column 2: results
- **Figure 1 (big):** bar chart, Track 3, per participant, official test split: Riemannian accuracy
  with each participant's own chance line (`per_subject_predefined.png`).
- **Table 1:** Track 3 official split: majority 0.200, EEGNet 0.257, ShallowConvNet 0.285,
  Riemannian 0.404 (threshold 0.224; shuffle p = 0.0099).
- **Figure 2:** the drop: within-subject 0.404 -> LOSO 0.199 on the same data (two bars, chance band).
- **Table 2:** Nieto within-subject / cross-session / LOSO (TODO from log).
- **Figure 3:** confusion matrix, Track 3 (`confusion_predefined.png`): "Help me"/"Thank you" best, "Stop" worst.

## Column 3: what it means
- **Finding 1:** a real, person-specific effect: ~40% on five phrases, far above the shuffle null.
- **Finding 2:** it does not transfer across people: chance under LOSO, with or without alignment (TODO).
- **Finding 3:** where the signal lives: frequency-band ablation (TODO), 30-40 Hz not needed.
- **Caveats, stated plainly:** phrase length may drive part of the effect; no muscle channel in
  Track 3; repeated cues not identifiable in the public files.
- **Why the 82.6% competition score is not a target:** not reproduced by us or by two re-analyses.
- **Reproduce it:** `python scripts/reproduce.py`; every number traces to a logged commit.

## Footer
Data: NEMAR nm000113 (CC-BY-4.0), OpenNeuro ds003626 (CC0). Code: GitHub link. No own recordings.
