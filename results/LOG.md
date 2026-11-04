# Experiment log

One row per experiment run on real data (append only; synthetic runs are never logged).
`chance threshold` is the Combrisson & Jerbi binomial threshold (alpha = 0.05) for that `n_test`.
Any acc > 0.90 on 3+ classes must not be logged until leakage has been ruled out.

| date | commit | dataset | config | protocol | seed | n_test | acc | chance threshold |
|------|--------|---------|--------|----------|------|--------|-----|------------------|
| 2026-11-04 | f5a84ca | bci2020_track3 | majority; notch=60.0 bp=0.5-40.0 reref=average fs=256 | predefined_val | 0 | 750 | 0.200 | 0.224 |
| 2026-11-04 | f5a84ca | bci2020_track3 | riemann; notch=60.0 bp=0.5-40.0 reref=average fs=256 | predefined_val | 0 | 750 | 0.405 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | riemann; notch=60.0 bp=0.5-30.0 reref=average fs=256 | predefined_val | 0 | 750 | 0.415 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | eegnet; notch=60.0 bp=0.5-40.0 reref=average fs=128 | predefined_val | 0 | 750 | 0.233 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | shallow; notch=60.0 bp=0.5-40.0 reref=average fs=128 | predefined_val | 0 | 750 | 0.240 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | majority; notch=60.0 bp=0.5-40.0 reref=average fs=256 | predefined | 0 | 750 | 0.200 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | shallow; notch=60.0 bp=0.5-40.0 reref=average fs=128 | predefined | 0 | 750 | 0.285 | 0.224 |
| 2026-11-04 | 72d7ae2 | bci2020_track3 | eegnet; notch=60.0 bp=0.5-40.0 reref=average fs=128 | predefined | 0 | 750 | 0.257 | 0.224 |
| 2026-11-05 | 72d7ae2 | bci2020_track3 | riemann; notch=60.0 bp=0.5-30.0 reref=average fs=256 perm_p=0.0099 (n=100) | predefined | 0 | 750 | 0.404 | 0.224 |
