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
| 2026-11-17 | 45d1f96 | bci2020_track3 | riemann; notch=60.0 bp=0.5-30.0 reref=average fs=256 | loso | 0 | 6000 | 0.199 | 0.208 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 | within_subject | 0 | 2236 | 0.224 | 0.265 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 | cross_session | 0 | 2236 | 0.263 | 0.265 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 | loso | 0 | 2236 | 0.247 | 0.265 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | majority; notch=None bp=0.5-40.0 reref=None fs=256 | within_subject | 0 | 2236 | 0.249 | 0.265 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | majority; notch=None bp=0.5-40.0 reref=None fs=256 | cross_session | 0 | 2236 | 0.250 | 0.265 |
| 2026-11-17 | 45d1f96 | nieto2022-inner | majority; notch=None bp=0.5-40.0 reref=None fs=256 | loso | 0 | 2236 | 0.250 | 0.265 |
| 2026-11-17 | 398c18b | nieto2022-inner | shallow; notch=None bp=0.5-40.0 reref=None fs=128 | within_subject | 0 | 2236 | 0.235 | 0.265 |
| 2026-11-17 | 398c18b | bci2020_track3 | riemann; notch=60.0 bp=1.0-8.0 reref=average fs=256 variant=band1-8 | predefined_val | 0 | 750 | 0.311 | 0.224 |
| 2026-11-17 | 398c18b | bci2020_track3 | riemann; notch=60.0 bp=1.0-8.0 reref=average fs=256 variant=band1-8 | loso | 0 | 6000 | 0.208 | 0.208 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=1.0-8.0 reref=None fs=256 variant=band1-8 | within_subject | 0 | 2236 | 0.233 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=1.0-8.0 reref=None fs=256 variant=band1-8 | cross_session | 0 | 2236 | 0.246 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=1.0-8.0 reref=None fs=256 variant=band1-8 | loso | 0 | 2236 | 0.268 | 0.265 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=8.0-13.0 reref=average fs=256 variant=band8-13 | predefined_val | 0 | 750 | 0.291 | 0.224 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=8.0-13.0 reref=average fs=256 variant=band8-13 | loso | 0 | 6000 | 0.197 | 0.208 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=8.0-13.0 reref=None fs=256 variant=band8-13 | within_subject | 0 | 2236 | 0.251 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=8.0-13.0 reref=None fs=256 variant=band8-13 | cross_session | 0 | 2236 | 0.275 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=8.0-13.0 reref=None fs=256 variant=band8-13 | loso | 0 | 2236 | 0.230 | 0.265 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=13.0-30.0 reref=average fs=256 variant=band13-30 | predefined_val | 0 | 750 | 0.403 | 0.224 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=13.0-30.0 reref=average fs=256 variant=band13-30 | loso | 0 | 6000 | 0.201 | 0.208 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=13.0-30.0 reref=None fs=256 variant=band13-30 | within_subject | 0 | 2236 | 0.234 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=13.0-30.0 reref=None fs=256 variant=band13-30 | cross_session | 0 | 2236 | 0.270 | 0.265 |
| 2026-11-17 | 80db923 | nieto2022-inner | riemann; notch=None bp=13.0-30.0 reref=None fs=256 variant=band13-30 | loso | 0 | 2236 | 0.259 | 0.265 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=30.0-40.0 reref=average fs=256 variant=band30-40 | predefined_val | 0 | 750 | 0.443 | 0.224 |
| 2026-11-17 | 80db923 | bci2020_track3 | riemann; notch=60.0 bp=30.0-40.0 reref=average fs=256 variant=band30-40 | loso | 0 | 6000 | 0.198 | 0.208 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=30.0-40.0 reref=None fs=256 variant=band30-40 | within_subject | 0 | 2236 | 0.209 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=30.0-40.0 reref=None fs=256 variant=band30-40 | cross_session | 0 | 2236 | 0.264 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=30.0-40.0 reref=None fs=256 variant=band30-40 | loso | 0 | 2236 | 0.256 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann_ea; notch=None bp=0.5-40.0 reref=None fs=256 variant=ea | within_subject | 0 | 2236 | 0.224 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann_ea; notch=None bp=0.5-40.0 reref=None fs=256 variant=ea | cross_session | 0 | 2236 | 0.263 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann_ea; notch=None bp=0.5-40.0 reref=None fs=256 variant=ea | loso | 0 | 2236 | 0.237 | 0.265 |
| 2026-11-17 | b2311d0 | bci2020_track3 | riemann_ea; notch=60.0 bp=0.5-30.0 reref=average fs=256 variant=ea | predefined_val | 0 | 750 | 0.415 | 0.224 |
| 2026-11-17 | b2311d0 | bci2020_track3 | riemann_ea; notch=60.0 bp=0.5-30.0 reref=average fs=256 variant=ea | loso | 0 | 6000 | 0.204 | 0.208 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 crop=1/0.5s variant=crops | within_subject | 0 | 2236 | 0.239 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 crop=1/0.5s variant=crops | cross_session | 0 | 2236 | 0.271 | 0.265 |
| 2026-11-17 | b2311d0 | nieto2022-inner | riemann; notch=None bp=0.5-40.0 reref=None fs=256 crop=1/0.5s variant=crops | loso | 0 | 2236 | 0.242 | 0.265 |
| 2026-11-18 | 3baedb3 | bci2020_track3 | riemann; notch=60.0 bp=0.5-30.0 reref=average fs=256 crop=1/0.5s variant=crops | predefined_val | 0 | 750 | 0.387 | 0.224 |
| 2026-11-18 | 3baedb3 | bci2020_track3 | riemann; notch=60.0 bp=0.5-30.0 reref=average fs=256 crop=1/0.5s variant=crops | loso | 0 | 6000 | 0.200 | 0.208 |
