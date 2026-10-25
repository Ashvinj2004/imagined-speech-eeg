# Experiment log

One row per experiment run on real data (append only; synthetic runs are never logged).
`chance threshold` is the Combrisson & Jerbi binomial threshold (alpha = 0.05) for that `n_test`.
Any acc > 0.90 on 3+ classes must not be logged until leakage has been ruled out.

| date | commit | dataset | config | protocol | seed | n_test | acc | chance threshold |
|------|--------|---------|--------|----------|------|--------|-----|------------------|
