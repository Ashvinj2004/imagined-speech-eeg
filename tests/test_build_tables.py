import importlib.util
from pathlib import Path

HEADER = "| date | commit | dataset | config | protocol | seed | n_test | acc | chance threshold |\n|---|---|---|---|---|---|---|---|---|\n"


def _mod():
    spec = importlib.util.spec_from_file_location("build_tables", Path(__file__).resolve().parents[1] / "scripts" / "build_tables.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_latest_row_wins_and_perm_shown(tmp_path):
    m = _mod()
    log = tmp_path / "LOG.md"
    log.write_text(HEADER
                   + "| 2026-11-01 | aaa | ds | riemann; bp=0.5-40.0 | predefined | 0 | 750 | 0.300 | 0.224 |\n"
                   + "| 2026-11-02 | bbb | ds | riemann; bp=0.5-30.0 perm_p=0.0099 (n=100) | predefined | 0 | 750 | 0.404 | 0.224 |\n"
                   + "| 2026-11-02 | bbb | ds | majority; bp=0.5-40.0 | predefined | 0 | 750 | 0.200 | 0.224 |\n"
                   + "| 2026-11-03 | ccc | ds | riemann; bp=8.0-13.0 variant=band | loso | 0 | 6000 | 0.210 | 0.208 |\n",
                   encoding="utf-8")
    df = m.read_log(log)
    assert len(df) == 4 and df["variant"].tolist() == ["", "", "", "band"]
    table = m.main_table(df)
    assert "0.404 (p=0.010, 100 shuffles)" in table and "0.300" not in table
    assert "| Majority class | 0.200 * |" in table  # below threshold flagged
    abl = m.ablation_table(df)
    assert "band" in abl and "8.0-13.0 Hz" in abl and "0.210" in abl
