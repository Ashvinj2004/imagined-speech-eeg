from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bci.eval.harness import evaluate
from bci.eval.report import append_log_row, plot_confusion, plot_per_subject
from bci.eval.splits import predefined_val
from bci.models import majority


@pytest.fixture
def split_ds(small_ds):
    rng = np.random.default_rng(0)
    split = rng.choice(["train", "val", "test"], size=small_ds.n_trials, p=[0.6, 0.2, 0.2])
    return replace(small_ds, split=split)


def test_predefined_val_never_touches_test(split_ds):
    folds = list(predefined_val(split_ds))
    assert len(folds) == len(np.unique(split_ds.subject))
    for f in folds:
        assert set(split_ds.split[f.train_idx]) == {"train"}
        assert set(split_ds.split[f.test_idx]) == {"val"}
        assert set(split_ds.subject[f.train_idx]) == set(split_ds.subject[f.test_idx]) == {f.subject}


def test_evaluate_returns_one_prediction_per_test_trial(split_ds):
    df, preds = evaluate(majority, split_ds, ["predefined_val"], return_predictions=True)
    assert len(preds) == df["n_test"].sum()
    assert set(preds.columns) == {"protocol", "subject", "trial", "y_true", "y_pred"}
    assert np.array_equal(preds["y_true"].to_numpy(), split_ds.y[preds["trial"].to_numpy()])
    assert isinstance(evaluate(majority, split_ds, ["predefined_val"]), pd.DataFrame)  # default unchanged


def test_runner_refuses_test_split_without_final(capsys):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_experiment", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--dataset", "synthetic", "--model", "majority", "--protocol", "predefined"]) == 2
    assert "refusing" in capsys.readouterr().err


def test_log_row_format(tmp_path):
    log = tmp_path / "LOG.md"
    row = append_log_row("track3", "riemann | C=1", "predefined_val", 0, 750, 0.4123, 0.2240, path=log, commit="abc1234")
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("| date | commit | dataset")
    assert lines[-1] == row
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:] == ["abc1234", "track3", "riemann / C=1", "predefined_val", "0", "750", "0.412", "0.224"]
    append_log_row("track3", "x", "loso", 1, 10, 0.5, 0.6, path=log, commit="abc1234")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 4  # header, separator, two rows


def test_figures_are_written(tmp_path, split_ds):
    _, preds = evaluate(majority, split_ds, ["predefined_val"], return_predictions=True)
    cm = plot_confusion(preds, split_ds.class_names, "t", tmp_path / "cm.png")
    assert cm.sum() == len(preds) and (tmp_path / "cm.png").stat().st_size > 0
    tab = plot_per_subject(preds, split_ds.n_classes, "t", tmp_path / "ps.png")
    assert len(tab) == len(np.unique(split_ds.subject)) and (tmp_path / "ps.png").exists()


def test_deep_wrapper_is_deterministic_and_maps_labels():
    pytest.importorskip("braindecode")
    from bci.models.deep import BraindecodeClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 8, 128)).astype(np.float32)
    y = np.repeat([3, 7], 20)  # labels need not be 0..k-1
    a = BraindecodeClassifier(arch="shallow", max_epochs=2, patience=2, seed=1).fit(X, y)
    b = BraindecodeClassifier(arch="shallow", max_epochs=2, patience=2, seed=1).fit(X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert set(a.predict(X)) <= {3, 7}
    assert a.mean_.shape == (8, 1)  # per-channel scaling, learned inside fit
