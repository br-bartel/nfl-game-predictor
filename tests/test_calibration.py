import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from nflpred.calibration import (
    MarginCalibrator,
    expected_calibration_error,
    reliability_table,
    walk_forward_calibrated,
)

RNG = np.random.default_rng(7)


def simulate(n=20_000, sd=13.5, scale=1.0):
    """Margins whose true win probability is Φ(scale * margin / sd)."""
    margin = RNG.normal(0, 7, n)
    p_true = norm.cdf(scale * margin / sd)
    return margin, p_true, (RNG.random(n) < p_true).astype(int)


# --- reliability_table / ECE -------------------------------------------------

def test_reliability_table_shape_and_counts():
    prob = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    outcome = np.array([0, 1, 0, 1, 1])
    t = reliability_table(prob, outcome, n_bins=10)
    assert t.columns == ["bin", "mean_pred", "observed_rate", "count"]
    assert t["bin"].to_list() == [0, 1, 9]  # 1.0 belongs in the last bin, not an 11th
    assert t["count"].to_list() == [1, 2, 2]
    assert t.filter(pl.col("bin") == 1)["observed_rate"].item() == pytest.approx(0.5)


def test_ece_small_when_calibrated_large_when_overconfident():
    _, p_true, y = simulate()
    assert expected_calibration_error(p_true, y) < 0.02
    overconfident = np.where(p_true > 0.5, 0.97, 0.03)
    assert expected_calibration_error(overconfident, y) > 0.15


# --- MarginCalibrator -------------------------------------------------------

@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_recovers_true_probabilities(method):
    # The truth is twice as steep as Φ(m/13.5), so the naive conversion is miscalibrated.
    margin, p_true, y = simulate(scale=2.0)
    cal = MarginCalibrator(method=method).fit(margin, y)
    p = cal.predict_proba(margin)
    assert np.mean(np.abs(p - p_true)) < 0.03
    assert expected_calibration_error(p, y) < expected_calibration_error(norm.cdf(margin / 13.5), y)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_is_monotonic_bounded_and_extrapolates(method):
    margin, _, y = simulate()
    cal = MarginCalibrator(method=method).fit(margin, y)
    grid = np.linspace(-60, 60, 241)  # wider than the training range
    p = cal.predict_proba(grid)
    assert np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) >= 0)


# --- walk_forward_calibrated ------------------------------------------------

def fake_preds():
    rows = []
    for season in [2019, 2020, 2021, 2022]:
        margin, _, y = simulate(n=500)
        rows.append(pl.DataFrame({"season": season, "pred_ridge": margin, "home_win": y}))
    return pl.concat(rows)


def test_walk_forward_adds_column_and_skips_first_season():
    out = walk_forward_calibrated(fake_preds(), "ridge")
    assert "prob_ridge" in out.columns
    assert out.filter(pl.col("season") == 2019)["prob_ridge"].null_count() == 500
    assert out.filter(pl.col("season") > 2019)["prob_ridge"].null_count() == 0


def test_walk_forward_never_sees_the_test_season():
    preds = fake_preds()
    base = walk_forward_calibrated(preds, "ridge")
    # Flip every 2022 outcome. A leak-free calibrator gives identical 2022 probabilities.
    flipped = preds.with_columns(
        home_win=pl.when(pl.col("season") == 2022).then(1 - pl.col("home_win"))
        .otherwise(pl.col("home_win"))
    )
    after = walk_forward_calibrated(flipped, "ridge")
    a = base.filter(pl.col("season") == 2022)["prob_ridge"].to_numpy()
    b = after.filter(pl.col("season") == 2022)["prob_ridge"].to_numpy()
    np.testing.assert_allclose(a, b)
