"""Exercise 1: turn predicted margins into calibrated win probabilities.

See docs/exercises/01-calibration.md for the concepts and hints.
Spec: tests/test_calibration.py
"""
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class MarginCalibrator:
    """Maps a predicted home margin (points) to P(home win).

    method: "platt" (logistic regression on the margin) or "isotonic".
    """

    method: str = "platt"
    model: object | None = None

    def fit(self, margin: np.ndarray, home_win: np.ndarray) -> "MarginCalibrator":
        # TODO(you): fit self.model on 1-D margins -> binary outcomes.
        #   - "platt": sklearn LogisticRegression on margin as the single feature.
        #     Think about what the fitted coefficient means compared with 1/13.5.
        #   - "isotonic": sklearn IsotonicRegression. What should happen to margins
        #     outside the range seen in training?
        # Return self so calls can be chained.
        raise NotImplementedError

    def predict_proba(self, margin: np.ndarray) -> np.ndarray:
        # TODO(you): return P(home win) for each margin, clipped away from exactly 0 and 1.
        #   Why clip? (Hint: log loss.)
        raise NotImplementedError


def reliability_table(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Bin predictions into equal-width probability bins on [0, 1].

    Returns one row per non-empty bin with columns:
      bin (int, 0-based), mean_pred (float), observed_rate (float), count (int)
    sorted by bin.
    """
    # TODO(you): this is the data behind a reliability diagram.
    #   Watch the edge case: which bin does prob == 1.0 land in?
    raise NotImplementedError


def expected_calibration_error(prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10) -> float:
    """ECE: the count-weighted mean of |mean_pred - observed_rate| across bins."""
    # TODO(you): build on reliability_table.
    raise NotImplementedError


def walk_forward_calibrated(
    preds: pl.DataFrame, model_name: str, method: str = "platt", min_prior_seasons: int = 1
) -> pl.DataFrame:
    """Add a `prob_{model_name}` column of calibrated P(home win) to walk-forward predictions.

    `preds` has one row per game with columns season, home_win, and pred_{model_name}. These are
    out-of-sample margins from evaluate.walk_forward. For each test season S, fit the calibrator
    ONLY on rows from seasons < S, then apply it to season S. Seasons with fewer than
    `min_prior_seasons` prior seasons available get a null probability.
    """
    # TODO(you): this is where leakage sneaks in. Read the "Nested out-of-sample" section
    #   of the exercise doc before writing it.
    raise NotImplementedError
