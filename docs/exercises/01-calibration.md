# Exercise 1: Calibrated Win Probability

**Skills:** probabilistic classification, calibration, proper scoring rules, and nested
out-of-sample fitting.
**Files:** `src/nflpred/calibration.py` (yours) and `tests/test_calibration.py` (the spec).
**Done when:** `pytest tests/test_calibration.py` passes, you've written up Part C, and you can
answer the interview check without notes.

## Why this matters

The models predict a **margin**, but most consumers of a sports model (and of most business
models: churn, fraud, conversion) need a **probability** they can act on. A probability is only
useful if it's *calibrated*: of all the games where you say 70%, the home team should win about
70%.

Right now `evaluate.py` converts margin to probability with `Φ(margin / 13.5)`. That's a
hand-picked assumption. Your job is to learn the mapping from data, measure whether it helps,
and do it without leaking the future.

## Concepts (read before coding)

**Proper scoring rules.** Brier score (mean squared error of the probability) and log loss reward
honest probabilities. Accuracy doesn't, because a model that says 51% and one that says 99% get
the same credit for a correct pick. Brier score decomposes into *calibration* and *refinement*
(sharpness): you want probabilities that are both honest and decisive.

**Reliability diagram and ECE.** Bin the predictions, then plot the mean predicted probability
against the observed win rate in each bin. A perfect model lies on the diagonal. Expected
calibration error (ECE) is the count-weighted average distance from the diagonal. It's handy but
crude: it depends on the bin count and ignores sharpness. Never report ECE without Brier score or
log loss next to it.

**Platt scaling vs. isotonic regression.**
- *Platt:* fit `P(win) = σ(a·margin + b)`. It has two parameters, so it's very stable with little
  data, but it assumes the S-curve shape is right.
- *Isotonic:* fits any monotonic step function. It's flexible, but it needs much more data and can
  overfit by memorizing noise in small samples.

You'll find out which one wins here. Predict the answer before you run it.

**Nested out-of-sample (where people get burned).** The calibrator learns from (prediction,
outcome) pairs. Those predictions must themselves be *out-of-sample*. If you calibrate on the
model's training-set predictions, the model looks more certain than it really is (it has seen
those games), and the calibrator learns the wrong, overconfident mapping.

The walk-forward predictions in `reports/walk_forward_predictions.csv` are already out-of-sample:
each season was predicted by a model trained only on earlier seasons. So for test season S, you
fit the calibrator on the walk-forward predictions from seasons before S and apply it to S. The
first season has nothing prior to learn from, which is why it gets a null.

## Tasks

### Part A: building blocks (unit tests)
1. `reliability_table` and `expected_calibration_error`
2. `MarginCalibrator.fit` and `predict_proba` for `"platt"` and `"isotonic"`

Run `pytest tests/test_calibration.py -k "reliability or ece or calibrator"`.

### Part B: no-leak walk-forward
3. `walk_forward_calibrated`. The test `test_walk_forward_never_sees_the_test_season` flips every
   outcome in the test season and checks that your probabilities don't move. Read it and
   understand *why* that proves there's no leak.

### Part C: analysis (your write-up)
4. In a notebook or script, load `reports/walk_forward_predictions.csv`, drop ties
   (`result == 0`), and for `ridge`, `lgbm`, and `vegas`, compare the naive `Φ(m/13.5)`, Platt,
   and isotonic approaches on Brier score and ECE for seasons 2020 and later.
5. Draw a reliability diagram for the best and worst setup.
6. Write 5–10 sentences in `docs/exercises/01-calibration-results.md` covering these questions:
   Did calibration help? Which method won, and why does that fit how much data you had? What
   does the fitted Platt slope imply compared with 1/13.5?

Then open a PR with your branch, and I'll review it.

## Hints (open only when stuck)

<details><summary>Hint 1: reliability bins</summary>

`np.floor(prob * n_bins)` gives the bin index, but `prob == 1.0` lands in bin `n_bins`, and you
want it in `n_bins - 1`. `np.minimum` or `np.clip` fixes that. Polars `group_by("bin").agg(...)`
then `.sort("bin")` does the rest.
</details>

<details><summary>Hint 2: Platt with sklearn</summary>

sklearn wants a 2-D X, so use `margin.reshape(-1, 1)`. `LogisticRegression` regularizes by
default (`C=1.0`). With one feature and thousands of rows it barely matters, but think about
whether you want regularization on a calibration fit at all. `predict_proba(X)[:, 1]` is
P(class 1).
</details>

<details><summary>Hint 3: isotonic extrapolation</summary>

Look at the `out_of_bounds` argument of `IsotonicRegression`. The default raises an error or
returns NaN outside the training range, and the extrapolation test feeds margins of ±60.
`y_min` and `y_max` keep outputs in [0, 1].
</details>

<details><summary>Hint 4: why clip?</summary>

Isotonic can output exactly 0 or 1. One confident miss then makes log loss infinite. Clip to
something like `[1e-4, 1 - 1e-4]`. A strictly monotonic test still passes because the clip
preserves order.
</details>

<details><summary>Hint 5: walk-forward structure</summary>

Loop over the sorted unique seasons. For each one, split into `prior = season < S` and
`test = season == S`. If `prior` has fewer than `min_prior_seasons` distinct seasons, add a null
column. Otherwise, fit on `prior` and predict on `test`. Concatenate the results at the end.
Watch the dtype of the null column (`pl.lit(None, dtype=pl.Float64)`) or the concat will
complain.
</details>

## Interview check (answer in your own words, out loud)

1. Your model has 65% accuracy. Why isn't that enough to say its probabilities are good?
2. You fit a calibrator on the training set's predictions. What goes wrong, and in which
   direction?
3. When would you choose isotonic over Platt, and what evidence from this exercise supports
   your answer?
4. A stakeholder sees "ECE = 0.02" and calls the model great. What would you show them next?
5. How did you *prove* your walk-forward calibration doesn't leak?
