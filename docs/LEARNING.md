# Learning Guide

This repo is built as a guided project. Claude acts as the senior engineer: it sets up the
scaffolding, interfaces, and tests and explains the reasoning behind them. You write the core
logic, so you can explain every design choice in an interview from your own experience.

## How each exercise works

1. Read the exercise doc in `docs/exercises/`. It covers the concept, why it matters, and what
   "done" means.
2. Look at the scaffold. Functions marked `# TODO(you)` raise `NotImplementedError`.
3. Run `pytest tests/` and make the failing tests pass. The tests are your spec.
4. Only open the hints when you're stuck. They go from a nudge to near-solution.
5. Push your work to a branch and open a PR. Claude reviews it like a senior engineer would,
   with questions first rather than rewrites.
6. Answer the "Interview check" questions at the bottom of the exercise in your own words.
   That's the real test.

## Exercise path

| # | Exercise | Skill | Status |
|---|----------|-------|--------|
| 0 | Defend the existing pipeline (below) | Leakage, validation design | read and answer |
| 1 | [Calibrated win probability](exercises/01-calibration.md) | Probabilistic classification, calibration, nested out-of-sample fitting | **ready** |
| 2 | Hyperparameter tuning with time-series CV | Model selection without leakage | planned |
| 3 | Model interpretation with SHAP | Explaining GBMs, feature attribution pitfalls | planned |
| 4 | Projecting starting QBs from the injury report | Feature engineering from messy operational data | planned |
| 5 | Experiment tracking with MLflow | Reproducibility, model registry | planned |
| 6 | Weekly scoring job and monitoring | Orchestration, drift and performance monitoring | planned (infrastructure choice pending) |

---

## Exercise 0: Defend the existing pipeline

Claude wrote the first version (ingest, features, and walk-forward baselines) before the switch
to guided mode. Don't just take it on faith. Read each module, then answer the questions without
looking at the notes. Each note is the explanation you'd give in an interview.

### `ingest.py`

**Q: Why cache past seasons but always re-pull the current one?**
<details><summary>Notes</summary>

Completed seasons are effectively immutable, while the current season gets new games every week
and nflverse sometimes corrects recent plays. Re-pulling only what can change keeps the job fast
and idempotent, so running it twice gives the same result. This is the batch version of
incremental loading.
</details>

**Q: Why keep only ~30 of ~370 play-by-play columns?**
<details><summary>Notes</summary>

Storage and speed matter, but the main reason is that a small, explicit schema shows what the
model depends on. If upstream renames a column you use, it fails loudly instead of silently
pulling in something new. The tradeoff is a re-ingest whenever you want a new feature.
</details>

### `features.py`

**Q: The feature for game *g* is built from games before *g*. Where exactly is that enforced, and
what would break if `.shift(1)` were removed?**
<details><summary>Notes</summary>

`pl.col(c).shift(1).ewm_mean(...).over("team")` shifts each team's history down one row before
smoothing. Without the shift, each game's own EPA would be in its own features, which is target
leakage. EPA in a game is strongly correlated with the final margin, so the backtest would look
spectacular and fail completely live. Interviewers love asking how you'd catch this. One answer
is a test that recomputes a game's features with that game's plays removed and checks that
nothing changes.
</details>

**Q: Why an exponentially weighted mean rather than the season-to-date average?**
<details><summary>Notes</summary>

Season-to-date resets to nothing in Week 1 and weights Week 1 the same as last week. An EWMA that
runs across seasons carries decayed prior-season form into Week 1 and reacts faster to change.
The half-life (8 games) is a bias-variance knob: shorter reacts to real changes but chases noise,
and longer is stable but slow. It's currently hand-picked, and tuning it properly is Exercise 2.
</details>

**Q: QB form is "shrunk" toward -0.10 EPA per dropback using 4 pseudo-games. Why?**
<details><summary>Notes</summary>

A backup with one great game would otherwise look elite. Shrinkage (Bayesian with a conjugate-ish
prior, or empirical Bayes in spirit) pulls small samples toward a sensible prior and leaves large
samples mostly alone. The prior is backup-level rather than league average because QBs with
little history are usually backups. The prior's strength (4) is another tunable choice.
</details>

**Q: Why map `SD→LAC`, `STL→LA`, `OAK→LV`?**
<details><summary>Notes</summary>

The schedules use the team code from that season while play-by-play uses current codes, so a join
on team silently dropped those teams' rows. This bug showed up as 141 games with missing features.
The general lesson is to always count nulls after a join.
</details>

### `evaluate.py`

**Q: Why walk-forward by season instead of random K-fold?**
<details><summary>Notes</summary>

Random folds put 2024 games in training while testing on 2022. The model can then learn from the
future (rule changes, scoring environment, team quality), and random splits also put correlated
games from the same week on both sides. Walk-forward copies how the model is used in practice:
train on the past and predict the next season. The cost is fewer training rows for early folds.
</details>

**Q: Why compare against the Vegas spread instead of just reporting accuracy?**
<details><summary>Notes</summary>

Accuracy alone means nothing without a reference point. Home field alone gets 54%. The closing
line is the strongest public benchmark there is, since it aggregates information from a market
with money on it. Saying "we close X% of the gap between naive and market" is a much more honest
framing.
</details>

**Q: Win probability is currently `Φ(margin / 13.5)`. What's wrong with that?**
<details><summary>Notes</summary>

It assumes the model's margin errors are normal with a fixed 13.5-point standard deviation, and
that the model's margins are on the right scale. Neither is checked. That's what Exercise 1 is
about.
</details>
