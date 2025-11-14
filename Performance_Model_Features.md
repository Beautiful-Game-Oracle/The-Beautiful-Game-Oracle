# Performance Model Feature Glossary

This document summarizes every feature currently used by the performance-view dense baseline, including how each signal is engineered inside `Football Predictor Models.ipynb`.

## Shared Mechanics
- `ROLLING_WINDOW = 5`, so all “avg5” metrics represent information from the previous five fixtures for each team.  
- `_prior_rolling_mean(df, team_col, value_col, window)` computes a shifted rolling mean (only prior matches) with fallbacks to the prior single value and finally the column median.  
- `_smoothed_avg(sum_series, games_frac, window)` converts last-5 cumulative sums into per-match averages, then shrinks toward the global prior via `alpha = games_played / window`.  
- `*_last_5` aggregates (goals, points, expected goals) are created upstream in the rolling feature pipeline (`analysis/dataset_vnext_scoping.py`).  
- Raw shots (`home_shots_for`, `away_shots_for`) and Elo ratings (`elo_*`) come from `build_dataset_version3.py` using Understat team logs and Elo timeseries.  
- Log ratios add an `EPS = 1e-3` to numerator/denominator before the logarithm to avoid infinities.

## Recency & Form Signals

| Feature | Description | Formula / Construction |
| --- | --- | --- |
| `home_recent_games_frac` | Share of the five-match window already played by the home club this season (shrinks noisy early-season stats). | `min(cumcount_home, 5) / 5`. |
| `away_recent_games_frac` | Same as above for the away club. | `min(cumcount_away, 5) / 5`. |
| `home_goals_for_avg5` | Smoothed goals scored per match over last five for the home team. | `_smoothed_avg(home_goals_for_last_5, home_recent_games_frac, 5)`. |
| `home_goals_against_avg5` | Smoothed goals allowed per match for the home team. | `_smoothed_avg(home_goals_against_last_5, home_recent_games_frac, 5)`. |
| `away_goals_for_avg5` | Smoothed goals scored per match for the away team. | `_smoothed_avg(away_goals_for_last_5, away_recent_games_frac, 5)`. |
| `away_goals_against_avg5` | Smoothed goals conceded per match for the away team. | `_smoothed_avg(away_goals_against_last_5, away_recent_games_frac, 5)`. |
| `att_gap_avg5` | Recent attacking edge (positive favors home). | `home_goals_for_avg5 - away_goals_for_avg5`. |
| `def_gap_avg5` | Recent defensive edge (positive when home concedes fewer). | `away_goals_against_avg5 - home_goals_against_avg5`. |
| `points_gap_avg5` | Differential in recent points earned per match. | `home_points_avg5 - away_points_avg5`, with each term from `_smoothed_avg(points_last_5, games_frac, 5)`. |
| `home_goals_against_avg5` | Included directly so the model can weight absolute defensive baseline. | `_smoothed_avg(home_goals_against_last_5, home_recent_games_frac, 5)`. |
| `away_goals_for_avg5` / `away_goals_against_avg5` | Explicitly exposed to let the model learn side-specific effects beyond the gap terms. | Same `_smoothed_avg` constructions as above. |
| `xg_att_gap_avg5` | Expected-goals attacking edge. | `home_xg_for_avg5 - away_xg_for_avg5`, each term smoothed from `*_xg_for_last_5`. |
| `xg_def_gap_avg5` | Expected-goals defensive edge. | `away_xg_against_avg5 - home_xg_against_avg5`. |
| `log_xg_ratio_avg5` | Log-scaled ratio of recent xG attack strength. | `log((home_xg_for_avg5 + 1e-3)/(away_xg_for_avg5 + 1e-3))`. |

## Shot Volume & Tempo Signals

| Feature | Description | Formula / Construction |
| --- | --- | --- |
| `home_shots_for_avg5` | Average shots per match for the home side over its last five fixtures. | `_prior_rolling_mean(home_shots_for, 5)` (uses Understat shot counts). |
| `away_shots_for_avg5` | Same for the away side. | `_prior_rolling_mean(away_shots_for, 5)`. |
| `home_shots_allowed_avg5` | Shots conceded per home match (opponent attempts). | `_prior_rolling_mean(home_shots_allowed, 5)` where `home_shots_allowed = away_shots_for`. |
| `away_shots_allowed_avg5` | Shots conceded per away match. | `_prior_rolling_mean(away_shots_allowed, 5)` where `away_shots_allowed = home_shots_for`. |
| `shot_vol_gap_avg5` | Differential in shot creation. | `home_shots_for_avg5 - away_shots_for_avg5`. |
| `shot_suppress_gap_avg5` | Differential in shot suppression. | `away_shots_allowed_avg5 - home_shots_allowed_avg5`. |
| `log_shot_ratio_avg5` | Log ratio of shot production with EPS stabilizer. | `log((home_shots_for_avg5 + 1e-3)/(away_shots_for_avg5 + 1e-3))` (replace inf/nan with 0). |
| `shots_tempo_avg5` | Average attacking tempo expected in the fixture. | `(home_shots_for_avg5 + away_shots_for_avg5) / 2`. |

## Volatility & Momentum Differentials

| Feature | Description | Formula / Construction |
| --- | --- | --- |
| `home_goal_diff_std5`, `away_goal_diff_std5` | Rolling standard deviation of each team’s goal difference over its last five matches; higher values imply erratic outcomes. | Maintain per-team deque of size 5 over goal differences, compute population std dev. |
| `goal_diff_std_gap5` | Relative volatility in goal difference between the two teams. | `home_goal_diff_std5 - away_goal_diff_std5`. |
| `home_goal_diff_exp_decay`, `away_goal_diff_exp_decay` | Exponential moving average of recent goal differences (α = 0.55) to capture fast-moving form swings. | `exp_avg_t = α * goal_diff_t + (1-α) * exp_avg_{t-1}` per team. |
| `goal_diff_exp_decay_gap` | Differential in exponential goal-diff momentum. | `home_goal_diff_exp_decay - away_goal_diff_exp_decay`. |
| `home_xg_diff_std5`, `away_xg_diff_std5` | Standard deviation of expected-goal differential per side across the trailing five fixtures. | Same std computation applied to xG differences. |
| `xg_diff_std_gap5` | Gap in xG volatility. | `home_xg_diff_std5 - away_xg_diff_std5`. |
| `home_xg_diff_exp_decay`, `away_xg_diff_exp_decay` | Exponential moving averages of xG differential, weighting latest matches more. | Same EMA with α = 0.55 on xG diffs. |
| `xg_diff_exp_decay_gap` | Differential of xG EMA momentum. | `home_xg_diff_exp_decay - away_xg_diff_exp_decay`. |
| `home_shot_diff_std5`, `away_shot_diff_std5` | Standard deviation of shot differential (shots for – shots against) across five matches. | Std over rolling deque fed by team_results shot data. |
| `shot_diff_std_gap5` | Relative shot volatility gap. | `home_shot_diff_std5 - away_shot_diff_std5`. |
| `home_shot_diff_exp_decay`, `away_shot_diff_exp_decay` | Exponential moving average of shot differential for each team. | EMA (α = 0.55) on shot differential. |
| `shot_diff_exp_decay_gap` | Difference in shot-based momentum proxies. | `home_shot_diff_exp_decay - away_shot_diff_exp_decay`. |

## Elo & Market Alignment Signals

| Feature | Description | Formula / Construction |
| --- | --- | --- |
| `elo_home_pre` | Pre-match Elo rating for the home club (Understat-derived, schedule-adjusted). | Directly ingested from `team_elos_timeseries.csv`, then numeric-cleaned/fillna median. |
| `elo_away_pre` | Pre-match Elo rating for the visitor. | Same ingestion pipeline. |
| `elo_mean_pre` | Overall fixture strength context. | `(elo_home_pre + elo_away_pre) / 2`. |
| `elo_gap_pre` | Elo differential (positive favors home). | `elo_home_pre - elo_away_pre`. |
| `elo_home_expectation` | Elo-implied win probability for the home side. | Pulled from `E_home` in the Elo timeseries; defaults to 0.5 when missing. |
| `elo_expectation_gap` | Probability gap implied by Elo. | `elo_home_expectation - (1 - elo_home_expectation)` (or 0 when expectations missing). |
| ~~`market_vs_elo_edge`~~ | ~~Difference between bookmaker home-win probability and Elo expectation, clipped to ±0.35. Removed from the performance view to avoid market odds leakage.~~ | ~~`clip(forecast_home_win - elo_home_expectation, -0.35, 0.35)`~~ |

These definitions should make it easier to spot redundant groups when pruning the performance model to a tighter, less biased feature slate.
