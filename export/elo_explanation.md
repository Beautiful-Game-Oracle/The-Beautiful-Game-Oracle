# Elo explanation and code overview

This document contains formulas and a code-level explanation for the Elo implementation in `getTeamElo.py`. Use it directly in reports or append to project documentation.

## Explanation and formulas (citation-ready)

Below is a concise, citation-friendly explanation of each formula and the rationale.

### 1) Expected score (probability)

We model the probability that Team A beats Team B using the standard Elo logistic transform with a 400-point scale:

If Team A has Elo rating E_A and Team B has Elo rating E_B, and we optionally add a home advantage H (in Elo points) to Team A, the expected score (probability Team A wins) is:

E_A = 1 / (1 + 10^{(E_B - (E_A + H)) / 400})

This is the same base formula used in classical Elo systems (Elo, 1978). The 400 constant controls the steepness of the logistic curve.

In the code, `expected_score(elo_a, elo_b, home_adv_a=0.0)` implements this, with `HOME_ADVANTAGE` added to the home team's Elo when computing the expected value.

### 2) Match result mapping to numeric score

We convert match outcomes into numeric scores S in {1, 0.5, 0}:

- Home win: S_home = 1.0, S_away = 0.0
- Away win: S_home = 0.0, S_away = 1.0
- Draw: S_home = S_away = 0.5

This is implemented by `result_to_score(home_goals, away_goals)`.

### 3) Margin-of-victory multiplier (goal_mult)

To give larger Elo swings for larger goal differences, the code multiplies the base K by a margin factor:

goal_mult = 1 + G * ln(1 + goal_diff)

where:
- goal_diff = |home_goals - away_goals|
- G is `GOAL_WEIGHT` (a tunable parameter)

Rationale: The natural log grows sublinearly, so large routs increase movement but do not explode K.

### 4) Upset multiplier (upset_mult)

To amplify rating movement when a lower-probability outcome occurs (an upset), we scale by:

upset_mult = 1 + U * (1 - p_winner)

where:
- p_winner is the expected probability of the team that actually achieved the better result.
- U is `UPSET_WEIGHT`.

Rationale: When an underdog (low p_winner) wins, (1 - p_winner) is large and the multiplier increases.

### 5) Combined dynamic K and update rule

The code combines base K with the two multipliers to form the dynamic K:

K_dyn = K * goal_mult * upset_mult

and then applies the standard Elo update:

Delta = K_dyn * (S - E)

where S is the match score (1 / 0.5 / 0) for the team and E is that team's expected score. The team's new rating is:

Elo_new = Elo_old + Delta

Both home and away deltas are computed symmetrically (Δ_away = -Δ_home up to rounding and separate numerical operations).

### 6) Home advantage

The home team gets an additive boost H to its Elo when computing expected probabilities (expressed in Elo points, e.g., 40).

## Small worked numeric example (step-by-step)

Parameters (from `getTeamElo.py`):
- START_ELO = 1500
- K = 20
- HOME_ADVANTAGE H = 40
- UPSET_WEIGHT U = 1.5
- GOAL_WEIGHT G = 1.0

Scenario:
- Home team Elo: 1500
- Away team Elo: 1600
- Final score: Home 2 — Away 1 (home wins)
- goal_diff = 1

Step 1 — effective Elo for expectation
- Add home advantage to home team's Elo: 1500 + 40 = 1540
- Elo difference for denominator: 1600 - 1540 = 60

Step 2 — expected score
- E_home = 1 / (1 + 10^(60/400)) ≈ 0.4146
- E_away ≈ 0.5854

Step 3 — result scores
- Home win → S_home = 1.0, S_away = 0.0

Step 4 — goal multiplier
- goal_mult = 1 + G * ln(1 + 1) = 1 + ln(2) ≈ 1.6931

Step 5 — upset multiplier
- p_winner = E_home = 0.4146
- upset_mult = 1 + 1.5 * (1 - 0.4146) ≈ 1.8781

Step 6 — dynamic K
- K_dyn ≈ 20 * 1.6931 * 1.8781 ≈ 63.58

Step 7 — Elo deltas
- Δ_home ≈ 63.58 * (1 - 0.4146) ≈ 37.24
- Δ_away ≈ -37.24

Step 8 — new ratings
- Home: 1537.24
- Away: 1562.76

## Code-level overview: what `getTeamElo.py` does

- Inputs
  - Directory `understat_data/{league}/Team_Results/team_results.csv`. Expected to be team-centric rows with `match_id`, `venue`, `goals_for`, `team`, `date`, `season`.

- Processing steps
  1. Reconstruct matches via `load_matches_from_league_file` (pairs home/away rows by `match_id`).
  2. Initialize each team's Elo to START_ELO on first encounter.
  3. For each match (sorted by season/date): compute expectations, map result to scores, compute `goal_mult` and `upset_mult`, combine into `dynamic_K`, compute deltas and update Elo and stats, append a timeseries record.
  4. Write outputs.

- Outputs (fields written)
  - `team_elos.csv`: `team`, `final_elo`, `played`, `wins`, `draws`, `losses`
  - `Team_Results/team_elos_timeseries.csv`: `match_id`, `date`, `league`, `home_team`, `away_team`, `home_elo_pre`, `away_elo_pre`, `E_home`, `E_away`, `goal_diff`, `dynamic_K`, `delta_home`, `delta_away`, `home_elo_post`, `away_elo_post`

## Edge cases, assumptions and limitations

- Order dependent: Elo updates are sequential; sorting by season/date matters. Missing dates may affect ordering.
- Season reset / carryover: unseen teams start at START_ELO. Persist `team_elos.csv` if cross-season carryover is desired.
- Missing or malformed rows: `load_matches_from_league_file` skips matches it cannot reconstruct.
- Draw handling: for draws, `p_winner` uses the favorite's probability.
- Parameter sensitivity: K, HOME_ADVANTAGE, UPSET_WEIGHT, GOAL_WEIGHT strongly affect magnitudes and should be tuned.

## Citation-ready paragraph

The team Elo ratings were computed using a modified Elo algorithm. Expected win probabilities use the Elo logistic transform with an additive home-field advantage. The update magnitude is dynamic, scaling the base K by a log-based margin-of-victory factor and an upset multiplier that rewards surprising results. Ratings are updated with Δ = K_dyn · (S − E). This approach follows the classical Elo framework (Elo, 1978) with sports-specific extensions for margin-of-victory and upset amplification.

Suggested citation lines:
- Arpad Elo, "The Rating of Chessplayers, Past and Present", Arco, 1978.
- "Elo rating system." Wikipedia: https://en.wikipedia.org/wiki/Elo_rating_system

## Conceptual overview (non-technical)

This short section explains the Elo system and how the script uses it in plain language for readers who haven't seen Elo before.

- Each team has a single number (its Elo) that represents how strong the team is thought to be. Higher numbers mean a higher chance to win.
- Before a match, we compute the probability each team will win based on their current Elos and a small home-field boost for the home side.
- After the match we compare the actual result (win/draw/loss) to the expected probability. If the result was surprising (an underdog win) or emphatic (large goal margin), we apply a larger rating change.

How the script applies the idea:

1. Read matches and pair home/away rows into single match records.
2. For each match in time order: look up current Elos (or start at a baseline), compute expected probabilities, observe the result, compute multipliers for goal margin and upset, scale the base update size (K), then apply the update to both teams. The script records pre/post Elos and the components used for debugging or plotting.

Simple analogies and interpretation:
- Think of each team as having a weight on a balance scale. The heavier team is the favorite. After the match we nudge the scale toward whichever team won; surprising results cause bigger nudges.
- Large gains mean a team beat a stronger opponent or won convincingly. Small gains mean the result was close to what the model expected.

Practical tips:
- Ratings are order-dependent — the script processes matches chronologically. Incorrect or missing dates can change results.
- New teams start at the baseline (e.g., 1500). Persist `team_elos.csv` if you want ratings to carry across runs.
- The margin and upset multipliers are heuristics that make the ratings respond more to surprising or emphatic results; tune their values if you prefer smoother or more reactive ratings.

You can include this conceptual overview in reports or present it as a one-page explanation for non-technical stakeholders.
