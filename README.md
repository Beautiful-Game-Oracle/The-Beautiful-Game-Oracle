# The-Beautiful-Game-Oracle

# Project Proposal  
## Predicting Match Outcomes in the English Premier League (EPL): A Comparative Data-Centric Approach

**Group:** Manan Tiwari (301628388), Tyler Nguyen (301458133), Yohann Pittappillil (301450169)

---

## Overview

When talking about collecting data in sports, we can categorize each sport into two realms: **discrete** and **continuous**.

- **Discrete sports** are defined as sports where actions and outcomes happen in separate, countable events.
- **Continuous sports** have performance measured on a continuous scale.

A sport like **football** has elements of both. Its gameplay flow is continuous (no set stopping time outside of halftime and fulltime), while its scoring events and many statistical models are discrete (e.g., expected points, shots).

This combination makes football difficult to predict—especially from a play-by-play basis. For now, our focus will be on outcomes that can be predicted, such as **winner**, **scoreline**, and **shot amounts**.

This project explores how different kinds of data influence predictive accuracy in football analytics. The goal is to forecast EPL match outcomes: **Home Win**, **Draw**, or **Away Win** by comparing three modelling paradigms:

- **Performance-based:** Recent team form and match performance metrics.  
- **Financial-based:** Club valuation, wage bills, and transfer spending as indicators of long-term strength.  
- **Market-based:** Bookmaker odds and implied probabilities representing collective market expectations.

By training and evaluating parallel models on these data sources, the project aims to understand how **data origin**, **quality**, and **representation** affect prediction accuracy, interpretability, and fairness.

---

## Motivation

**Track 2:** ML Project with Data Exploration focusing on building, analyzing, and comparing multiple models trained on distinct datasets to evaluate their predictive behaviour.

The motivation for this project comes from our shared interest in football analytics and machine learning experimentation. It provides an opportunity to evaluate how different **representations of knowledge**—on-field performance, financial investment, or collective market belief—affect predictive accuracy.

---

## Research Focus

1. Which dataset type best predicts the most accurate results?  
2. How does excluding or altering subsets of data (e.g., missing injuries or outdated valuations) affect outcomes?  
3. What can comparative analysis reveal about how different types of signals influence match outcomes?

---

## Data & Methods

**Sources:**

- Football-Data.co.uk — Match results and bookmaker odds  
  <https://www.football-data.co.uk/englandm.php>
- Transfermarkt API — Player valuations, transfer spending, and squad information  
  <https://transfermarkt-api.fly.dev/docs>
- FPL Elo Insights — Team and player performance metrics  
  <https://github.com/olbauday/FPL-Elo-Insights>
- Understat API — Supplementary event-level data  
  <https://understat.readthedocs.io/>

**Timeline (6 Weeks):**

1–2: Collect, clean, and align datasets  
3: Engineer derived and rolling features per dataset  
4: Train and tune models (XGBoost / Random Forest / Logistic Baseline)  
5: Run comparative and data-valuation analyses  
6: Visualize results; prepare final report

---

## Expected Outcomes

- A comparative analysis showing which modelling perspective best predicts EPL outcomes.  
- Insights into how performance, financial, and market factors influence predictive accuracy and generalization.  
- A final report demonstrating practical machine learning experimentation and data exploration in the context of sports analytics.

### Reinforcement Learning Extension Plan

- **Objective:** Frame match outcome prediction as a sequential decision problem where an agent allocates predictive confidence or staking decisions across a season, using historical fixtures as state transitions.
- **Environment:** Use cleaned league results to build season-long trajectories; state features combine engineered match deltas, financial indicators, and market odds snapshots aggregated per fixture. Reward options: (1) log-loss improvement over baseline predictions, (2) betting return using implied odds, or (3) calibration-aware scoring such as Brier reward.
- **Agent Design:** Start with off-policy evaluation using historical policy (baseline model). Candidate algorithms include contextual bandits for per-fixture actions, or episodic RL (e.g., PPO, A2C) with rolling bankroll state to capture temporal dependencies.
- **Training Strategy:** Pretrain models on past seasons (≥3 years) with curriculum that gradually increases action space complexity. Use offline RL with conservative Q-learning or doubly robust estimators to mitigate distribution shift.
- **Integration Path:** Align RL agent inputs with existing `match_features_df`; reuse TensorFlow pipelines for feature normalization. Compare agent-derived policies against supervised baselines via season replay simulations and run attribution (Shapley/LOO) on policy value drivers.
- **Next Steps:** Formalize reward definition, create environment wrapper (Gym-style), and prototype contextual bandit as low-risk stepping stone before full RL rollout.

### Momentum Interaction Reward Shaping

- The `momentum_policy_rl` baseline now feeds bookmaker implied probabilities alongside momentum features when constructing training episodes; each season trajectory carries a `(states, actions, market_probs)` tuple for reward analysis.
- REINFORCE updates still optimise expected discounted accuracy, but correct draw predictions receive an additive bonus (`draw_correct_bonus`) to offset the class imbalance that previously discouraged neutral outcomes.
- Correctly calling a result that the market priced as an underdog yields a scaled upset bonus (`market_upset_bonus`) proportional to how far the implied probability falls below the configurable `market_underdog_threshold`.
- Reward parameters are configurable through `FEATURE_SETS["momentum_policy_rl"]["reward_config"]`, enabling quick sensitivity passes without rewriting the agent; defaults target higher draw coverage while spotlighting genuine market overperformance.

### Performance Dense Preprocessing

- The supervised `performance_dense` baseline now constructs a private copy of the match table before training, limiting each run to its own feature view and eliminating cross-model data leakage.
- Rolling aggregates are smoothed with season/team priors before conversion to per-match averages, preventing cold-start zeros from overwhelming the dense layers early in the season.
- Compact, ratio-based features (e.g., log goal/xG/points ratios plus attack/defence gaps) tailor the input space to what a dense net can learn efficiently while stripping redundant raw sums.
- Features are z-scored using training-split statistics (`scale_strategy="zscore"`) and paired with seasonal sin/cos context, stabilising optimisation without distorting the RL or market pipelines.
- Splits receive their own `tf.data` pipelines derived from the scaled arrays, so validation/test performance reflects the exact preprocessing state used at inference time.
