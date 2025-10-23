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
