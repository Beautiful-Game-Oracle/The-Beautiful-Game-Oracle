# Dataset Version 7 – Change Log

## Summary
Dataset_Version_7 extends the Version 5 enrichment flow by layering volatility and momentum-differential diagnostics plus ensuring the modeling stack targets the new CSV. This release pairs with `build_dataset_version7.py`, which mirrors the V5 builder while writing `Dataset_Version_7.csv` and populating the new feature set.

## Data Engineering Updates
- **Volatility window (5 matches)** – Each team now maintains rolling standard deviations for goal difference, expected-goal difference, and shot differential ( `*_xg_diff_std5`, `*_shot_diff_std5`).
- **Exponential momentum (α = 0.55)** – Exponential moving averages for the same signals capture rapid swings (`*_goal_diff_exp_decay`, `*_xg_diff_exp_decay`, `*_shot_diff_exp_decay`).
- **Gap features** – Home vs away differentials for both std-dev and EMA metrics (`*_std_gap5`, `*_exp_decay_gvap`) highlight relative instability.
- **Sorted processing** – `build_dataset_version5.py` (and the derived V7 variant) now sorts matches chronologically before computing volatility so early fixtures receive proper warm-up handling.
- **Legacy fallbacks** – Notebook config now prioritises `Dataset_Version_7.csv` while retaining Version 6/3/1 paths as fallbacks.

## Modeling Integration
- **Performance view refresh** – `PerformanceFeatureView.feature_columns()` now lists every volatility column so the dense performance model ingests them alongside existing form/xG/shot metrics.
- **Feature glossary** – `Performance_Model_Features.md` gained a new “Volatility & Momentum Differentials” section describing the added inputs for reference during attribution or pruning.

## Next Steps
1. Regenerate `Dataset_Version_7.csv` via `python build_dataset_version7.py` to materialize the new columns.
2. Rerun `Football Predictor Models.ipynb` end-to-end so train/val/test splits, logs, and plots capture the refreshed feature space.
3. Recompute SHAP/LOO analyses to quantify the impact of the volatility signals.
