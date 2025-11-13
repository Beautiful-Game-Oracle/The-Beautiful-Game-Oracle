# Dataset_Version_5 Changes

## Why the update?
The SHAP + leave-one-out diagnostics introduced in `Football Predictor Models.ipynb` highlighted three concrete data issues:
- Cross-signal drift between bookmaker probabilities and Elo expectations (`market_vs_elo_edge` drove the largest accuracy drop despite a middling SHAP rank).
- Highly correlated raw Elo columns (`elo_home_pre`, `elo_away_pre`) amplified noise and occasionally *hurt* accuracy when present.
- Redundant temporal dummies (weekday sine/cosine variants) and shot aggregates that provided little attribution signal yet inflated the feature space.

Dataset_Version_5 tackles those findings so the downstream baselines receive cleaner, attribution-aligned inputs.

## What changed in the dataset builder?
1. **Stabilised market-vs-Elo edge** – `build_dataset_version5.py` now writes a clipped `market_vs_elo_edge` column directly into the CSV (difference between `forecast_home_win` and `elo_home_expectation`, limited to ±0.35). This keeps outlier bookmaker moves from overwhelming the dense net while preserving the useful deviation signal that SHAP flagged.
2. **Precomputed Elo gap features** – The exporter reconstructs `elo_gap_pre`, `elo_mean_pre`, `elo_expectation_gap`, plus per-season z-scores. Providing these ready-made, season-normalised markers lets the notebook drop redundant raw ratings without re-deriving stats each run.
3. **Documented strength summaries** – The script still writes the full Elo summary columns but now zero-fills any missing values explicitly so later z-scores stay stable.

Quick snapshot across the EPL slice (1,140 rows):
- `elo_gap_pre`: min **-221.706**, max **207.613**, mean **0.329**.
- `elo_expectation_gap`: min **-0.280**, max **0.677**, mean **0.156**.
- `market_vs_elo_edge`: min **-0.350**, max **0.350** (clip enforced), mean **-0.089**.

## Notebook alignment
- The configuration cells now point to `Dataset_Version_5.csv` / `Dataset_Version_5` labels so experiment metadata stays consistent.
- Dataset preparation no longer fabricates weekday dummies or trig features; `weekday_cols` is empty and the redundant columns are dropped before feature views are built.
- The dense and market views consume the new `elo_gap_pre*`, `elo_expectation_gap*`, and stabilised `market_vs_elo_edge` columns directly, ensuring SHAP + LOO analysis references the same cleaned signals defined in the CSV.

Together these updates keep the codebase faithful to the attribution learnings and give future data-augmentation passes a cleaner baseline to iterate on.
