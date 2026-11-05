# Loom walkthrough script (about 2–3 minutes)

## 0:00–0:20 — Goal and workflow

“This project predicts posted freight rates for future truckloads. I organized it into EDA, time-based validation, model comparison, and final prediction generation. The key rule was to preserve chronology: the model only learns from dates before the period it predicts.”

**Show:** project folder and `stage1_eda.py` through `stage4_predictions.py`.

## 0:20–0:55 — EDA and data quality

“There are 48,000 labeled loads from January 1 through October 31, 2025, and 12,000 future validation loads from November 1 through December 31. Weight has missing values and some negative values; I take its absolute value and impute missing values inside the training pipeline. Market index is also missing and was imputed from training data. Posted rates are right-skewed, so the model learns a capped log-transformed target. The reported road distance is highly correlated with coordinate-based distance, but differs by more than 25 miles for most rows; I retain the supplied road distance and treat the haversine feature as a separate signal rather than replacing it.”

**Show:** one of `outputs/eda/posted_rate_distribution.png`, `outputs/eda/distance_vs_haversine.png`, or `outputs/eda/numeric_feature_drift.png`.

## 0:55–1:25 — Validation design

“The final time holdout trains on January through August and evaluates September and October. I also used expanding-window monthly folds from May through August. This better reflects deployment than a random split, which could train on later market conditions and test on earlier loads. Preprocessing is fitted within each training partition.”

**Show:** `outputs/stage2/holdout_metrics.csv` and `outputs/stage3/cv_metrics.csv`.

## 1:25–2:00 — Model choice and results

“I compared a median-rate-per-mile baseline, Ridge on log rates, and LightGBM. On the September–October holdout, the no-market-signals LightGBM achieved MAE $126.99 and RMSE $639.94, versus MAE $161.85 for Ridge and $256.95 for the simple baseline. The four-fold mean MAE was about $134 for LightGBM and $134 for Ridge, so they are close across folds. I chose the no-signals model because it performed best on the final holdout and does not rely on forecasting market inputs that are missing from the December chart data. The estimated-signals variant performed worse on the holdout.”

**Show:** `outputs/stage3/holdout_metrics.csv` and the top rows of `outputs/stage3/feature_importance.csv`.

## 2:00–2:30 — Final outputs and close

“Stage 4 retrains the selected model on all labeled rows, writes the 12,000-row `validation_predictions.csv`, and fills the 31 December dates. The scorer checks the required file formats and creates this chart; it does not calculate Spotter’s final score. The submissions passed those format checks, and final leaderboard metrics will be available after submission.”

**Show:** `validation_predictions.csv`, `december-chart-inputs.csv`, `scorer_results/candidate_december.png`, then the checklist.

