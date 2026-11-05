# Freight Rate Prediction

This project predicts `posted_rate` for truckloads using the supplied historical load data. The workflow uses a chronological evaluation, compares baselines with LightGBM, and produces predictions for the future validation loads and the fixed-lane December chart.

## Setup

Use Python 3.10 or later. From the project folder, install the pinned dependency ranges:

```powershell
python -m pip install -r requirements.txt
```

Keep the supplied CSV inputs in this folder:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

## Run the workflow

Run these commands from the project folder, in order:

```powershell
python stage1_eda.py
python stage2_validation.py
python stage3_modeling.py
python stage4_predictions.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Stage 3 runs six limited Optuna trials over expanding-window folds. Stage 4 uses the selected no-market-signals LightGBM model and writes the submission and December predictions. The scorer validates their format and creates the December chart; it does not calculate the final leaderboard metrics.

## Outputs

- EDA figures: `outputs/eda/`
- Stage 2 holdout and expanding-window CV metrics: `outputs/stage2/`
- Stage 3 model comparisons, tuning, feature importance, and error breakdowns: `outputs/stage3/`
- Final validation submission: `validation_predictions.csv` (`load_id,predicted_rate`)
- Completed December input: `december-chart-inputs.csv`
- Scorer chart: `scorer_results/candidate_december.png`

The current final files passed `score.py`: 12,000 positive validation predictions and 31 positive December predictions with the required columns and IDs. Final validation metrics are pending submission to Spotter. Recorded holdout metrics for the selected no-signals LightGBM are MAE $126.99, RMSE $639.94, MAPE 5.43%, and RMSLE 0.1621 (Sep–Oct 2025).

## Submission materials

- [Assessment report (PDF)](freight-rate-assessment-report.pdf)
- [Loom script](LOOM_SCRIPT.md)
- [Submission checklist](SUBMISSION_CHECKLIST.md)

