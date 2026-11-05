from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import stage3_modeling as stage3


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
VALIDATION_PATH = ROOT / "validation.csv"
TEMPLATE_PATH = ROOT / "validation-predictions-template.csv"
DECEMBER_PATH = ROOT / "december-chart-inputs.csv"
OPTUNA_TRIALS_PATH = ROOT / "outputs" / "stage3" / "optuna_trials.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

PREDICTION_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def load_best_parameters() -> dict[str, object]:
    if not OPTUNA_TRIALS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Stage 3 tuning results: {OPTUNA_TRIALS_PATH}"
        )
    trials = pd.read_csv(OPTUNA_TRIALS_PATH)
    completed = trials.loc[trials["state"].eq("COMPLETE")].copy()
    if completed.empty:
        raise ValueError("No completed Stage 3 Optuna trials are available.")
    best = completed.loc[completed["value"].idxmin()]
    parameter_names = (
        "learning_rate",
        "max_depth",
        "min_child_samples",
        "n_estimators",
        "num_leaves",
        "reg_lambda",
    )
    parameters: dict[str, object] = {}
    for name in parameter_names:
        value = best[f"params_{name}"]
        if name in {"max_depth", "min_child_samples", "n_estimators", "num_leaves"}:
            parameters[name] = int(value)
        else:
            parameters[name] = float(value)
    parameters["include_signals"] = False
    print(
        f"Using Stage 3 best trial #{int(best['number'])} "
        f"(CV MAE=${best['value']:.2f}): {parameters}"
    )
    return parameters


def validate_final_files(
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    december: pd.DataFrame,
) -> None:
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {PREDICTION_COLUMNS}.")
    if len(predictions) != 12_000:
        raise ValueError(f"Expected 12,000 validation predictions, got {len(predictions)}.")
    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        raise ValueError("Predictions contain missing or duplicate load IDs.")
    if predictions["load_id"].astype(str).tolist() != validation["load_id"].astype(str).tolist():
        raise ValueError("Prediction IDs/order do not match validation.csv.")
    prediction_values = pd.to_numeric(predictions["predicted_rate"], errors="coerce")
    if prediction_values.isna().any() or not np.isfinite(prediction_values).all():
        raise ValueError("Validation predictions contain missing/non-finite rates.")
    if (prediction_values <= 0).any():
        raise ValueError("Validation predictions must all be positive.")
    if list(december.columns) != DECEMBER_COLUMNS:
        raise ValueError(f"December columns must remain exactly {DECEMBER_COLUMNS}.")
    if len(december) != 31:
        raise ValueError(f"Expected 31 December rows, got {len(december)}.")
    december_values = pd.to_numeric(december["predicted_rate"], errors="coerce")
    if december_values.isna().any() or not np.isfinite(december_values).all():
        raise ValueError("December predictions contain missing/non-finite rates.")
    if (december_values <= 0).any():
        raise ValueError("December predictions must all be positive.")


def main() -> None:
    data = stage3.load_training_data()
    validation = pd.read_csv(VALIDATION_PATH)
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")
    if validation["date"].isna().any():
        raise ValueError("Validation data contains invalid dates.")
    if data["date"].max() >= validation["date"].min():
        raise ValueError("Validation dates must be strictly after all training dates.")

    template = pd.read_csv(TEMPLATE_PATH)
    if list(template.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction template columns must be {PREDICTION_COLUMNS}.")
    if template["load_id"].duplicated().any():
        raise ValueError("Prediction template contains duplicate load IDs.")
    if set(template["load_id"].astype(str)) != set(validation["load_id"].astype(str)):
        raise ValueError("Prediction template IDs do not match validation.csv.")

    parameters = load_best_parameters()

    print(f"Retraining chosen no-signals LightGBM on all {len(data):,} labeled rows...")
    _, validation_prediction, _, target_cap = stage3.fit_variant(
        data, validation, parameters, include_signals=False
    )
    predictions = template.copy()
    prediction_by_id = pd.Series(
        validation_prediction,
        index=validation["load_id"].astype(str),
        name="predicted_rate",
    )
    predictions["predicted_rate"] = predictions["load_id"].astype(str).map(
        prediction_by_id
    )

    december_original = pd.read_csv(DECEMBER_PATH)
    if list(december_original.columns) != DECEMBER_COLUMNS:
        raise ValueError("December input must retain the scorer's required seven columns.")
    december = december_original.copy()
    december["date"] = pd.to_datetime(december["date"], errors="coerce")
    if december["date"].isna().any():
        raise ValueError("December inputs contain invalid dates.")
    coordinate_map = stage3.make_city_coordinate_map(data)
    december_with_coordinates = stage3.attach_coordinates(december, coordinate_map)
    coordinate_columns = [
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]
    if december_with_coordinates[coordinate_columns].isna().any().any():
        raise ValueError("Could not resolve coordinates for the December lane.")

    print("Retraining the same model on all labeled rows for the December lane...")
    _, december_prediction, _, _ = stage3.fit_variant(
        data,
        december_with_coordinates,
        parameters,
        include_signals=False,
    )
    december["predicted_rate"] = december_prediction

    validate_final_files(predictions, validation, december)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    december.to_csv(DECEMBER_PATH, index=False, date_format="%Y-%m-%d")

    daily_changes = december["predicted_rate"].diff().abs().dropna()
    mean_rate = float(december["predicted_rate"].mean())
    max_daily_change = float(daily_changes.max()) if not daily_changes.empty else 0.0
    max_daily_change_pct = max_daily_change / mean_rate * 100 if mean_rate else np.nan
    print(f"Final training target cap (p99): ${target_cap:.2f}")
    print(f"Validation rows: {len(predictions):,}; all predictions are finite and positive.")
    print(f"Saved: {PREDICTIONS_PATH.name}")
    print(
        "December curve: "
        f"min=${december['predicted_rate'].min():.2f}, "
        f"mean=${mean_rate:.2f}, "
        f"max=${december['predicted_rate'].max():.2f}, "
        f"std=${december['predicted_rate'].std():.2f}, "
        f"max daily change=${max_daily_change:.2f} ({max_daily_change_pct:.2f}% of mean)"
    )
    print(f"Saved completed chart inputs: {DECEMBER_PATH.name}")


if __name__ == "__main__":
    main()
