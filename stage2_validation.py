from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
OUTPUT_DIR = ROOT / "outputs" / "stage2"
HOLDOUT_START = pd.Timestamp("2025-09-01")
CV_VALIDATION_MONTHS = pd.date_range("2025-05-01", "2025-08-01", freq="MS")
TARGET_CAP_QUANTILE = 0.99
RIDGE_ALPHA = 10.0

NUMERIC_FEATURES = [
    "log_distance",
    "weight",
    "market_index",
    "quote_signal",
    "days_since_start",
    "day_of_year_sin",
    "day_of_year_cos",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> pd.DataFrame:
    data = pd.read_csv(TRAIN_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Training data contains invalid dates.")
    if data["posted_rate"].isna().any() or (data["posted_rate"] <= 0).any():
        raise ValueError("Training target must contain only positive, non-missing rates.")
    if (data["distance"] <= 0).any():
        raise ValueError("Distance must be positive for the rate-per-mile baseline.")
    return data.sort_values("date").reset_index(drop=True)


def make_features(frame: pd.DataFrame, date_origin: pd.Timestamp) -> pd.DataFrame:
    features = frame.copy()
    features["weight"] = pd.to_numeric(features["weight"], errors="coerce").abs()
    features["log_distance"] = np.log1p(features["distance"].astype(float))
    features["days_since_start"] = (
        features["date"] - date_origin
    ).dt.total_seconds() / 86400
    day_angle = 2 * np.pi * features["date"].dt.dayofyear / 365.25
    features["day_of_year_sin"] = np.sin(day_angle)
    features["day_of_year_cos"] = np.cos(day_angle)
    return features[FEATURES]


def build_ridge_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.maximum(predicted, 0)
    residual = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAPE_pct": float(np.mean(np.abs(residual / actual)) * 100),
        "RMSLE": float(
            np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
        ),
    }


def fit_and_score(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    date_origin: pd.Timestamp,
    split_name: str,
) -> list[dict[str, object]]:
    if train.empty or evaluation.empty:
        raise ValueError(f"{split_name} has an empty train or evaluation partition.")
    if train["date"].max() >= evaluation["date"].min():
        raise ValueError(f"{split_name} is not strictly time-ordered.")

    y_train = train["posted_rate"].to_numpy(dtype=float)
    y_eval = evaluation["posted_rate"].to_numpy(dtype=float)
    x_train = make_features(train, date_origin)
    x_eval = make_features(evaluation, date_origin)

    # Median is robust to rate outliers. The cap for Ridge is learned per training split.
    median_rate_per_mile = float((train["posted_rate"] / train["distance"]).median())
    baseline_prediction = median_rate_per_mile * evaluation["distance"].to_numpy()

    target_cap = float(train["posted_rate"].quantile(TARGET_CAP_QUANTILE))
    capped_log_target = np.log1p(np.minimum(y_train, target_cap))
    model = build_ridge_pipeline()
    model.fit(x_train, capped_log_target)
    ridge_prediction = np.maximum(np.expm1(model.predict(x_eval)), 0)

    records: list[dict[str, object]] = []
    for model_name, prediction in (
        ("median_rate_per_mile", baseline_prediction),
        ("ridge_log_target", ridge_prediction),
    ):
        records.append(
            {
                "split": split_name,
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "evaluation_start": evaluation["date"].min().date().isoformat(),
                "evaluation_end": evaluation["date"].max().date().isoformat(),
                "n_train": len(train),
                "n_evaluation": len(evaluation),
                "model": model_name,
                "ridge_target_cap_train_p99": target_cap,
                **calculate_metrics(y_eval, prediction),
            }
        )
    return records


def run_evaluation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_origin = data["date"].min()
    holdout_train = data.loc[data["date"] < HOLDOUT_START]
    holdout_eval = data.loc[data["date"] >= HOLDOUT_START]
    holdout_records = fit_and_score(
        holdout_train, holdout_eval, date_origin, "holdout_sep_oct"
    )

    cv_records: list[dict[str, object]] = []
    for validation_month in CV_VALIDATION_MONTHS:
        next_month = validation_month + pd.offsets.MonthBegin(1)
        fold_train = data.loc[data["date"] < validation_month]
        fold_eval = data.loc[
            (data["date"] >= validation_month) & (data["date"] < next_month)
        ]
        fold_name = f"cv_{validation_month:%Y_%m}"
        cv_records.extend(
            fit_and_score(fold_train, fold_eval, date_origin, fold_name)
        )

    holdout = pd.DataFrame(holdout_records)
    cv = pd.DataFrame(cv_records)
    return holdout, cv


def main() -> None:
    data = load_data()
    holdout, cv = run_evaluation(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_path = OUTPUT_DIR / "holdout_metrics.csv"
    cv_path = OUTPUT_DIR / "cv_metrics.csv"
    holdout.to_csv(holdout_path, index=False)
    cv.to_csv(cv_path, index=False)

    print("Time-based holdout metrics (Sep-Oct 2025):")
    print(
        holdout[
            ["model", "n_train", "n_evaluation", "MAE", "RMSE", "MAPE_pct", "RMSLE"]
        ].round(4).to_string(index=False)
    )
    print("\nExpanding-window CV metrics by fold:")
    print(
        cv[
            [
                "split",
                "model",
                "n_train",
                "n_evaluation",
                "MAE",
                "RMSE",
                "MAPE_pct",
                "RMSLE",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nExpanding-window CV mean by model:")
    print(
        cv.groupby("model")[["MAE", "RMSE", "MAPE_pct", "RMSLE"]]
        .mean()
        .round(4)
        .to_string()
    )
    print(f"\nSaved: {holdout_path.relative_to(ROOT)}")
    print(f"Saved: {cv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
