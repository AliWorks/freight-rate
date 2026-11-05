from __future__ import annotations

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
DECEMBER_PATH = ROOT / "december-chart-inputs.csv"
STAGE2_CV_PATH = ROOT / "outputs" / "stage2" / "cv_metrics.csv"
STAGE2_HOLDOUT_PATH = ROOT / "outputs" / "stage2" / "holdout_metrics.csv"
OUTPUT_DIR = ROOT / "outputs" / "stage3"

HOLDOUT_START = pd.Timestamp("2025-09-01")
TIME_ORIGIN = pd.Timestamp("2025-01-01")
CV_VALIDATION_MONTHS = pd.date_range("2025-05-01", "2025-08-01", freq="MS")
LANE_SMOOTHING = 20.0
TARGET_CAP_QUANTILE = 0.99
OPTUNA_TRIALS = 6
SIGNAL_TREND_WEEKS = 8
RANDOM_SEED = 17

NUMERIC_FEATURES = [
    "log_distance",
    "weight",
    "haversine_miles",
    "lane_log_rate_encoding",
    "days_since_start",
    "day_of_week",
    "month",
    "day_of_year_sin",
    "day_of_year_cos",
]
SIGNAL_FEATURES = ["market_index", "quote_signal"]
CATEGORICAL_FEATURES = [
    "weight_bin",
    "equipment",
    "pickup",
    "delivery",
]
BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(TRAIN_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Training data contains invalid dates.")
    if data["posted_rate"].isna().any() or (data["posted_rate"] <= 0).any():
        raise ValueError("Training target must contain only positive rates.")
    if (data["distance"] <= 0).any():
        raise ValueError("Distance must be positive.")
    data["weight"] = pd.to_numeric(data["weight"], errors="coerce").abs()
    return data.sort_values(["date", "load_id"]).reset_index(drop=True)


def make_city_coordinate_map(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    pickup = data[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery = data[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coordinates = pd.concat([pickup, delivery], ignore_index=True).dropna()
    coordinates["coord"] = list(
        zip(coordinates["lat"].round(5), coordinates["lon"].round(5))
    )
    variants = coordinates.groupby("city")["coord"].nunique()
    inconsistent = variants[variants > 1]
    if not inconsistent.empty:
        raise ValueError(
            "City coordinates are inconsistent: "
            + ", ".join(map(str, inconsistent.index[:10]))
        )
    return (
        coordinates.drop_duplicates("city")
        .set_index("city")["coord"]
        .to_dict()
    )


def attach_coordinates(
    frame: pd.DataFrame,
    coordinate_map: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    result = frame.copy()
    for role, lat_col, lon_col in (
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ):
        if lat_col not in result:
            result[lat_col] = np.nan
        if lon_col not in result:
            result[lon_col] = np.nan
        city_coords = result[role].map(coordinate_map)
        missing_coords = result[lat_col].isna() | result[lon_col].isna()
        result.loc[missing_coords, lat_col] = city_coords[missing_coords].map(
            lambda value: value[0] if isinstance(value, tuple) else np.nan
        )
        result.loc[missing_coords, lon_col] = city_coords[missing_coords].map(
            lambda value: value[1] if isinstance(value, tuple) else np.nan
        )
    return result


def haversine_miles(frame: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(frame["pickup_lat"].to_numpy(dtype=float))
    lat2 = np.radians(frame["delivery_lat"].to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = np.radians(
        frame["delivery_lon"].to_numpy(dtype=float)
        - frame["pickup_lon"].to_numpy(dtype=float)
    )
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def add_calendar_and_route_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["lane"] = result["pickup"].astype(str) + "__" + result["delivery"].astype(str)
    result["log_distance"] = np.log1p(result["distance"].astype(float))
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce").abs()
    result["weight_bin"] = pd.cut(
        result["weight"],
        bins=[-np.inf, 10000, 20000, 30000, 40000, np.inf],
        labels=["<=10k", "10k-20k", "20k-30k", "30k-40k", ">40k"],
    ).astype("object")
    result["haversine_miles"] = haversine_miles(result)
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["days_since_start"] = (
        result["date"] - TIME_ORIGIN
    ).dt.total_seconds() / 86400
    result["day_of_year_sin"] = np.sin(
        2 * np.pi * result["date"].dt.dayofyear / 365.25
    )
    result["day_of_year_cos"] = np.cos(
        2 * np.pi * result["date"].dt.dayofyear / 365.25
    )
    return result


def add_historical_lane_encoding(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Encode training rows from earlier dates only; evaluation uses fit rows only."""
    fit_order = fit.sort_values(["date", "load_id"])
    fit_target = np.log1p(fit_order["posted_rate"].to_numpy(dtype=float))
    daily = (
        pd.DataFrame(
            {
                "date": fit_order["date"].to_numpy(),
                "lane": fit_order["lane"].to_numpy(),
                "target": fit_target,
            }
        )
        .groupby(["date", "lane"], sort=True)["target"]
        .agg(["sum", "count"])
    )

    lane_sum: dict[str, float] = {}
    lane_count: dict[str, int] = {}
    global_sum = 0.0
    global_count = 0
    encoded_by_date_lane: dict[tuple[pd.Timestamp, str], float] = {}
    encoded_by_date: dict[pd.Timestamp, float] = {}

    for date in sorted(daily.index.get_level_values("date").unique()):
        global_prior = global_sum / global_count if global_count else np.nan
        date_rows = daily.loc[date]
        date_lane_values: dict[str, float] = {}
        for lane, row in date_rows.iterrows():
            count = lane_count.get(lane, 0)
            if count:
                date_lane_values[lane] = (
                    lane_sum[lane] + LANE_SMOOTHING * global_prior
                ) / (count + LANE_SMOOTHING)
            else:
                date_lane_values[lane] = global_prior

        encoded_by_date[date] = global_prior
        for lane, value in date_lane_values.items():
            encoded_by_date_lane[(date, lane)] = value

        for lane, row in date_rows.iterrows():
            lane_sum[lane] = lane_sum.get(lane, 0.0) + float(row["sum"])
            lane_count[lane] = lane_count.get(lane, 0) + int(row["count"])
            global_sum += float(row["sum"])
            global_count += int(row["count"])

    fit_encoding = pd.Series(
        [
            encoded_by_date_lane.get((date, lane), np.nan)
            for date, lane in zip(fit["date"], fit["lane"])
        ],
        index=fit.index,
        name="lane_log_rate_encoding",
        dtype=float,
    )

    global_mean = global_sum / global_count if global_count else np.nan
    eval_encoding = evaluation["lane"].map(
        lambda lane: (
            lane_sum.get(lane, 0.0) + LANE_SMOOTHING * global_mean
        )
        / (lane_count.get(lane, 0) + LANE_SMOOTHING)
        if global_count
        else np.nan
    )
    eval_encoding.name = "lane_log_rate_encoding"
    return fit_encoding, eval_encoding


def estimate_future_signals(
    fit: pd.DataFrame,
    future_dates: pd.Series,
) -> pd.DataFrame:
    """Extrapolate 8 recent weekly means; fit data must end before future dates."""
    if fit.empty:
        raise ValueError("Cannot forecast market signals without training history.")
    if fit["date"].max() >= future_dates.min():
        raise ValueError("Signal forecast must not use evaluation-period observations.")

    history = fit.set_index("date")[SIGNAL_FEATURES].sort_index()
    weekly = history.resample("W-SUN").mean().dropna(how="all").tail(
        SIGNAL_TREND_WEEKS
    )
    predicted: dict[str, np.ndarray] = {}
    future_day = future_dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)

    for column in SIGNAL_FEATURES:
        series = weekly[column].dropna()
        if series.empty:
            raise ValueError(f"No historical values available for {column}.")
        x = np.array([date.toordinal() for date in series.index], dtype=float)
        y = series.to_numpy(dtype=float)
        if len(series) >= 2:
            slope, intercept = np.polyfit(x, y, deg=1)
            values = slope * future_day + intercept
        else:
            values = np.full(len(future_dates), y[-1], dtype=float)
        lower = float(history[column].quantile(0.01))
        upper = float(history[column].quantile(0.99))
        predicted[column] = np.clip(values, lower, upper)
    return pd.DataFrame(predicted, index=future_dates.index)


def build_feature_matrices(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    include_signals: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_features = add_calendar_and_route_features(fit)
    eval_features = add_calendar_and_route_features(evaluation)
    train_lane, eval_lane = add_historical_lane_encoding(
        train_features, eval_features
    )
    train_features["lane_log_rate_encoding"] = train_lane
    eval_features["lane_log_rate_encoding"] = eval_lane

    if include_signals:
        forecast = estimate_future_signals(fit, evaluation["date"])
        for column in SIGNAL_FEATURES:
            eval_features[column] = forecast[column].to_numpy()
        train_columns = BASE_FEATURES + SIGNAL_FEATURES
    else:
        train_columns = BASE_FEATURES
    eval_columns = train_columns
    return (
        train_features[train_columns],
        eval_features[eval_columns],
        train_lane,
        eval_lane,
    )


def build_model(parameters: dict[str, object]) -> Pipeline:
    numeric_features = NUMERIC_FEATURES + (
        SIGNAL_FEATURES if parameters.get("include_signals", False) else []
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "include_signals"
    }
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                lgb.LGBMRegressor(
                    objective="regression",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=-1,
                    **model_parameters,
                ),
            ),
        ]
    )


def score_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    actual = np.asarray(actual, dtype=float)
    error = predicted - actual
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE_pct": float(np.mean(np.abs(error / actual)) * 100),
        "RMSLE": float(
            np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))
        ),
    }


def fold_partitions(data: pd.DataFrame):
    for validation_month in CV_VALIDATION_MONTHS:
        next_month = validation_month + pd.offsets.MonthBegin(1)
        fit = data.loc[data["date"] < validation_month]
        evaluation = data.loc[
            (data["date"] >= validation_month) & (data["date"] < next_month)
        ]
        yield f"cv_{validation_month:%Y_%m}", fit, evaluation


def make_target(fit: pd.DataFrame) -> tuple[np.ndarray, float]:
    y = fit["posted_rate"].to_numpy(dtype=float)
    cap = float(np.quantile(y, TARGET_CAP_QUANTILE))
    return np.log1p(np.minimum(y, cap)), cap


def tune_lightgbm(data: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    """Tune signal-free model on four expanding-window folds; holdout stays untouched."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, object] = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.12, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 7, 47),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        fold_mae: list[float] = []
        for _, fit, evaluation in fold_partitions(data):
            x_fit, x_eval, _, _ = build_feature_matrices(
                fit, evaluation, include_signals=False
            )
            target, _ = make_target(fit)
            model = build_model(parameters)
            model.fit(x_fit, target)
            prediction = np.expm1(model.predict(x_eval))
            fold_mae.append(
                mean_absolute_error(
                    evaluation["posted_rate"].to_numpy(dtype=float),
                    np.maximum(prediction, 0.0),
                )
            )
        return float(np.mean(fold_mae))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    params = dict(study.best_params)
    params["include_signals"] = False
    trials = study.trials_dataframe()
    return params, trials


def fit_variant(
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    parameters: dict[str, object],
    include_signals: bool,
):
    x_fit, x_eval, _, _ = build_feature_matrices(
        fit, evaluation, include_signals=include_signals
    )
    target, cap = make_target(fit)
    params = dict(parameters)
    params["include_signals"] = include_signals
    model = build_model(params)
    model.fit(x_fit, target)
    prediction = np.maximum(np.expm1(model.predict(x_eval)), 0.0)
    return model, prediction, x_fit.columns.tolist(), cap


def evaluate_cv(
    data: pd.DataFrame,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    holdout_predictions: dict[str, pd.DataFrame] = {}

    for split_name, fit, evaluation in fold_partitions(data):
        baseline_rate_mile = float(
            (fit["posted_rate"] / fit["distance"]).median()
        )
        baseline_prediction = baseline_rate_mile * evaluation["distance"].to_numpy()
        records.append(
            {
                "split": split_name,
                "model": "median_rate_per_mile",
                **score_metrics(evaluation["posted_rate"], baseline_prediction),
            }
        )
        for include_signals, variant in (
            (False, "lightgbm_A_no_signals"),
            (True, "lightgbm_B_estimated_signals"),
        ):
            _, prediction, _, _ = fit_variant(
                fit, evaluation, parameters, include_signals
            )
            records.append(
                {
                    "split": split_name,
                    "model": variant,
                    **score_metrics(evaluation["posted_rate"], prediction),
                }
            )

    cv = pd.DataFrame(records)
    if STAGE2_CV_PATH.exists():
        stage2 = pd.read_csv(STAGE2_CV_PATH)
        stage2 = stage2[stage2["model"].eq("ridge_log_target")].copy()
        stage2 = stage2.rename(columns={"model": "stage2_model"})
        stage2["model"] = "stage2_ridge_log_target"
        cv = pd.concat(
            [
                cv,
                stage2[["split", "model", "MAE", "RMSE", "MAPE_pct", "RMSLE"]],
            ],
            ignore_index=True,
        )
    return cv, pd.DataFrame()


def evaluate_holdout(
    data: pd.DataFrame,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fit = data.loc[data["date"] < HOLDOUT_START]
    evaluation = data.loc[data["date"] >= HOLDOUT_START].copy()
    actual = evaluation["posted_rate"].to_numpy(dtype=float)
    records: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, Pipeline] = {}

    baseline_rate_mile = float((fit["posted_rate"] / fit["distance"]).median())
    baseline_prediction = baseline_rate_mile * evaluation["distance"].to_numpy()
    predictions["median_rate_per_mile"] = baseline_prediction
    records.append(
        {
            "model": "median_rate_per_mile",
            "train_rows": len(fit),
            "evaluation_rows": len(evaluation),
            **score_metrics(actual, baseline_prediction),
        }
    )

    for include_signals, variant in (
        (False, "lightgbm_A_no_signals"),
        (True, "lightgbm_B_estimated_signals"),
    ):
        model, prediction, _, cap = fit_variant(
            fit, evaluation, parameters, include_signals
        )
        models[variant] = model
        predictions[variant] = prediction
        records.append(
            {
                "model": variant,
                "train_rows": len(fit),
                "evaluation_rows": len(evaluation),
                "training_target_cap_p99": cap,
                **score_metrics(actual, prediction),
            }
        )

    if STAGE2_HOLDOUT_PATH.exists():
        stage2 = pd.read_csv(STAGE2_HOLDOUT_PATH)
        stage2 = stage2[stage2["model"].eq("ridge_log_target")]
        if not stage2.empty:
            row = stage2.iloc[0]
            records.append(
                {
                    "model": "stage2_ridge_log_target",
                    "train_rows": int(row["n_train"]),
                    "evaluation_rows": int(row["n_evaluation"]),
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "MAPE_pct": row["MAPE_pct"],
                    "RMSLE": row["RMSLE"],
                }
            )

    prediction_frame = evaluation[
        ["load_id", "date", "pickup", "delivery", "equipment", "distance", "posted_rate"]
    ].copy()
    prediction_frame["distance_bucket"] = pd.cut(
        prediction_frame["distance"],
        bins=[0, 500, 1000, 1500, 2000, np.inf],
        labels=["<=500", "500-1000", "1000-1500", "1500-2000", ">2000"],
        include_lowest=True,
    ).astype("object")
    for name, prediction in predictions.items():
        prediction_frame[f"prediction_{name}"] = prediction
    return pd.DataFrame(records), prediction_frame, models


def make_error_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    model_columns = [
        column for column in predictions if column.startswith("prediction_")
    ]
    groupings = {
        "equipment": ["equipment"],
        "distance_bucket": ["distance_bucket"],
        "month": [predictions["date"].dt.to_period("M").astype(str)],
    }
    records: list[dict[str, object]] = []
    for model_column in model_columns:
        for group_name, group_key in groupings.items():
            if group_name == "month":
                grouped = predictions.groupby(group_key[0], observed=True, dropna=False)
            else:
                grouped = predictions.groupby(
                    group_key, observed=True, dropna=False
                )
            for value, frame in grouped:
                if isinstance(value, tuple):
                    value = value[0]
                records.append(
                    {
                        "model": model_column.removeprefix("prediction_"),
                        "breakdown": group_name,
                        "group": str(value),
                        "n": len(frame),
                        **score_metrics(
                            frame["posted_rate"].to_numpy(),
                            frame[model_column].to_numpy(),
                        ),
                    }
                )
    return pd.DataFrame(records)


def extract_importance(
    model: Pipeline,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    fitted_preprocessor: ColumnTransformer = model.named_steps["preprocessing"]
    fitted_lgbm: lgb.LGBMRegressor = model.named_steps["model"]
    names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_lgbm.feature_importances_
    result = pd.DataFrame(
        {
            "model": model_name,
            "feature": names,
            "importance": importances,
        }
    )
    total = result["importance"].sum()
    result["importance_share"] = (
        result["importance"] / total if total else 0.0
    )
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def generate_december_predictions(
    data: pd.DataFrame,
    parameters: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    december = pd.read_csv(DECEMBER_PATH)
    december["date"] = pd.to_datetime(december["date"], errors="coerce")
    if december["date"].isna().any():
        raise ValueError("December chart inputs contain invalid dates.")
    coordinate_map = make_city_coordinate_map(data)
    december = attach_coordinates(december, coordinate_map)
    missing_coordinates = december[
        ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    ].isna().any(axis=1)
    if missing_coordinates.any():
        raise ValueError("Unable to infer coordinates for one or more December cities.")

    forecasts = estimate_future_signals(data, december["date"])
    december["estimated_market_index"] = forecasts["market_index"].to_numpy()
    december["estimated_quote_signal"] = forecasts["quote_signal"].to_numpy()
    output = december[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "estimated_market_index",
            "estimated_quote_signal",
        ]
    ].copy()

    coordinate_ready = attach_coordinates(december, coordinate_map)
    for include_signals, variant in (
        (False, "lightgbm_A_no_signals"),
        (True, "lightgbm_B_estimated_signals"),
    ):
        model, prediction, _, _ = fit_variant(
            data, coordinate_ready, parameters, include_signals
        )
        output[f"predicted_rate_{variant}"] = prediction
    return output, forecasts.reset_index(drop=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_training_data()
    print(
        f"Loaded {len(data):,} training rows "
        f"({data['date'].min().date()} through {data['date'].max().date()})."
    )

    print(f"\nTuning LightGBM on {OPTUNA_TRIALS} trials × 4 expanding folds...")
    best_parameters, trials = tune_lightgbm(data)
    trials.to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)
    print("Selected parameters:", best_parameters)
    print(
        f"Best mean CV MAE during tuning: "
        f"${float(trials.loc[trials['value'].idxmin(), 'value']):.2f}"
    )

    cv, _ = evaluate_cv(data, best_parameters)
    cv.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    holdout, holdout_predictions, holdout_models = evaluate_holdout(
        data, best_parameters
    )
    holdout.to_csv(OUTPUT_DIR / "holdout_metrics.csv", index=False)
    holdout_predictions.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    breakdown = make_error_breakdown(holdout_predictions)
    breakdown.to_csv(OUTPUT_DIR / "holdout_error_breakdown.csv", index=False)

    importance_frames = []
    fit = data.loc[data["date"] < HOLDOUT_START]
    evaluation = data.loc[data["date"] >= HOLDOUT_START]
    for model_name, model in holdout_models.items():
        include_signals = model_name.endswith("estimated_signals")
        x_fit, _, _, _ = build_feature_matrices(fit, evaluation, include_signals)
        importance_frames.append(
            extract_importance(model, x_fit.columns.tolist(), model_name)
        )
    importance = pd.concat(importance_frames, ignore_index=True)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    december_predictions, december_forecasts = generate_december_predictions(
        data, best_parameters
    )
    december_predictions.to_csv(
        OUTPUT_DIR / "december_predictions.csv", index=False
    )
    december_forecasts.to_csv(
        OUTPUT_DIR / "december_estimated_signals.csv", index=False
    )

    print("\nHoldout comparison (September–October, real metrics):")
    print(
        holdout[["model", "MAE", "RMSE", "MAPE_pct", "RMSLE"]]
        .round(4)
        .to_string(index=False)
    )
    print("\nCV metrics:")
    print(
        cv.groupby("model")[["MAE", "RMSE", "MAPE_pct", "RMSLE"]]
        .mean()
        .round(4)
        .to_string()
    )
    print("\nHoldout error breakdown:")
    print(
        breakdown[breakdown["breakdown"].eq("equipment")][
            ["model", "group", "n", "MAE", "RMSE", "MAPE_pct"]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\nTop features by variant:")
    for model_name, group in importance.groupby("model"):
        print(f"\n{model_name}")
        print(
            group[["feature", "importance", "importance_share"]]
            .head(12)
            .round(4)
            .to_string(index=False)
        )
    print("\nDecember forecast signals (first/last five days):")
    print(
        december_predictions[
            ["date", "estimated_market_index", "estimated_quote_signal"]
        ]
        .iloc[np.r_[0:5, -5:0]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nSaved Stage 3 outputs to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
