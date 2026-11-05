from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train-test.csv"
VALIDATION_PATH = ROOT / "validation.csv"
OUTPUT_DIR = ROOT / "outputs" / "eda"


def haversine_miles(frame: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(frame["pickup_lat"].to_numpy())
    lat2 = np.radians(frame["delivery_lat"].to_numpy())
    delta_lat = lat2 - lat1
    delta_lon = np.radians(
        frame["delivery_lon"].to_numpy() - frame["pickup_lon"].to_numpy()
    )
    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    return 3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def inspect_dataset(name: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {name.upper()} OVERVIEW ===")
    print(f"Shape: {frame.shape}")
    print(f"Date range: {frame['date'].min()} to {frame['date'].max()}")
    print("Dtypes:")
    print(frame.dtypes.astype(str).to_string())
    missing = frame.isna().sum()
    print("Missing values:")
    print(missing[missing.gt(0)].to_string() if missing.any() else "None")

    print("\nDuplicate and domain checks:")
    print(f"Exact duplicate rows: {frame.duplicated().sum()}")
    print(f"Duplicate load IDs: {frame['load_id'].duplicated().sum()}")
    print(f"Pickup equals delivery: {frame['pickup'].eq(frame['delivery']).sum()}")
    for column in ("distance", "weight", "market_index", "quote_signal", "posted_rate"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        print(
            f"{column}: negative={(values < 0).sum()}, "
            f"zero={(values == 0).sum()}, missing={values.isna().sum()}"
        )
    print(f"Unique pickup cities: {frame['pickup'].nunique()}")
    print(f"Unique delivery cities: {frame['delivery'].nunique()}")
    print("Equipment counts:")
    print(frame["equipment"].value_counts(dropna=False).to_string())


def inspect_geography(train: pd.DataFrame) -> None:
    train["haversine_miles"] = haversine_miles(train)
    train["distance_error_miles"] = train["distance"] - train["haversine_miles"]
    train["distance_ratio"] = train["distance"] / train["haversine_miles"].replace(
        0, np.nan
    )
    print("\n=== GEOGRAPHY ===")
    print(
        "Distance/haversine correlation:",
        train[["distance", "haversine_miles"]].corr().iloc[0, 1],
    )
    print(
        "Absolute distance error quantiles:",
        train["distance_error_miles"].abs().quantile(
            [0.5, 0.9, 0.95, 0.99, 1]
        ).round(2).to_dict(),
    )
    print(
        "Distance/haversine ratio quantiles:",
        train["distance_ratio"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        .round(3)
        .to_dict(),
    )
    print(
        "Rows with absolute distance error > 25 miles:",
        f"{train['distance_error_miles'].abs().gt(25).sum()} / {len(train)}",
    )

    coordinate_frames = []
    for role, city, lat, lon in (
        ("pickup", "pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery", "delivery_lat", "delivery_lon"),
    ):
        part = train[[city, lat, lon]].rename(
            columns={city: "city", lat: "lat", lon: "lon"}
        )
        part["role"] = role
        coordinate_frames.append(part)
    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    coordinates["coordinate"] = list(
        zip(coordinates["lat"].round(5), coordinates["lon"].round(5))
    )
    city_variants = coordinates.groupby("city")["coordinate"].nunique()
    print(
        "Cities with multiple coordinate pairs:",
        f"{city_variants.gt(1).sum()} / {len(city_variants)}",
    )
    for role in ("pickup", "delivery"):
        variants = coordinates.loc[coordinates["role"].eq(role)].groupby("city")[
            "coordinate"
        ].nunique()
        print(
            f"{role.title()} cities with multiple coordinate pairs:",
            f"{variants.gt(1).sum()} / {len(variants)}",
        )


def inspect_target(train: pd.DataFrame) -> None:
    target = train["posted_rate"]
    train["rate_per_mile"] = target / train["distance"].replace(0, np.nan)
    q1, q3 = target.quantile([0.25, 0.75])
    upper_fence = q3 + 1.5 * (q3 - q1)
    print("\n=== POSTED RATE ===")
    print(
        target.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .round(2)
        .to_dict()
    )
    print(f"Skew: {target.skew():.3f}")
    print(f"log1p skew: {np.log1p(target.clip(lower=0)).skew():.3f}")
    print(
        f"IQR outliers: {((target < q1 - 1.5 * (q3 - q1)) | (target > upper_fence)).sum()} "
        f"(upper fence: {upper_fence:.2f})"
    )
    print("Rate per mile summary:")
    print(
        train["rate_per_mile"]
        .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        .round(3)
        .to_dict()
    )
    print("Rate per mile by equipment:")
    print(
        train.groupby("equipment")["rate_per_mile"]
        .agg(["count", "mean", "median", "std"])
        .round(3)
        .to_string()
    )
    for column in ("distance", "weight"):
        print(
            f"Rate vs {column}: Pearson="
            f"{train[['posted_rate', column]].corr().iloc[0, 1]:.3f}, "
            f"Spearman="
            f"{train[['posted_rate', column]].corr(method='spearman').iloc[0, 1]:.3f}"
        )


def inspect_time_and_drift(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    train["week"] = train["date"].dt.to_period("W").dt.start_time
    train["month"] = train["date"].dt.to_period("M").dt.start_time
    print("\n=== WEEKLY TRENDS ===")
    weekly = train.groupby("week").agg(
        rows=("posted_rate", "size"),
        mean_rate=("posted_rate", "mean"),
        mean_rate_per_mile=("rate_per_mile", "mean"),
        market_index=("market_index", "mean"),
        quote_signal=("quote_signal", "mean"),
    )
    print(pd.concat([weekly.head(4), weekly.tail(4)]).round(3).to_string())

    print("\n=== MONTHLY TRENDS ===")
    monthly = train.groupby("month").agg(
        rows=("posted_rate", "size"),
        mean_rate=("posted_rate", "mean"),
        mean_rate_per_mile=("rate_per_mile", "mean"),
        market_index=("market_index", "mean"),
        quote_signal=("quote_signal", "mean"),
    )
    print(monthly.round(3).to_string())
    for column in ("market_index", "quote_signal"):
        sample = train[[column, "posted_rate"]].dropna()
        print(
            f"{column} vs posted_rate: Pearson="
            f"{sample.corr().iloc[0, 1]:.3f}, Spearman="
            f"{sample.corr(method='spearman').iloc[0, 1]:.3f}, "
            f"n={len(sample)}"
        )

    print("\n=== TRAIN / VALIDATION DRIFT ===")
    numeric_columns = (
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
    )
    smd_values = {}
    for column in numeric_columns:
        train_values = train[column].astype(float)
        valid_values = validation[column].astype(float)
        pooled_std = np.sqrt((train_values.std() ** 2 + valid_values.std() ** 2) / 2)
        smd = (
            (valid_values.mean() - train_values.mean()) / pooled_std
            if pooled_std
            else np.nan
        )
        smd_values[column] = smd
        print(
            f"{column}: train mean={train_values.mean():.3f}, "
            f"valid mean={valid_values.mean():.3f}, "
            f"train missing={train_values.isna().mean():.3%}, "
            f"valid missing={valid_values.isna().mean():.3%}, SMD={smd:.3f}"
        )

    for column in ("pickup", "delivery", "equipment"):
        train_share = train[column].value_counts(normalize=True)
        valid_share = validation[column].value_counts(normalize=True)
        categories = set(train_share.index) | set(valid_share.index)
        total_variation = 0.5 * sum(
            abs(train_share.get(value, 0) - valid_share.get(value, 0))
            for value in categories
        )
        unseen = sorted(set(validation[column].dropna()) - set(train[column].dropna()))
        print(
            f"{column}: train unique={train[column].nunique()}, "
            f"validation unique={validation[column].nunique()}, "
            f"total variation={total_variation:.4f}, unseen={unseen}"
        )
    return monthly, smd_values


def save_plots(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    monthly: pd.DataFrame,
    smd_values: dict[str, float],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(train["posted_rate"], bins=65, color="#286a8a")
    axes[0].set(title="Posted rate (original scale)", xlabel="Posted rate", ylabel="Loads")
    axes[1].hist(np.log1p(train["posted_rate"]), bins=65, color="#d27a2c")
    axes[1].set(
        title="Posted rate (log1p scale)",
        xlabel="log1p(posted_rate)",
        ylabel="Loads",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "posted_rate_distribution.png", dpi=160)
    plt.close(figure)

    equipment_order = train.groupby("equipment")["rate_per_mile"].median().sort_values().index
    cap = train["rate_per_mile"].quantile(0.99)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.boxplot(
        [
            train.loc[train["equipment"].eq(value), "rate_per_mile"]
            .dropna()
            .clip(upper=cap)
            for value in equipment_order
        ],
        tick_labels=equipment_order,
        showfliers=False,
    )
    axis.set(
        title="Rate per mile by equipment (values capped at overall p99)",
        ylabel="Posted rate / distance",
    )
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rate_per_mile_by_equipment.png", dpi=160)
    plt.close(figure)

    sample = train.sample(min(len(train), 7000), random_state=17)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, column in zip(axes, ("distance", "weight")):
        axis.scatter(
            sample[column],
            sample["posted_rate"],
            s=7,
            alpha=0.22,
            color="#286a8a",
            edgecolors="none",
        )
        axis.set(xlabel=column, ylabel="Posted rate", title=f"Posted rate vs {column}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rate_vs_distance_weight.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.plot(monthly.index, monthly["mean_rate"], marker="o", label="Mean posted rate")
    axis.set_ylabel("Mean posted rate")
    second_axis = axis.twinx()
    second_axis.plot(
        monthly.index,
        monthly["mean_rate_per_mile"],
        color="#d27a2c",
        marker="s",
        label="Mean rate / mile",
    )
    second_axis.set_ylabel("Mean rate / mile")
    axis.set(title="Monthly rate trends", xlabel="Month")
    first_lines, first_labels = axis.get_legend_handles_labels()
    second_lines, second_labels = second_axis.get_legend_handles_labels()
    axis.legend(first_lines + second_lines, first_labels + second_labels)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "monthly_rate_trends.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.6), sharex=True)
    for axis, column, color in zip(
        axes,
        ("market_index", "quote_signal"),
        ("#42855b", "#8b5aa7"),
    ):
        values = train.groupby("week")[column].mean()
        axis.plot(values.index, values.values, color=color, marker=".", linewidth=1.5)
        axis.set_ylabel(column)
        axis.set_title(f"Weekly mean {column}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "weekly_market_quote_trends.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    sample_geo = train.sample(min(len(train), 7000), random_state=17)
    limit = max(
        float(train["distance"].quantile(0.995)),
        float(train["haversine_miles"].quantile(0.995)),
    )
    axis.scatter(
        sample_geo["haversine_miles"],
        sample_geo["distance"],
        s=7,
        alpha=0.25,
        color="#286a8a",
        edgecolors="none",
    )
    axis.plot([0, limit], [0, limit], color="#c44e52", linestyle="--", label="y=x")
    axis.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Haversine distance (miles)",
        ylabel="Recorded distance",
        title="Recorded distance vs straight-line distance",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "distance_vs_haversine.png", dpi=160)
    plt.close(figure)

    drift_columns = list(smd_values)
    drift = [smd_values[column] for column in drift_columns]
    colors = ["#c44e52" if abs(value) >= 0.2 else "#4c8c6b" for value in drift]
    figure, axis = plt.subplots(figsize=(8.5, 4.7))
    axis.barh(drift_columns, drift, color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(0.2, color="gray", linestyle="--", linewidth=0.8)
    axis.axvline(-0.2, color="gray", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="Standardized mean difference (validation - train)",
        title="Numeric feature drift",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "numeric_feature_drift.png", dpi=160)
    plt.close(figure)

    print("\n=== SAVED PLOTS ===")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path.relative_to(ROOT))


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")

    inspect_dataset("train", train)
    inspect_dataset("validation", validation)
    inspect_geography(train)
    inspect_target(train)
    monthly, smd_values = inspect_time_and_drift(train, validation)
    save_plots(train, validation, monthly, smd_values)


if __name__ == "__main__":
    main()
