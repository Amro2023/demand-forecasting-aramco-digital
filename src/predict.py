import os
import argparse
import pandas as pd
import numpy as np
from joblib import load

PROCESSED_PATH = os.path.join("data", "processed", "processed.csv")
MODEL_PATH = os.path.join("models", "model.joblib")
OUT_PATH = os.path.join("reports", "forecast.csv")

FEATURES = [
    "dow", "week", "month", "year",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_7", "roll_28"
]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df

def forecast_one_group(model, hist: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # hist contains demand history for a single (store_id, item_id) group
    hist = hist.sort_values("date").copy()
    last_date = hist["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # We'll iteratively simulate demand to compute lag features
    sim = hist[["date", "demand"]].copy()

    rows = []
    for d in future_dates:
        row = {"date": d}

        # compute lags from sim
        for lag in (1, 7, 14, 28):
            lag_date = d - pd.Timedelta(days=lag)
            val = sim.loc[sim["date"] == lag_date, "demand"]
            row[f"lag_{lag}"] = float(val.iloc[0]) if len(val) else 0.0

        # rolling means based on previous days in sim
        past = sim[sim["date"] < d].sort_values("date")
        row["roll_7"] = float(past["demand"].tail(7).mean()) if len(past) else 0.0
        row["roll_28"] = float(past["demand"].tail(28).mean()) if len(past) else 0.0

        rows.append(row)

        tmp = pd.DataFrame([row])
        tmp = add_time_features(tmp)

        X = tmp[FEATURES]
        yhat = float(model.predict(X)[0])

        # no negatives
        yhat = max(0.0, yhat)

        # append simulated demand so future lags work
        sim = pd.concat([sim, pd.DataFrame([{"date": d, "demand": yhat}])], ignore_index=True)

    out = pd.DataFrame(rows)
    out = add_time_features(out)
    out["forecast"] = sim[sim["date"].isin(future_dates)]["demand"].values
    return out[["date", "forecast"]]

def main(horizon: int):
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError("Run preprocess first: python src/preprocess.py")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train model first: python src/train.py")

    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    model = load(MODEL_PATH)

    results = []
    for (store_id, item_id), g in df.groupby(["store_id", "item_id"]):
        f = forecast_one_group(model, g[["date", "demand"]], horizon=horizon)
        f["store_id"] = store_id
        f["item_id"] = item_id
        results.append(f)

    forecast = pd.concat(results, ignore_index=True)
    forecast.to_csv(OUT_PATH, index=False)

    print(f"✅ Forecast written: {OUT_PATH}")
    print("Preview:")
    print(forecast.head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    main(args.horizon)

