
## 3) `src/preprocess.py`
```python
import os
import glob
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join("data", "processed", "processed.csv")

def load_raw_csv() -> pd.DataFrame:
    files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not files:
        raise FileNotFoundError(
            "No CSV found in data/raw/. Add a raw dataset first (e.g., data/raw/demand.csv)."
        )
    # Take the first CSV for simplicity
    df = pd.read_csv(files[0])

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Required columns
    if "date" not in df.columns or "demand" not in df.columns:
        raise ValueError("Raw CSV must contain columns: date, demand")

    # Optional dimensions
    if "store_id" not in df.columns:
        df["store_id"] = "ALL"
    if "item_id" not in df.columns:
        df["item_id"] = "ALL"

    df["date"] = pd.to_datetime(df["date"])
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    df = df.dropna(subset=["date", "demand"])
    return df

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Daily aggregation per store/item
    daily = (
        df.groupby(["store_id", "item_id", "date"], as_index=False)["demand"]
        .sum()
        .sort_values(["store_id", "item_id", "date"])
    )
    return daily

def ensure_date_continuity(daily: pd.DataFrame) -> pd.DataFrame:
    # Fill missing dates per store/item with demand=0 (simple baseline assumption)
    out = []
    for (store_id, item_id), g in daily.groupby(["store_id", "item_id"]):
        g = g.sort_values("date")
        all_days = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        gg = g.set_index("date").reindex(all_days).fillna({"demand": 0})
        gg = gg.rename_axis("date").reset_index()
        gg["store_id"] = store_id
        gg["item_id"] = item_id
        out.append(gg)
    return pd.concat(out, ignore_index=True)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    return df

def add_lag_features(df: pd.DataFrame, lags=(1, 7, 14, 28)) -> pd.DataFrame:
    df = df.sort_values(["store_id", "item_id", "date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["demand"].shift(lag)
    # Rolling means
    df["roll_7"] = df.groupby(["store_id", "item_id"])["demand"].shift(1).rolling(7).mean()
    df["roll_28"] = df.groupby(["store_id", "item_id"])["demand"].shift(1).rolling(28).mean()
    return df

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    raw = load_raw_csv()
    daily = aggregate_daily(raw)
    daily = ensure_date_continuity(daily)
    daily = add_time_features(daily)
    daily = add_lag_features(daily)

    # Drop early rows where lags are missing
    daily = daily.dropna()

    daily.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote processed dataset: {OUT_PATH}")
    print("Shape:", daily.shape)
    print("Columns:", list(daily.columns))

if __name__ == "__main__":
    main()
