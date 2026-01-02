import os
import pandas as pd

RAW_PATH = "data/raw/m5_demand.csv"
OUT_PATH = "data/processed/processed.csv"

def main():
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(
            f"Missing {RAW_PATH}. Put your raw data there first."
        )

    df = pd.read_csv(RAW_PATH)

    # Expect columns: date,demand,store_id,item_id
    required = {"date", "demand", "store_id", "item_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clean + sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Basic feature engineering
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["lag_1"] = df.groupby(["store_id", "item_id"])["demand"].shift(1)
    df["lag_7"] = df.groupby(["store_id", "item_id"])["demand"].shift(7)
    df["roll_7"] = (
        df.groupby(["store_id", "item_id"])["demand"]
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    df = df.dropna().reset_index(drop=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Wrote processed file: {OUT_PATH} (rows={len(df)})")

if __name__ == "__main__":
    main()

