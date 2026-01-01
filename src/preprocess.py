import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/sample_sales.csv")
PROCESSED_DATA_PATH = Path("data/processed/processed_sales.csv")

def preprocess():
    df = pd.read_csv(RAW_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Simple feature engineering
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_7"] = df["sales"].rolling(7).mean()

    df = df.dropna()
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()

