import os
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROCESSED_PATH = os.path.join("data", "processed", "processed.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

FEATURES = [
    "dow", "week", "month", "year",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_7", "roll_28"
]

def time_split(df: pd.DataFrame, test_days: int = 28):
    # last N days as test
    df = df.sort_values("date")
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test

def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError("Run preprocess first: python src/preprocess.py")

    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])

    # Simple baseline: train one global model across all store/item rows
    train_df, test_df = time_split(df, test_days=28)

    X_train = train_df[FEATURES]
    y_train = train_df["demand"]
    X_test = test_df[FEATURES]
    y_test = test_df["demand"]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    dump(model, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")
    print(f"📉 Holdout MAE (last 28 days): {mae:.3f}")
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

if __name__ == "__main__":
    main()
