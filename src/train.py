import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = Path("data/processed/processed_sales.csv")
MODEL_PATH = Path("models/demand_model.pkl")

def train():
    df = pd.read_csv(DATA_PATH)

    X = df[["lag_1", "lag_7", "rolling_7"]]
    y = df["sales"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"Training MAE: {mae:.2f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()

