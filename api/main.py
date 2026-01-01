import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

FORECAST_PATH = os.path.join("reports", "forecast.csv")

app = FastAPI(title="Demand Forecasting API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/forecast")
def get_forecast(
    store_id: str = Query(default="ALL"),
    item_id: str = Query(default="ALL"),
    limit: int = Query(default=28, ge=1, le=365)
):
    if not os.path.exists(FORECAST_PATH):
        raise HTTPException(status_code=404, detail="Forecast file missing. Run: python src/predict.py")

    df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    df = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date").head(limit)

    if df.empty:
        raise HTTPException(status_code=404, detail="No forecast found for that store_id/item_id")

    return {
        "store_id": store_id,
        "item_id": item_id,
        "rows": [{"date": d.date().isoformat(), "forecast": float(v)} for d, v in zip(df["date"], df["forecast"])]
    }

