# Demand Forecasting (Aramco Digital-style demo)

End-to-end demand forecasting repo:
- preprocess raw sales/demand data
- train a simple ML model with time + lag features
- generate future forecasts
- serve forecasts via FastAPI

## Quickstart (Mac/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Put a CSV into data/raw/ (see schema below)
# 2) Preprocess -> Train -> Predict
python src/preprocess.py
python src/train.py
python src/predict.py --horizon 28

# API
uvicorn api.main:app --reload
