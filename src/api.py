from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# ========================
# LOAD MODEL
# ========================
model = mlflow.pyfunc.load_model(
    "/app/mlruns/1/7322ba02129e43ba8944de595794176e/artifacts/model"
)


# ========================
# INPUT SCHEMA
# ========================
class PredictionInput(BaseModel):
    price_usd: float
    market_cap_usd: float
    volume_usd: float


@app.get("/")
def home():
    return {
        "message": "BTC Prediction API Running"
    }


# ========================
# PREDICT
# ========================
@app.post("/predict")
def predict(data: PredictionInput):

    # load historical data
    history = pd.read_csv(
        "data/processed/btc_features.csv"
    )

    # ========================
    # AUTO FEATURE ENGINEERING
    # ========================
    lag_1 = history["price_usd"].iloc[-1]

    lag_24 = history["price_usd"].iloc[-24]

    ma_24 = (
        history["price_usd"]
        .tail(24)
        .mean()
    )

    price_change = (
        data.price_usd - lag_1
    ) / lag_1

    # ========================
    # BUILD INPUT
    # ========================
    input_df = pd.DataFrame([{
        "price_usd":
            data.price_usd,

        "market_cap_usd":
            data.market_cap_usd,

        "volume_usd":
            data.volume_usd,

        "price_change":
            price_change,

        "ma_24":
            ma_24,

        "lag_1":
            lag_1,

        "lag_24":
            lag_24
    }])

    prediction = model.predict(
        input_df
    )

    return {
        "predicted_btc_price":
            float(prediction[0])
    }