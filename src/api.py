from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# ========================
# LOAD MODEL
# ========================
model = mlflow.pyfunc.load_model(
    "/app/mlruns/1/fa16683048bd4531a8757f864c141fcf/artifacts/best_model"
)

@app.get("/")
def home():
    return {
        "message": "BTC Prediction API Running"
    }


@app.get("/predict")
def predict():

    # load latest data
    df = pd.read_csv(
        "data/processed/btc_features.csv"
    )

    X = df.drop(
        columns=["target", "datetime_utc"]
    )

    latest_data = X.tail(1)

    prediction = model.predict(latest_data)

    return {
        "predicted_btc_price":
        float(prediction[0])
    }