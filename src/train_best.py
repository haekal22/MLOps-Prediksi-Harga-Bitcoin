import pandas as pd
import mlflow
import mlflow.xgboost
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from mlflow.tracking import MlflowClient
# LOAD DATA
df = pd.read_csv("data/processed/btc_features.csv")

X = df.drop(columns=["target", "datetime_utc"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 🔥 BEST PARAMETER (hasil grid search)
n_estimators = 1000
max_depth = 1
learning_rate = 0.04

mlflow.set_experiment("BTC Prediction")
MODEL_NAME = "btc-price-model"
THRESHOLD = 550 
client = MlflowClient()

with mlflow.start_run():

    mlflow.log_param("model", "XGBoost-Best")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_metric("rmse", rmse)

    # log model
    mlflow.xgboost.log_model(model, "model")

    print(f"BEST RMSE: {rmse}")

    # =========================
    # TAHAP 5: MODEL REGISTRY
    # =========================
    if rmse < THRESHOLD:

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Staging"
        )

        print("✅ MODEL MASUK STAGING")

    else:
        print("❌ Model tidak lolos threshold")