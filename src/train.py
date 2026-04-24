import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# ========================
# LOAD DATA
# ========================
df = pd.read_csv("data/processed/btc_features.csv")

# ========================
# SPLIT FEATURE & TARGET
# ========================
X = df.drop(columns=["target", "datetime_utc"])
y = df["target"]

# ========================
# TRAIN TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ========================
# EXPERIMENT PARAMETER
# ========================
n_estimators = 100
max_depth = 5
learning_rate = 0.1

# ========================
# MLFLOW START
# ========================
mlflow.set_experiment("BTC Prediction")

with mlflow.start_run():

    # LOG PARAM
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("learning_rate", learning_rate)

    # MODEL
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    # PREDICT
    y_pred = model.predict(X_test)

    # METRIC
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # LOG METRIC
    mlflow.log_metric("rmse", rmse)

    # LOG MODEL
    mlflow.xgboost.log_model(model, "model")

    print(f"RMSE: {rmse}")