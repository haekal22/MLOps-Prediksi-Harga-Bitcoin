import pandas as pd
import mlflow
import mlflow.xgboost
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from mlflow.tracking import MlflowClient

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("data/processed/btc_features.csv")

X = df.drop(columns=["target", "datetime_utc"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ==========================================
# 2. HYPERPARAMETERS
# ==========================================
n_estimators = 1000
max_depth = 1
learning_rate = 0.05

mlflow.set_experiment("BTC Prediction")
MODEL_NAME = "btc-price-model"
THRESHOLD = 800 
client = MlflowClient()

# ==========================================
# 3. TRAINING & MLFLOW LOGGING
# ==========================================
with mlflow.start_run() as run:

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

    # Log metrik ke MLflow Server
    mlflow.log_metric("rmse", rmse)

    # Log model pake fungsi asli bawaan kamu (Aman karena di-bypass dari YAML)
    mlflow.xgboost.log_model(model, "model")

    print(f"BEST RMSE: {rmse}")