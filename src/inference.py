import mlflow.pyfunc
import pandas as pd

# ========================
# LOAD MODEL PRODUCTION
# ========================
model = mlflow.pyfunc.load_model(
    "models:/btc-price-model/Production"
)

# ========================
# LOAD DATA
# ========================
df = pd.read_csv("data/processed/btc_features.csv")

# pisahin feature & target
X = df.drop(columns=["target", "datetime_utc"])
y = df["target"]

# ambil 5 data terakhir
sample_X = X.tail(5)
sample_y = y.tail(5)

# ========================
# PREDICT
# ========================
pred = model.predict(sample_X)

# ========================
# OUTPUT
# ========================
print("Sample Input:")
print(sample_X)

print("\nActual (Harga Asli Next Step):")
print(sample_y.values)

print("\nPrediction:")
print(pred)