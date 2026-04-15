import pandas as pd
import os
import glob

# ========================
# LOAD DATA TERBARU
# ========================
list_file = glob.glob("data/raw/*.csv")

if not list_file:
    raise FileNotFoundError("Tidak ada file di folder data/raw")

latest_file = max(list_file, key=os.path.getctime)

print(f"Menggunakan file: {latest_file}")

df = pd.read_csv(latest_file)

# ========================
# CLEANING
# ========================
df["datetime_wib"] = pd.to_datetime(df["datetime_wib"])

df = df.sort_values("datetime_wib")

df = df.drop_duplicates()

df = df.dropna()

df = df.reset_index(drop=True)

# ========================
# FEATURE ENGINEERING
# ========================

# --- CHANGE FEATURES ---
df["price_change"] = df["price_usd"].pct_change()
df["volume_change"] = df["volume_usd"].pct_change()
df["marketcap_change"] = df["market_cap_usd"].pct_change()

# --- MOVING AVERAGE ---
df["ma_3"] = df["price_usd"].rolling(window=3).mean()
df["ma_7"] = df["price_usd"].rolling(window=7).mean()

# --- VOLATILITY ---
df["volatility_3"] = df["price_usd"].rolling(window=3).std()

# --- LAG FEATURES ---
df["lag_1"] = df["price_usd"].shift(1)
df["lag_2"] = df["price_usd"].shift(2)

# --- TIME FEATURES ---
df["day"] = df["datetime_wib"].dt.day
df["month"] = df["datetime_wib"].dt.month
df["weekday"] = df["datetime_wib"].dt.weekday

# --- TARGET (PREDIKSI NEXT STEP) ---
df["target"] = df["price_usd"].shift(-1)

# ========================
# HANDLE MISSING (AKIBAT FEATURE)
# ========================
df = df.dropna().reset_index(drop=True)

# ========================
# SAVE DATA
# ========================
os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/btc_features.csv"
df.to_csv(output_file, index=False)

# ========================
# OUTPUT
# ========================
print("\nPreprocessing + Feature Engineering selesai")
print(df.head())
print(f"\nFile hasil disimpan di: {output_file}")