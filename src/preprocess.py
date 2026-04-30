import pandas as pd
import numpy as np
import os
import glob

# ========================
# LOAD SEMUA DATA
# ========================
list_file = glob.glob("data/raw/*.csv")

if not list_file:
    raise FileNotFoundError("Tidak ada file di folder data/raw")

print(f"Jumlah file ditemukan: {len(list_file)}")

df_list = [pd.read_csv(f) for f in list_file]
df = pd.concat(df_list, ignore_index=True)

# ========================
# CLEANING
# ========================
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

df = df.sort_values("datetime_utc")

df = df.drop_duplicates(subset=["datetime_utc"])

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
df["ma_6"] = df["price_usd"].rolling(window=6).mean()
df["ma_24"] = df["price_usd"].rolling(window=24).mean()

# --- EXPONENTIAL MOVING AVERAGE ---
df["ema_6"] = df["price_usd"].ewm(span=6).mean()
df["ema_24"] = df["price_usd"].ewm(span=24).mean()

# --- VOLATILITY ---
df["volatility_6"] = df["price_usd"].rolling(window=6).std()
df["volatility_24"] = df["price_usd"].rolling(window=24).std()

# --- MOMENTUM ---
df["momentum_3"] = df["price_usd"] - df["price_usd"].shift(3)
df["momentum_6"] = df["price_usd"] - df["price_usd"].shift(6)
df["momentum_24"] = df["price_usd"] - df["price_usd"].shift(24)

# --- LAG FEATURES ---
df["lag_1"] = df["price_usd"].shift(1)
df["lag_2"] = df["price_usd"].shift(2)
df["lag_24"] = df["price_usd"].shift(24)

# --- PRICE VS MA ---
df["price_ma6_diff"] = df["price_usd"] - df["ma_6"]
df["price_ma24_diff"] = df["price_usd"] - df["ma_24"]

# --- RSI (Relative Strength Index) ---
delta = df["price_usd"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
df["rsi"] = 100 - (100 / (1 + rs))

# --- TIME FEATURES ---
df["hour"] = df["datetime_utc"].dt.hour
df["day"] = df["datetime_utc"].dt.day
df["month"] = df["datetime_utc"].dt.month
df["weekday"] = df["datetime_utc"].dt.weekday

# --- CYCLICAL TIME (IMPORTANT) ---
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# --- RATIO FEATURE ---
df["price_to_ma24"] = df["price_usd"] / df["ma_24"]

# --- TARGET (NEXT STEP) ---
df["target"] = df["price_usd"].shift(-1)

# ========================
# HANDLE MISSING
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
print(f"\nJumlah data akhir: {len(df)}")
print(f"File hasil disimpan di: {output_file}")