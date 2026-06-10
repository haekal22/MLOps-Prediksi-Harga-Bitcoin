import pandas as pd
import numpy as np
import os
import glob


list_file = glob.glob("data/raw/*.csv")

if not list_file:
    raise FileNotFoundError("Tidak ada file di folder data/raw")

print(f"Jumlah file ditemukan: {len(list_file)}")

df_list = [pd.read_csv(f) for f in list_file]
df = pd.concat(df_list, ignore_index=True)

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

df = df.sort_values("datetime_utc")

df = df.drop_duplicates(subset=["datetime_utc"])

df = df.dropna()

df = df.reset_index(drop=True)


df["price_change"] = df["price_usd"].pct_change()

df["ma_24"] = df["price_usd"].rolling(window=24).mean()

# Harga historis (lag feature)
df["lag_1"] = df["price_usd"].shift(1)
df["lag_24"] = df["price_usd"].shift(24)


df["target"] = df["price_usd"].shift(-1)

df = df.dropna().reset_index(drop=True)

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/btc_features.csv"
df.to_csv(output_file, index=False)

print("\nPreprocessing + Feature Engineering selesai")
print(df.head())
print(f"\nJumlah data akhir: {len(df)}")
print(f"File hasil disimpan di: {output_file}")