import pandas as pd
import os
import glob

list_file = glob.glob("data/raw/*.csv")

if not list_file:
    raise FileNotFoundError("Tidak ada file di folder data/raw")

latest_file = max(list_file, key=os.path.getctime)

print(f"Menggunakan file: {latest_file}")

df = pd.read_csv(latest_file)

df["datetime_wib"] = pd.to_datetime(df["datetime_wib"])

df = df.sort_values("datetime_wib")

df = df.drop_duplicates()

df = df.dropna()

df["price_change"] = df["price_usd"].diff()
df["volume_change"] = df["volume_usd"].diff()

df["target_price_next_day"] = df["price_usd"].shift(-1)

df = df.dropna()

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/btc_clean.csv"
df.to_csv(output_file, index=False)

print("\nPreprocessing selesai")
print(df.head())
print(f"\nFile hasil disimpan di: {output_file}")