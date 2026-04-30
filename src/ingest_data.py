import requests
import pandas as pd
from datetime import datetime, timezone
import os

# pastikan folder ada
os.makedirs("data/raw", exist_ok=True)

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

params = {
    "vs_currency": "usd",
    "days": "90",
    "interval": "hourly"
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    rows = []

    for i in range(len(data["prices"])):
        timestamp_ms = data["prices"][i][0]

        price = data["prices"][i][1]
        market_cap = data["market_caps"][i][1]
        volume = data["total_volumes"][i][1]

        timestamp = timestamp_ms / 1000
        date_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)

        rows.append({
            "datetime_utc": date_utc,
            "price_usd": price,
            "market_cap_usd": market_cap,
            "volume_usd": volume
        })

    df = pd.DataFrame(rows)

    # 🔥 penting untuk time series
    df = df.sort_values("datetime_utc")
    df = df.drop_duplicates(subset=["datetime_utc"])
    df = df.reset_index(drop=True)

    # convert ke string biar konsisten
    df["datetime_utc"] = df["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")

    filename = datetime.now().strftime(
        "data/raw/btc_market_%Y%m%d_%H%M%S.csv"
    )

    df.to_csv(filename, index=False)

    print("Data berhasil diambil dari API")
    print(df.head())
    print(f"\nJumlah data: {len(df)}")
    print(f"File disimpan di: {filename}")

except requests.exceptions.RequestException as e:
    print(f"Gagal mengambil data dari API: {e}")
except Exception as e:
    print(f"Terjadi error: {e}")