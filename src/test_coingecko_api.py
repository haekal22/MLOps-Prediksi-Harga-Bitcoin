import requests

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

params = {
    "vs_currency": "usd",
    "days": "1"
}

response = requests.get(url, params=params)

data = response.json()

print("Sample price data:")
print(data["prices"][:5])