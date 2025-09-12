import requests

url = "https://api.itick.org/stock/kline?region=US&code=AAPL&kType=8&et=20250910&limit=10"

headers = {
"accept": "application/json",
"token": "98096e30dddc41eb99fc14feca2b7a51f5f2e7822f804a51801d47e6f30df751"
}

response = requests.get(url, headers=headers)

print(response.text)

