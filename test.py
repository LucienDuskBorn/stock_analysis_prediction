from yahoofinancials import YahooFinancials

# 初始化对象
yf = YahooFinancials('AAPL')
start_date = "2025-06-01"
end_date = "2025-06-20"
# 获取历史收盘价
historical_data = yf.get_historical_price_data(start_date, end_date, time_interval='daily')
print(historical_data)
