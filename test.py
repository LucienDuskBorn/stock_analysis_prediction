import akshare as ak

stock_us_hist_df = ak.stock_us_hist(symbol='106.TTE', period="daily", start_date="20200101", end_date="20240214", adjust="qfq")
print(stock_us_hist_df)