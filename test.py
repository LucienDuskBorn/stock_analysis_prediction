import akshare as ak

stock_us_hist_df = ak.stock_us_hist(symbol='MSFT', period="daily", start_date="20250901", end_date="20240910", adjust="qfq")
print(stock_us_hist_df)