import akshare as ak

stock_news_em_df = ak.stock_news_em(symbol="06936")
stock_news_em_df.to_csv("C:\\D\\pythonWork\\stock_analysis_prediction\\data_cache\stock_news_em_df.csv")
print(stock_news_em_df)