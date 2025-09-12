import akshare as ak
import pandas as pd
import requests
stock_individual_basic_info_us_xq_df = ak.stock_individual_basic_info_us_xq("NVDA")
print(stock_individual_basic_info_us_xq_df)