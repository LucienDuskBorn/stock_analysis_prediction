import akshare as ak

market_cap_pe = ak.stock_us_famous_spot_em(symbol='科技类')
for index,row in market_cap_pe.iterrows():
    if row['代码'].find('NVDA')!= -1:
        print(f"代码:{row[11]}")
        market = row[9]
        print(f"market:{market}")
        market_pe =  row[10]
        print(f"市盈率:{market_pe}")
market_cap_pe.to_csv("C:\\D\\pythonWork\\send_email\\stock.csv",encoding="GBK")
print(market_cap_pe)