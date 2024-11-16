from Account import Account
from get_btc_info import get_btc_data
account = Account(10000,"BTC")
df = get_btc_data("1s")
print(df.columns)
for row in df.itertuples():
    print(row)