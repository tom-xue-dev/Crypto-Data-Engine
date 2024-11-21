from dotenv import load_dotenv
import os
from binance.client import Client

load_dotenv()  # read from local .env file
# Test
api_key = os.environ["BINANCE_API_KEY_TEST"]
api_secret = os.environ["BINANCE_API_SECRET_TEST"]
client = Client(api_key, api_secret, testnet=True)

# Live trading
# api_key = os.environ['BINANCE_API_KEY_LIVE']
# api_secret = os.environ['BINANCE_API_SECRET_LIVE']
# client = Client(api_key, api_secret)

buy_order = client.create_test_order(
    symbol="BTCUSDT", side="BUY", type="MARKET", quantity=0.05
)
