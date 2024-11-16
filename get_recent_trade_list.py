import pandas as pd
import json
import websocket
from dotenv import load_dotenv
load_dotenv() # read from local .env file

import os
from binance.client import Client

# Test
api_key = os.environ['BINANCE_API_KEY_TEST']
api_secret = os.environ['BINANCE_API_SECRET_TEST']
client = Client(api_key, api_secret, testnet=True)

# Live trading
#api_key = os.environ['BINANCE_API_KEY_LIVE']
#api_secret = os.environ['BINANCE_API_SECRET_LIVE']
#client = Client(api_key, api_secret)


symbol = 'btcusdt'
socket = f'wss://stream.binance.com:9443/ws/{symbol}@trade'

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

ws = websocket.WebSocketApp(socket,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

ws.run_forever()