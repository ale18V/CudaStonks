import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

tickers = [
    ('SP500', '^GSPC'),
    ('NASDAQ100', '^IXIC'),
    ('RUSSELL2000', '^RUT')
]

end_date = datetime.now()
start_date = end_date - timedelta(days=729)
DATA_DIR = "data"


def fetch_and_save_stock_data(file_name: str, ticker_name: str, interval: str):
    ticker = yf.Ticker(ticker_name)
    stock_data = ticker.history(start=datetime(year=1990, month=1, day=1), end=datetime(
        year=2024, month=1, day=1), interval=interval, actions=False)
    file_name = os.path.join(DATA_DIR, f'{file_name}_{interval}_data.csv')
    stock_data.to_csv(file_name)
    print(f'Saved {ticker_name} data to {file_name}')


for ticker in tickers:
    fetch_and_save_stock_data(ticker[0], ticker[1], '1d')
