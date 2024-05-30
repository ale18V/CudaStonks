import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'GOOG']

end_date = datetime.now()
start_date = end_date - timedelta(days=729)
DATA_DIR = "data"


def fetch_and_save_stock_data(ticker, start_date, end_date, interval):
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    file_name = os.path.join(DATA_DIR, f'{ticker}_{interval}_data.csv')
    stock_data.to_csv(file_name)
    print(f'Saved {ticker} data to {file_name}')


for ticker in tickers:
    fetch_and_save_stock_data(ticker, start_date, end_date, '1h')
