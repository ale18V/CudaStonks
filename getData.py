import yfinance as yf
import pandas as pd
import datetime
import os

tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'GOOG']

end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=5*365)
DATA_DIR = "data"


def fetch_and_save_stock_data(ticker):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    file_name = os.path.join(DATA_DIR, f'{ticker}_5_years_stock_data.csv')
    stock_data.to_csv(file_name)
    print(f'Saved {ticker} data to {file_name}')


for ticker in tickers:
    fetch_and_save_stock_data(ticker)
