import pandas as pd
import yfinance as yf
from stock_technical_indicators import StockTechnicalIndicators

class DataProcessor:
    def __init__(self, stock_ticker, period):
        self.stock = stock_ticker
        self.period = period
        self.data = None
        self.cont_data_frame = None
        self.binary_data_frame = None
    
    def data_loader(self):
        stock = self.stock
        period = self.period
        try:
            cont_data_frame = pd.read_csv(f'data/{stock}_{period}_data_cont.csv')
            binary_data_frame = pd.read_csv(f'data/{stock}_{period}_data_binary.csv')
            print(f'Loaded Processed {stock}_{period}_ data from file')
        except FileNotFoundError:
            try:
                stock_data = pd.read_csv(f'data/{stock}_{period}_data.csv')
                print(f'Loaded {stock}_{period}_ data from file')
            except FileNotFoundError:
                print(f'Loading {stock}_{period}_ data from Yahoo Finance')
                stock_data = yf.Ticker(stock).history(period=period, interval='1d')
                stock_data.to_csv(f'data/{stock}_{period}_data.csv')
                print(f'Loaded {stock}_{period}_ data from Yahoo Finance') 
            sti = StockTechnicalIndicators(stock_data)
            cont_data_frame, binary_data_frame = sti.get_dataframes(days=30, interval=1)
            cont_data_frame.to_csv(f'data/{stock}_{period}_data_cont.csv', index=False)
            binary_data_frame.to_csv(f'data/{stock}_{period}_data_binary.csv', index=False)
            print(f'Saved Processed {stock}_{period}_ data to file')