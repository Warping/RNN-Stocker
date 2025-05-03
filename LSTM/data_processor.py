import pandas as pd
import yfinance as yf
from stock_technical_indicators import StockTechnicalIndicators

class DataProcessor:
    def __init__(self):
        pass
    
    def data_load(self, stock, period):
        self.stock = stock
        self.period = period
        try:
            cont_data_frame = pd.read_csv(f'../data/{stock}_{period}_data_cont.csv')
            binary_data_frame = pd.read_csv(f'../data/{stock}_{period}_data_binary.csv')
            print(f'Loaded Processed {stock}_{period}_ data from file')
        except FileNotFoundError:
            try:
                stock_data = pd.read_csv(f'../data/{stock}_{period}_data.csv')
                print(f'Loaded {stock}_{period}_ data from file')
            except FileNotFoundError:
                print(f'Loading {stock}_{period}_ data from Yahoo Finance')
                stock_data = yf.Ticker(stock).history(period=period, interval='1d')
                stock_data.to_csv(f'../data/{stock}_{period}_data.csv')
                print(f'Loaded {stock}_{period}_ data from Yahoo Finance') 
            sti = StockTechnicalIndicators(stock_data)
            cont_data_frame, binary_data_frame = sti.get_dataframes(days=30, interval=1)
            cont_data_frame.to_csv(f'../data/{stock}_{period}_data_cont.csv', index=False)
            binary_data_frame.to_csv(f'../data/{stock}_{period}_data_binary.csv', index=False)
            print(f'Saved Processed {stock}_{period}_ data to file')
        self.cont_data_frame = cont_data_frame
        self.binary_data_frame = binary_data_frame
        
    def drop_columns(self, drop_columns):
        if isinstance(drop_columns, str):
            drop_columns = drop_columns.strip('[ ]').split(',')
        if drop_columns != []:
            print(f'Dropping columns: {drop_columns}')
            # Drop columns from both data frames
            self.binary_data_frame = self.binary_data_frame.drop(columns=drop_columns)
            # Drop columns from continuous data frame
            self.cont_data_frame = self.cont_data_frame.drop(columns=drop_columns)
        else:
            print(f'No columns to drop')
            
    def data_normalize(self, avg_period, smoothing_window):
        self.features = len(self.cont_data_frame.columns)

        # Apply gaussian filter to smooth data
        # Apply rolling mean to smooth data
        self.cont_data_frame = self.cont_data_frame.rolling(window=smoothing_window, min_periods=1).mean()
        # cont_data_frame = cont_data_frame.apply(lambda x: gaussian_filter(x, sigma=2), axis=0)

        # Normalize every avg_period day period to avg_period day average
        print(f'Normalizing every {avg_period} day period to {avg_period} day average')
        for i in range(0, len(self.cont_data_frame), avg_period):
            if i + avg_period > len(self.cont_data_frame):
                # Remove last period if it is not complete
                print(f'Removing last period {i} to {len(self.cont_data_frame)}')
                self.cont_data_frame = self.cont_data_frame.iloc[:i]
                break
            self.cont_data_frame.iloc[i:i+avg_period, :] = self.cont_data_frame.iloc[i:i+avg_period, :] - self.cont_data_frame.iloc[i:i+avg_period, :].mean()
            self.cont_data_frame.iloc[i:i+avg_period, :] = self.cont_data_frame.iloc[i:i+avg_period, :] / self.cont_data_frame.iloc[i:i+avg_period, :].std()

        print(f'Normalizing {self.stock}_{self.period}_data_frame')
        for i in range(self.features):
            # Normalize data to be between 0 and 1
            print(f'Normalizing {self.cont_data_frame.columns[i]} -- Min: {self.cont_data_frame.iloc[:, i].min()}, Max: {self.cont_data_frame.iloc[:, i].max()}')
            # if data_frame.iloc[:, i].max() == data_frame.iloc[:, i].min():
            #     data_frame.iloc[:, i] = 0.0
            #     continue
            self.cont_data_frame.iloc[:, i] = (self.cont_data_frame.iloc[:, i] - self.cont_data_frame.iloc[:, i].min()) / (self.cont_data_frame.iloc[:, i].max() - self.cont_data_frame.iloc[:, i].min())
    
        self.cont_data_frame.to_csv(f'../data/{self.stock}_{self.period}_data_frame_normalized.csv', index=False)
        print(f'Saved Normalized {self.stock}_{self.period}_ data to file')
        