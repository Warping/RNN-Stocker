import numpy as np
import pandas as pd

class StockTechnicalIndicators:
    
    def __init__(self, stock_data):
        self.stock_data = stock_data
        # self.data_frame, self.binary_data_frame = self.technical_analysis_dataframe()
        # print(self.data_frame)
        # print(self.binary_data_frame)
        
    def get_dataframes(self, days=30, interval=1):
        # Constrct a dataframe with all the technical indicators
        # The dataframe should have the following columns:
        # - Simple Moving Average (SMA)
        # - Weighted Moving Average (WMA)
        # - Momentum (MOM)
        # - Stochastic K% (STCK)
        # - Stochastic D% (STCD)
        # - Relative Strength Index (RSI)
        # - Signal Line (SIG)
        # - Larry Williams R% (LWR)
        # - Accumulation Distribution Oscillator (ADO)
        # - Commodity Channel Index (CCI)
        
        # The dataframe should have the following index:
        # - The date of the stock data
        
        # The dataframe should have the following rows:
        # - The technical indicator values for each date in the stock data
        
        # The dataframe should be returned
        data = {
            # 'VAL': [],
            'SMA': [],
            'WMA': [],
            'MOM': [],
            'STCK': [],
            'STCD': [],
            'RSI': [],
            'MACD': [],
            'LWR': [],
            # 'ADO': [],
            'CCI': []
        }
        data_val = [self.current_price(i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['SMA'] = [self.simple_moving_average(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['WMA'] = [self.weighted_moving_average(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['MOM'] = [self.momentum(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['STCK'] = [self.stochastic_kpercent(1, days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['STCD'] = [self.stochastic_dpercent(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['RSI'] = [self.relative_strength_index(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['MACD'] = [self.moving_average_convergence_divergence(12, 26, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['LWR'] = [self.larry_williams_r(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        # data['ADO'] = [self.accumulation_distribution_osciallator(i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data['CCI'] = [self.commodity_channel_index(days, i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        indices = [self.current_date(i) for i in range(0, len(self.stock_data) - days, interval)][::-1]
        data_frame = pd.DataFrame(data, index=indices)
        
        # Construct a binary dataframe with the following columns:
        
        binary_data = {
            'SMA': [],
            'WMA': [],
            'MOM': [],
            'STCK': [],
            'STCD': [],
            'RSI': [],
            'MACD': [],
            'LWR': [],
            # 'ADO': [],
            'CCI': []
        }
        
        binary_data['SMA'] = [1 if data_val[i] > data_frame['SMA'].iloc[i] else -1 for i in range(1, len(data_frame))]
        binary_data['WMA'] = [1 if data_val[i] > data_frame['WMA'].iloc[i] else -1 for i in range(1, len(data_frame))]
        binary_data['MOM'] = [1 if data_frame['MOM'].iloc[i] > 0 else -1 for i in range(1, len(data_frame))]
        binary_data['STCK'] = [1 if data_frame['STCK'].iloc[i] - data_frame['STCK'].iloc[i-1] > 0 else -1 for i in range(1, len(data_frame))]
        binary_data['STCD'] = [1 if data_frame['STCD'].iloc[i] - data_frame['STCD'].iloc[i-1] > 0 else -1 for i in range(1, len(data_frame))]
        binary_data['LWR'] = [1 if data_frame['LWR'].iloc[i] - data_frame['LWR'].iloc[i-1] > 0 else -1 for i in range(1, len(data_frame))]
        binary_data['MACD'] = [1 if data_frame['MACD'].iloc[i] - data_frame['MACD'].iloc[i-1] > 0 else -1 for i in range(1, len(data_frame))]
        # binary_data['ADO'] = [1 if data_frame['ADO'].iloc[i] - data_frame['ADO'].iloc[i-1] > 0 else -1 for i in range(1, len(data_frame))]
        binary_data['RSI'] = [1 if data_frame['RSI'].iloc[i] < 30 or (data_frame['RSI'].iloc[i] - data_frame['RSI'].iloc[i-1] > 0 and data_frame['RSI'].iloc[i] < 70) else -1 for i in range(1, len(data_frame))]
        binary_data['CCI'] = [1 if data_frame['CCI'].iloc[i] < -200 or (data_frame['CCI'].iloc[i] - data_frame['CCI'].iloc[i-1] > 0 and data_frame['CCI'].iloc[i] < 200) else -1 for i in range(1, len(data_frame))]
        binary_indices = [self.current_date(i) for i in range(0, len(self.stock_data) - days, interval)][::-1][1:]
        binary_data_frame = pd.DataFrame(binary_data, index=binary_indices)
        return data_frame, binary_data_frame
        
    
    def current_date(self, days_back=0):
        return self.stock_data.index[-(1+days_back)]
          
    def current_price(self, days_back=0):
        return self.stock_data['Close'].iloc[-(1+days_back)]
        
    def simple_moving_average(self, days, days_back=0):
        return self.stock_data['Close'].rolling(window=days).mean().iloc[-(1+days_back)]
    
    def weighted_moving_average(self, days, days_back=0):
        weights = np.arange(1, days+1)
        if days_back == 0:
            return np.dot(self.stock_data['Close'].iloc[-days:], weights) / weights.sum()
        else:
            return np.dot(self.stock_data['Close'].iloc[-(days+days_back):-days_back], weights) / weights.sum()
    
    def momentum(self, days, days_back=0):
        return self.stock_data['Close'].iloc[-(1+days_back)] - self.stock_data['Close'].iloc[-(days+days_back)]
    
    def stochastic_kpercent(self, last_day, days, days_back=0):
        if days_back == 0:
            lowest_low = self.stock_data['Low'].iloc[-days:].min()
            highest_high = self.stock_data['High'].iloc[-days:].max()
        else:
            lowest_low = self.stock_data['Low'].iloc[-(days+days_back):-days_back].min()
            highest_high = self.stock_data['High'].iloc[-(days+days_back):-days_back].max()
        return 100 * (self.stock_data['Close'].iloc[-(last_day+days_back)] - lowest_low) / (highest_high - lowest_low)
    
    def stochastic_dpercent(self, days, days_back=0):
        # return average of last n stochastic k percent values
        last_n_kpercent = [self.stochastic_kpercent(i, days, days_back=days_back) for i in range(1, days+1)]
        return np.mean(last_n_kpercent)
    
    def relative_strength_index(self, days, days_back=0):
        sum_upward = 0
        sum_downward = 0
        for i in range(1, days):
            change = self.stock_data['Close'].iloc[-(i+days_back)] - self.stock_data['Close'].iloc[-(i+days_back)-1]
            if change > 0:
                sum_upward += change
            else:
                sum_downward -= change

        if sum_downward == 0:
            return 100
        return 100 - (100 / (1 + sum_upward / sum_downward))
    
    def moving_average_convergence_divergence(self, short_days, long_days, days_back=0):
        short_ema = self.stock_data['Close'].ewm(span=short_days, adjust=False).mean()
        long_ema = self.stock_data['Close'].ewm(span=long_days, adjust=False).mean()
        macd = short_ema - long_ema
        return macd.iloc[-(1+days_back)]
    
    def signal_line(self, short_days, long_days, signal_days, days_back=0):
        short_ema = self.stock_data['Close'].ewm(span=short_days, adjust=False).mean()
        long_ema = self.stock_data['Close'].ewm(span=long_days, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_days, adjust=False).mean()
        return signal_line.iloc[-(1+days_back)]
    
    def commodity_channel_index(self, days, days_back=0):
        typical_price = (self.stock_data['High'] + self.stock_data['Low'] + self.stock_data['Close']) / 3
        current_price = typical_price.iloc[-(1+days_back)]
        moving_average = self.simple_moving_average(days, days_back=days_back)
        mean_deviation = typical_price.rolling(window=days).std().iloc[-(1+days_back)]
        return (current_price - moving_average) / (0.015 * mean_deviation)
    
    def larry_williams_r(self, days, days_back=0):
        if days_back == 0:
            highest_high = self.stock_data['High'].iloc[-days:].max()
            lowest_low = self.stock_data['Low'].iloc[-days:].min()
        else:
            highest_high = self.stock_data['High'].iloc[-(days+days_back):-days_back].max()
            lowest_low = self.stock_data['Low'].iloc[-(days+days_back):-days_back].min()
        return 100 * (highest_high - self.stock_data['Close'].iloc[-(1+days_back)]) / (highest_high - lowest_low)
    
    # def accumulation_distribution_osciallator(self, days_back=0):
    #     high = self.stock_data['High'].iloc[-(1+days_back)]
    #     low = self.stock_data['Low'].iloc[-(1+days_back)]
    #     close = self.stock_data['Close'].iloc[-(1+days_back)]
    #     if high == low:
    #         return 0
    #     return (high - close) / (high - low)