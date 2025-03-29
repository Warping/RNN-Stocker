import yfinance as yf
from datetime import timedelta
from datetime import date as dt
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import stock_technical_indicators as sti

class Portfolio:
    def __init__(self, capital=0.0):
        self.stocks = {}
        self.capital = capital

    def purchase_stock(self, stock_name, quantity, date=None):
        price_per_stock = self.get_stock_value(stock_name, date)
        if price_per_stock is not None:
            self.capital -= quantity * price_per_stock
            print(f"Purchased {quantity} shares of {stock_name} at ${price_per_stock:.2f} per share")
            if stock_name in self.stocks:
                self.stocks[stock_name] += quantity
            else:
                self.stocks[stock_name] = quantity
        else:
            print(f"Could not purchase {quantity} shares of {stock_name}")

    def sell_stock(self, stock_name, quantity, date=None):
        if stock_name in self.stocks and self.stocks[stock_name] >= quantity:
            price_per_stock = self.get_stock_value(stock_name, date)
            if price_per_stock is not None:
                self.stocks[stock_name] -= quantity
                self.capital += quantity * price_per_stock
                print(f"Sold {quantity} shares of {stock_name} at ${price_per_stock:.2f} per share")
                if self.stocks[stock_name] == 0:
                    del self.stocks[stock_name]
            else:
                print(f"Could not sell {quantity} shares of {stock_name}")
        else:
            print("Not enough stock to sell")

    def display_portfolio(self, date=None):
        print(f"Portfolio: - {date}")
        total_value = 0.0
        for stock, quantity in self.stocks.items():
            stock_value = self.get_stock_value(stock, date)
            total_stock_value = stock_value * quantity
            print(f"-\tTicker: {stock} - Quantity: {quantity} - Share Value: ${stock_value:.2f} - Total Value: ${total_stock_value:.2f}")
            total_value += total_stock_value
        print("-" * 50)
        print(f"Total Stock Value: ${total_value:.2f}")
        print(f"Total Bankroll: ${self.capital:.2f}")
        print(f"Total Portfolio Value: ${total_value + self.capital:.2f}")

    def get_stock_value(self, stock_name, date=None):
        if date is None or date == dt.today():
            stock_data = yf.download(tickers=stock_name, period='1d', interval='1m', progress=False)
        else:
            # start = date - timedelta(days=1)
            end = date
            stock_data = yf.download(tickers=stock_name, end=end, progress=False)
        if not stock_data.empty:
            close_price = stock_data['Close'].iloc[-1].values[0]
            # print(f"Stock: {stock_name} - Price: {ticker_price}")
            return close_price
        else:
            print(f"Could not retrieve data for {stock_name}")
            return None
        
    def graph_stock(self, stock_name):
        stock_data = yf.Ticker(stock_name).history(period='1mo')
        if stock_data is not None:
            # stock_data['Datetime'] = pd.to_datetime(stock_data['Datetime'])
            # stock_data.set_index('Datetime', inplace=True)
            # stock_data = yf.download(tickers=stock_name, period='1d', interval='1m', progress=False)
            mpf.plot(stock_data, type='candle', volume=True, style='yahoo', title=f'{stock_name} Stock Price', ylabel='Price', ylabel_lower='Volume')
        else:
            print(f"Could not graph {stock_name}")
            
    def tech_analysis(self, stock_name, interval=1):
        stock_data = yf.Ticker(stock_name).history(period='10y', interval='1d')
        # days = 10
        # days_back = 5
        if stock_data is not None:
            tech_indicators = sti.StockTechnicalIndicators(stock_data)
            data_frame, binary_data_frame = tech_indicators.get_dataframes(interval=interval)
            # Graph the data
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))
            for column in data_frame.columns:
                ax1.plot(data_frame[column], label=column)
            for column in binary_data_frame.columns:
                ax2.plot(binary_data_frame[column], label=column)
            ax1.legend()
            ax2.legend()
            plt.title(f'{stock_name} Technical Indicators')
            plt.show()
        else:
            print(f"Could not analyze {stock_name}")

# Example usage
# portfolio = Portfolio()
# portfolio.purchase_stock("AAPL", 10, 150)
# portfolio.purchase_stock("GOOGL", 5, 1000)
# portfolio.sell_stock("AAPL", 5, 160)
# portfolio.display_total_capital()
# portfolio.display_portfolio()
# portfolio.display_total_portfolio_value()