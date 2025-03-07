# Raw Package 
import numpy as np 
import pandas as pd
from datetime import date, timedelta
from matplotlib import pyplot as plt

# Market Data 
import yfinance as yf

# Graphing / Visualization 
import plotly.graph_objs as go 

# Stock Data
stock = 'MSTR'

start = date.today() - timedelta(days=365)
end = date.today() + timedelta(days=2)

start.strftime('%Y-%m-%d')
end.strftime('%Y-%m-%d')

asset = pd.DataFrame(yf.download(tickers=stock, start=start, end=end))

plt.plot(asset['Close'])
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
