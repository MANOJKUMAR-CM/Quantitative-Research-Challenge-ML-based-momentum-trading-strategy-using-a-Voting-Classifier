import yfinance as yf
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Stocks
stocks = ['AAPL', 'META', 'TSLA', 'JPM', 'AMZN']

# Training Data
data = {}
for stock in stocks:
    df = yf.download(stock, start='2015-01-01', end='2025-01-01')
    data[stock] = df


def add_features(df):
    df['SMA_50'] = SMAIndicator(df['Close'].squeeze(), window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(df['Close'].squeeze(), window=200).sma_indicator()
    df['EMA_21'] = EMAIndicator(df['Close'].squeeze(), window=21).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'].squeeze(), window=14).rsi()
    df['MACD'] = MACD(df['Close'].squeeze()).macd()
    df['MACD_Signal'] = MACD(df['Close'].squeeze()).macd_signal()
    df['Stoch'] = StochasticOscillator(df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()).stoch()
    df['BBW'] = BollingerBands(df['Close'].squeeze()).bollinger_wband()
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()

    # Return-based features
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    df['Monthly_Return'] = df['Close'].pct_change(21)

    # **Drop NaNs after feature calculation**
    df.dropna(inplace=True)

    return df


for stock, df in data.items():
    data[stock] = add_features(df)
    
    
def define_labels(df):
    df['Signal'] = 0  # Hold by default
    
    # Buy Signals: Buy when RSI < 30 OR MACD crosses above Signal OR OBV rising
    buy_condition = (df['RSI'] < 30) | (df['MACD'] > df['MACD_Signal']) & (df['OBV'].diff() > 0)
    df.loc[buy_condition, 'Signal'] = 1  
    
    # Sell Signals: Sell when RSI > 70 OR MACD crosses below Signal OR OBV dropping
    sell_condition = (df['RSI'] > 70) | (df['MACD'] < df['MACD_Signal']) & (df['OBV'].diff() < 0)
    df.loc[sell_condition, 'Signal'] = -1  
    
    return df

# Apply labels to each stock
data = {stock: define_labels(df) for stock, df in data.items()}