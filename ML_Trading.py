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