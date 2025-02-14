import pandas as pd
import os

# Load FOREX datasets
FOREX_USDEUR = pd.read_csv('data/usdeurdata.csv', parse_dates=["Date"], index_col="Date")

# Load Stocks datasets
STOCKS_APPL = pd.read_csv('data/appldata.csv', parse_dates=["Date"], index_col="Date")

# Multi Stocks
STOCKS_APPL_GME = {
    "AAPL": pd.read_csv('data/appldata.csv', parse_dates=["Date"], index_col="Date"),  # Apple stock data
    "GME": pd.read_csv('data/gmedata.csv', parse_dates=["Date"], index_col="Date"),  # Gamestop stock data
}