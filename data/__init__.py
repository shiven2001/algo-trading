import pandas as pd
import os

# Load FOREX datasets
FOREX_USDEUR = pd.read_csv('data/usdeurdata.csv', parse_dates=["Date"], index_col="Date")

# Load Stocks datasets
STOCKS_APPL = pd.read_csv('data/appldata.csv', parse_dates=["Date"], index_col="Date")