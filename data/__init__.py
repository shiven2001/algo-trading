import pandas as pd
import os

# Get the absolute path of the "data" folder
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load FOREX datasets
FOREX_USDEUR = pd.read_csv(os.path.join(DATA_DIR, "usdeurdata.csv"), parse_dates=["Date"], index_col="Date")

# Load Stocks datasets
STOCKS_APPL = pd.read_csv(os.path.join(DATA_DIR, 'appldata.csv'), parse_dates=["Date"], index_col="Date")

# Multi Stocks
STOCKS_APPL_GME = {
    "AAPL": pd.read_csv(os.path.join(DATA_DIR, 'appldata.csv'), parse_dates=["Date"], index_col="Date"),  # Apple stock data
    "GME": pd.read_csv(os.path.join(DATA_DIR, 'gmedata.csv'), parse_dates=["Date"], index_col="Date"),  # Gamestop stock data
}
