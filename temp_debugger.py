from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import pandas as pd

''' TODO: ADD MORE ACTIONS IN FUTURE '''
class Actions(Enum):
    Sell = 0
    Buy = 1

''' TODO: ADD NO TRADE POSITION IN FUTURE '''
class Positions(Enum):
    Short = 0
    Long = 1
    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long
    
class BetaStocksEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 3}
    def __init__(self, df, trade_fee_percent, capital, window_size, frame_bound, render_mode=None):
        super().__init__()  # No arguments required

        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata['render_modes']
        assert len(frame_bound) == 2
        self.capital = capital
        self.frame_bound = frame_bound  # Ensure this is set before accessing it
        self.render_mode = render_mode
        self.trade_fee_percent = trade_fee_percent
        self.df = df
        self.window_size = window_size

        print(f"capital: {self.capital}")
        print(f"trade_fee_percent: {self.trade_fee_percent}")

        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1])

        # action_space and observation_space
        self.action_space = gym.spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        )

        # episode index within the frame bound of prices array
        self._start_tick = self.window_size # start tick after window_size
        self._end_tick = len(self.prices) - 1 # end tick after prices array is over -1

        # trading index tick
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None

        # others
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None
        
        # buy and hold
        self.buy_hold_profit = self._calculate_buy_hold_profit()
        print(f"buy_hold_profit: {self.buy_hold_profit}")

    # observation processing (ALSO IMPORTANT)
    # The price index is two ahead of csv file. e.g. 10 price index is 12 row index in csv file
    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        # Compute additional features
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices.astype(np.float32), signal_features.astype(np.float32)
    
    def _get_trade_price(self, price, slippage_percent=0.0005):
        return price * (1 + random.uniform(-slippage_percent, slippage_percent))  

    def _calculate_buy_hold_profit(self):
        """Calculates profit if we simply bought at the start and sold at the end."""
        #entry_price = self._get_trade_price(self.prices[0 + self.window_size])  # First price
        #exit_price = self._get_trade_price(self.prices[len(self.prices) - 1])  # Last price
        entry_price = self.prices[0 + self.window_size]  # First price
        exit_price = self.prices[len(self.prices) - 1]  # Last price
        shares = (self.capital * (1 - self.trade_fee_percent)) / entry_price  # Fee on entry
        buy_hold_profit = (shares * (1 - self.trade_fee_percent)) * exit_price  # Fee on exit
        return buy_hold_profit
    
df = pd.read_csv('./data/googldata.csv', parse_dates=["Date"], index_col="Date")
BetaStocksEnv(df=df, trade_fee_percent=0, capital=10000, window_size=5, frame_bound=(10,100), render_mode=None)
