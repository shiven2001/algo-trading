from gymnasium.envs.registration import register
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath(".")) 
import data
import gymnasium as gym

register(
    id='forex-v0',
    entry_point='gym_envs.forex_env:ForexEnv',
    kwargs={
        'df': deepcopy(data.FOREX_USDEUR),
        'window_size': 24,
        'frame_bound': (24, len(data.FOREX_USDEUR))
    }
)

register(
    id='stocks-v0',
    entry_point='gym_envs.stocks_env:StocksEnv',
    kwargs={
        'df': deepcopy(data.STOCKS_APPL),
        'window_size': 30,
        'frame_bound': (30, len(data.STOCKS_APPL))
    }
)

register(
    id='zero-stocks-v0',
    entry_point='gym_envs.zero_stocks_env:ZeroStocksEnv',
    kwargs={
        'df': deepcopy(data.STOCKS_APPL),
        'window_size': 30,
        'frame_bound': (30, len(data.STOCKS_APPL))
    }
)

register(
    id='alpha-stocks-v0',
    entry_point='gym_envs.alpha_stocks_env:AlphaStocksEnv',
    kwargs={
        'df': deepcopy(data.STOCKS_APPL),
        'window_size': 30,
        'frame_bound': (30, len(data.STOCKS_APPL))
    }
)


register(
    id='beta-stocks-v0',
    entry_point='gym_envs.beta_stocks_env:BetaStocksEnv',
    kwargs={
        'df': deepcopy(data.STOCKS_APPL),
        'window_size': 30,
        'frame_bound': (30, len(data.STOCKS_APPL))
    }
)

print(gym.registry.keys())  # List of registered environments
