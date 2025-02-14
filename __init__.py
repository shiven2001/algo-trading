from gymnasium.envs.registration import register
from copy import deepcopy
import data

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
    id='multi-stocks-v0',
    entry_point='gym_envs.multi_stocks_env:MultiStocksEnv',
    kwargs={
        'df': deepcopy(data.STOCKS_APPL_GME),
        'window_size': 30,
        'frame_bound': (30, len(data.STOCKS_APPL_GME))
    }
)