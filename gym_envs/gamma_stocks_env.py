from time import time
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

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
    
class GammaStocksEnv(gym.Env):
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
        print(f"buy_hold_end capital: {self.buy_hold_profit}")
        print(f"buy_hold_profit: {self.buy_hold_profit - self.capital}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.np_random = np.random.RandomState(seed)
        self.action_space.seed(seed)
        self._truncated = False
        self.initial_capital = self.capital
        # within the price array
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        # start position is a short
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.0
        self._total_profit = 0.0
        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        # reward step and total profit at end
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward
        step_profit = self._calculate_profit(action)
        self._total_profit += step_profit
        self.capital += step_profit

        # trade logic (update position AFTER calculating profit and reward)
        trade = False
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        # saving positions in _position_history array
        self._position_history.append(self._position)
        observation = self._get_observation()
        # _get_info is total_reward, total_profit, position
        info = self._get_info()
        # history is total_reward, total_profit, position
        self._update_history(info)

        # Check if the episode has been truncated and reset the environment
        if self._truncated:
            self.reset()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position
        )

    # observation about signal_features from begining of current tick to back window size
    def _get_observation(self):
        return self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        ''' Can use both render or render_all functions depending on requirement '''
        self.render()
        #self.render_all()

    # render shows red means short position and green means long position
    # Visualizes only the most recent trade. It updates the chart incrementally during the simulation.
    def render(self, mode='human'):
        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata['render_fps']) - process_time
        assert pause_time > 0., "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)


    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    # observation processing (ALSO IMPORTANT)
    # The price index is two ahead of csv file. e.g. 10 price index is 12 row index in csv file
    def _process_data(self):
        """Calculates the price array as well as signal_features for obervations"""
        prices = self.df.loc[:, 'Close'].to_numpy()
        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0]-self.window_size:self.frame_bound[1]]
        # Compute additional features
        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))
        return prices.astype(np.float32), signal_features.astype(np.float32)

    # Reward function (MOST IMPORTANT)
    def _calculate_reward(self, action):
        """Calculates reward on each step."""
        step_reward = 0
        trade = False
        # check if trade was executed
        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True
        # if trade was executed, then 
        if trade:
            current_price = self._get_trade_price(price=self.prices[self._current_tick],action=action)
            last_trade_price = self._get_trade_price(price=self.prices[self._last_trade_tick],action=action)
            #current_price = self.prices[self._current_tick]
            #last_trade_price = self.prices[self._last_trade_tick]

            # Reward Function
            price_diff = current_price - last_trade_price
            # Calculate log return
            log_return = np.log(current_price / last_trade_price)

            if self._position == Positions.Long:
                step_reward = log_return
            elif self._position == Positions.Short:
                step_reward = -log_return  # Profit when shorting and price drops

            # Add risk-adjusted reward (Sharpe Ratio component)
            #step_reward /= np.std(self.prices[:self._current_tick] + 1e-8)

        # Penalize excessive trading
        if trade:
            step_reward -= 0.001  # Small penalty for each trade

        return step_reward
    
    def _calculate_profit(self, action):
        """Calculates profit on each step."""
        step_profit = 0
        trade = False

        if (
            (action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)
        ):
            trade = True

        if trade or self._truncated:
            entry_price = self._get_trade_price(price=self.prices[self._last_trade_tick],action=action)
            exit_price = self._get_trade_price(price=self.prices[self._current_tick],action=action)

            # Calculate profit based on the price difference
            if self._position == Positions.Long:
                shares = self.capital / entry_price  # Buy shares at entry price
                step_profit = shares * (exit_price - entry_price)  # Profit is price difference
            elif self._position == Positions.Short: 
                shares = self.capital / entry_price  # Sell shares at entry price
                step_profit = shares * (entry_price - exit_price)  # Profit when price drops
            
            # Apply trade fee
            step_profit *= (1 - self.trade_fee_percent)

        return step_profit
    
    def _calculate_buy_hold_profit(self):
        """Calculates profit if we simply bought at the start and sold at the end."""
        entry_price = self._get_trade_price(price=self.prices[0 + self.window_size],action=1)  # First price
        exit_price = self._get_trade_price(price=self.prices[len(self.prices) - 1], action=0)  # Last price
        #entry_price = self.prices[0 + self.window_size]  # First price
        #exit_price = self.prices[len(self.prices) - 1]  # Last price
        shares = (self.capital * (1 - self.trade_fee_percent)) / entry_price  # Fee on entry
        buy_hold_profit = (shares * (1 - self.trade_fee_percent)) * exit_price  # Fee on exit
        return buy_hold_profit

    def _get_trade_price(self, price, action, slippage_mean=0, slippage_std=0.001):
        volatility = np.std(self.prices[max(0, self._current_tick - 50):self._current_tick])  # Look-back window
        slippage_std = volatility * 0.001  # Adjust slippage with volatility
        """Calculates slippage compensation."""
        # Simulate slippage using a normal distribution
        slippage = self.np_random.normal(slippage_mean, slippage_std)
        if action == Actions.Buy.value:
            # Buy action: slippage tends to increase the price (positive slippage)
            return price * (1 + slippage)
        elif action == Actions.Sell.value:
            # Sell action: slippage tends to decrease the price (negative slippage)
            return price * (1 - slippage)
        else:
            return price
