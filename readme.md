Training and testing data can be obtained from:

1. yfinance
2. https://www.marketwatch.com/
3. https://www.dukascopy.com/swiss/english/marketwatch/historical/

For crypto, training and testing data can be obtained from:

1. https://www.binance.com/my-MM/support/faq/how-to-download-historical-market-data-on-binance-5810ae42176b4770b880ce1f14932262

Things to consider when making an Depp Learning Trading Algo:

1. Slippage
2. Trading Fees
3. Leverage
4. etc.

Generally, size of trade should be 5%-10% if the total account.

Test first without commision fees. It it works consistenly, then signal is good. Then need to integrate fees into the strategy.

Good way to make the market is to first identify the market:

1. Trending Market (Uptrend or Downtrend)
2. Consolidating Market

For Consolidating Market, bet on reversals. For Trending Market, bet on trend continuation.

Approach:

Start with one stock (e.g., AAPL) to fine-tune your RL agent.
Once it’s stable, train on multiple similar stocks (e.g., tech stocks like AAPL, MSFT, GOOGL).
Later, train on a mixed portfolio to generalize further.

When trading Apple (AAPL) stock on Futubull, the platform charges the following fees:
Commission Fee: $0.0049 per share, with a minimum of $0.99 per order and a maximum of 0.5% of the transaction amount.
Platform Fee: $0.005 per share, with a minimum of $1 per order and a maximum of 0.5% of the transaction amount.
Settlement Fee: $0.003 per share.

1. Reward Based on Log Returns
   Instead of price difference. Logarithmic returns scale better over different price ranges.
   More stable for learning.

“rt​\=log(Pt−1​Pt​​)”

###Alpha Env:

1. Actions
   The agent can take two discrete actions, represented by the Actions enum:
   Sell (0): Short the stock (betting that the price will decrease).
   Buy (1): Go long on the stock (betting that the price will increase).
2. Observations
   The observation space consists of:
   A rolling window of historical price data.
   The price change (difference between consecutive prices).
   The observation is a NumPy array of shape (window_size, 2).
3. Rewards
   If the agent executes a trade, the reward is based on the price difference from the last trade:
   Long position: Reward is the price increase.
   Short position: Reward is the price decrease.
   If no trade is made, the reward is 0.
4. Positions
   The agent can hold one of two positions:
   Short (0): Betting on price decline.
   Long (1): Betting on price increase.
   The opposite() function allows easy switching between positions.
5. Termination Conditions
   The episode ends when the last available price data point is reached.
   The \_truncated flag is set to True when the episode ends.
6. Trading Strategy
   The environment allows the agent to switch between Long and Short positions.
   If the agent buys while in a short position, it switches to long.
   If the agent sells while in a long position, it switches to short.
   Every trade has an associated fee (trade_fee_percent).
7. Profit Calculation
   \_update_profit(action): Updates total profit after a trade.
   max_possible_profit(): Computes the best-case profit assuming perfect trades.
8. Rendering
   Uses matplotlib to visualize price movements and trades (green dots for long, red dots for short).
   Allows saving plots and pausing rendering.
9. Data Processing
   \_process_data(): Extracts closing prices and computes price differences.
   Uses self.df, which is assumed to be a Pandas DataFrame containing stock price data.
