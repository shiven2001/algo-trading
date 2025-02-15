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
