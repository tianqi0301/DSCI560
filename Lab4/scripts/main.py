import pandas as pd
import yfinance as yf
from portfolio import Portfolio
from indicators import sma, rsi
from strategies import hybrid_strategy
from metrics import calculate_metrics

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "SPY", "QQQ"
]

ALLOCATION = 0.30

portfolio = Portfolio(1000000)
dfs = {}

for symbol in SYMBOLS:
    data = yf.download(symbol, period="1d", interval="1m", auto_adjust=False, progress=False)
    if data.empty:
        continue

    df = pd.DataFrame()
    df["time"] = data.index
    df["price"] = data["Close"].values
    df["volume"] = data["Volume"].values
    df.reset_index(drop=True, inplace=True)

    df["sma"] = sma(df["price"], 5)
    df["rsi"] = rsi(df["price"], 7)

    dfs[symbol] = df

min_len = min(len(df) for df in dfs.values())

for i in range(15, min_len):
    prices = {}

    for symbol, df in dfs.items():
        current_price = df.loc[i, "price"]
        prices[symbol] = current_price

        signal = hybrid_strategy(df.iloc[: i + 1])

        if signal == 1:
            shares = int((portfolio.cash * ALLOCATION) // current_price)
            if shares > 0:
                portfolio.buy(symbol, current_price, shares)

        elif signal == -1:
            portfolio.sell(symbol, current_price, portfolio.positions.get(symbol, 0))

        print(
            f"{df.loc[i,'time']} | {symbol} | "
            f"Signal: {signal} | "
            f"Cash: ${portfolio.cash:,.2f} | "
            f"Positions: {portfolio.positions} | "
            f"Total Value: ${portfolio.value(prices):,.2f}"
        )

    portfolio.history.append(portfolio.value(prices))

ret, sharpe = calculate_metrics(portfolio.history)

print("\nFinal Results")
print("Total Return:", round(ret * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe, 2))
