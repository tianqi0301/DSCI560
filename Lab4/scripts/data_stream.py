import yfinance as yf
from datetime import datetime

def stream_price(symbol="AAPL"):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")
    if data.empty:
        return None
    return {
        "time": datetime.now(),
        "symbol": symbol,
        "price": float(data["Close"].iloc[-1]),
        "volume": int(data["Volume"].iloc[-1])
    }
