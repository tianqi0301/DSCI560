def hybrid_strategy(df):
    if len(df) < 15:
        return 0
    price = df["price"].iloc[-1]
    sma_val = df["sma"].iloc[-1]
    rsi_val = df["rsi"].iloc[-1]
    if price > sma_val and rsi_val < 70:
        return 1
    if price < sma_val or rsi_val > 70:
        return -1
    return 0
