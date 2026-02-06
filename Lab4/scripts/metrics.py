import numpy as np
import pandas as pd

def calculate_metrics(values):
    returns = pd.Series(values).pct_change().dropna()
    total_return = (values[-1] / values[0]) - 1
    sharpe = 0
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    return total_return, sharpe
