import yfinance as yf
import pandas as pd
from statistics import mean, pstdev


def generate_market_snapshot(symbol: str, days: int):
    # Download data
    df = yf.download(symbol, period=f"{days+30}d", interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    else:
        prices = df["Close"]
    prices = prices.dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    # Compute returns
    returns = prices.pct_change().dropna()

    # Keep last N days
    prices = prices[-days:]
    returns = returns[-days:]
    
    # Metrics
    momentum_20 = returns[-20:].sum() if len(returns) >= 20 else returns.sum()
    realized_vol = returns.std()

    return {
        "symbol": symbol,
        "last_price": float(prices.iloc[-1]),
        "mean_return": float(returns.mean()),
        "realized_volatility": float(realized_vol),
        "momentum_20": float(momentum_20),
        "returns": returns.tolist(),
        "prices": prices.tolist(),
    }