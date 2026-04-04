"""
Technical analysis tools for price prediction and support/resistance levels.
"""
from __future__ import annotations
import yfinance as yf
import pandas as pd
from statistics import mean, stdev


def calculate_moving_averages(prices: list, windows: list = None) -> dict:
    """
    Calculate moving averages for given windows.
    
    Args:
        prices: list of price data
        windows: list of window sizes (default: [20, 50, 200])
        
    Returns:
        dict with MA values and signals
    """
    if windows is None:
        windows = [20, 50, 200]
    
    prices_array = pd.Series(prices)
    mas = {}
    
    for window in windows:
        if len(prices_array) >= window:
            ma = prices_array.rolling(window=window).mean()
            mas[f"ma_{window}"] = float(ma.iloc[-1])
    
    current = prices[-1]
    signals = []
    
    if "ma_20" in mas and "ma_50" in mas:
        if mas["ma_20"] > mas["ma_50"]:
            signals.append("bullish_crossover")
        elif mas["ma_20"] < mas["ma_50"]:
            signals.append("bearish_crossover")
    
    return {
        "moving_averages": mas,
        "current_price": float(current),
        "signals": signals,
    }


def calculate_rsi(prices: list, period: int = 14) -> dict:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: list of price data
        period: RSI period (default: 14)
        
    Returns:
        dict with RSI value and signal
    """
    if len(prices) < period:
        return {"rsi": None, "signal": "insufficient_data"}
    
    prices_array = pd.Series(prices)
    deltas = prices_array.diff()
    
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # RSI signal interpretation
    signal = "neutral"
    if rsi > 70:
        signal = "overbought"
    elif rsi < 30:
        signal = "oversold"
    
    return {
        "rsi": float(rsi),
        "signal": signal,
        "period": period,
    }


def calculate_bollinger_bands(prices: list, period: int = 20, std_dev: int = 2) -> dict:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: list of price data
        period: moving average period
        std_dev: number of standard deviations
        
    Returns:
        dict with bands and current price position
    """
    if len(prices) < period:
        return {"bands": None, "signal": "insufficient_data"}
    
    prices_array = pd.Series(prices)
    sma = prices_array.rolling(window=period).mean()
    std = prices_array.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    current_price = prices[-1]
    middle_band = float(sma.iloc[-1])
    
    # Determine position
    if current_price > float(upper_band.iloc[-1]):
        position = "above_upper"
    elif current_price < float(lower_band.iloc[-1]):
        position = "below_lower"
    else:
        position = "within_bands"
    
    return {
        "upper_band": float(upper_band.iloc[-1]),
        "middle_band": middle_band,
        "lower_band": float(lower_band.iloc[-1]),
        "current_price": current_price,
        "position": position,
        "bandwidth": float((upper_band.iloc[-1] - lower_band.iloc[-1]) / middle_band),
    }


def calculate_support_resistance(prices: list, window: int = 20) -> dict:
    """
    Calculate support and resistance levels using rolling max/min.
    
    Args:
        prices: list of price data
        window: lookback window
        
    Returns:
        dict with support/resistance levels
    """
    if len(prices) < window:
        return {
            "resistance": prices[-1],
            "support": prices[-1],
            "distance": 0,
        }
    
    prices_array = pd.Series(prices)
    
    resistance = float(prices_array[-window:].max())
    support = float(prices_array[-window:].min())
    current = prices[-1]
    
    return {
        "resistance": resistance,
        "support": support,
        "current_price": current,
        "distance_to_resistance": (resistance - current) / current if current > 0 else 0,
        "distance_to_support": (current - support) / current if current > 0 else 0,
    }


def generate_price_target(
    current_price: float,
    returns: list,
    volatility: float,
    direction: str = "neutral",
    horizon_days: int = 20,
) -> dict:
    """
    Generate price targets based on historical volatility and direction.
    
    Args:
        current_price: current stock price
        returns: historical daily returns
        volatility: realized volatility
        direction: 'bullish', 'bearish', or 'neutral'
        horizon_days: days ahead for target
        
    Returns:
        dict with price targets
    """
    # Expected move based on volatility
    expected_move = current_price * volatility * (horizon_days ** 0.5)
    
    base_target = current_price
    if direction == "bullish":
        upside = current_price + expected_move * 1.5
        downside = current_price + expected_move * 0.25
    elif direction == "bearish":
        upside = current_price - expected_move * 0.25
        downside = current_price - expected_move * 1.5
    else:  # neutral
        upside = current_price + expected_move * 0.5
        downside = current_price - expected_move * 0.5
    
    return {
        "current_price": float(current_price),
        "upside_target": float(upside),
        "downside_target": float(downside),
        "expected_move": float(expected_move),
        "direction": direction,
        "horizon_days": horizon_days,
        "upside_percent": float((upside - current_price) / current_price * 100),
        "downside_percent": float((downside - current_price) / current_price * 100),
    }


def technical_analysis_summary(symbol: str, prices: list, returns: list) -> dict:
    """
    Generate a comprehensive technical analysis summary.
    
    Args:
        symbol: stock symbol
        prices: list of prices
        returns: list of returns
        
    Returns:
        dict with comprehensive technical analysis
    """
    volatility = pd.Series(returns).std()
    
    summary = {
        "symbol": symbol,
        "moving_averages": calculate_moving_averages(prices),
        "rsi": calculate_rsi(prices),
        "bollinger_bands": calculate_bollinger_bands(prices),
        "support_resistance": calculate_support_resistance(prices),
    }
    
    return summary
