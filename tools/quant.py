from __future__ import annotations
from statistics import mean, pstdev


def build_quant_context(returns, sentiment_score: float):
    base_mean = mean(returns)
    base_vol = pstdev(returns) if len(returns) > 1 else 0.0
    recent = returns[-5:] if len(returns) >= 5 else returns
    short_mean = mean(recent)

    expected_return = 0.45 * base_mean + 0.35 * short_mean + 0.20 * sentiment_score * 0.02
    expected_volatility = max(0.0001, base_vol * (1 + max(0, -sentiment_score) * 0.25))

    if expected_return > 0.001:
        direction = "bullish"
    elif expected_return < -0.001:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "expected_return": float(expected_return),
        "expected_volatility": float(expected_volatility),
        "direction": direction,
        "base_mean": float(base_mean),
        "short_mean": float(short_mean),
        "sentiment_score": float(sentiment_score),
        "notes": "Replace this with ARIMAX/GARCH when you connect real quant models.",
    }
