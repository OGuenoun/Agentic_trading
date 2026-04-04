"""
Trading tools module - provides market data, sentiment analysis, quantitative analysis, 
technical analysis, and portfolio management functions.
"""

from tools.market import generate_market_snapshot
from tools.sentiment import analyze_headline_sentiment, aggregate_sentiments, headline_sentiment_proxy
from tools.news import fetch_newsapi_headlines
from tools.quant import build_quant_context
from tools.technical import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_support_resistance,
    generate_price_target,
    technical_analysis_summary,
)
from tools.portfolio import (
    calculate_position_size,
    calculate_kelly_fraction,
    calculate_risk_reward_ratio,
    optimize_portfolio_weights,
    calculate_portfolio_metrics,
)

__all__ = [
    # Market
    "generate_market_snapshot",
    # Sentiment
    "analyze_headline_sentiment",
    "aggregate_sentiments",
    "headline_sentiment_proxy",
    # News
    "fetch_newsapi_headlines",
    # Quant
    "build_quant_context",
    # Technical
    "calculate_moving_averages",
    "calculate_rsi",
    "calculate_bollinger_bands",
    "calculate_support_resistance",
    "generate_price_target",
    "technical_analysis_summary",
    # Portfolio
    "calculate_position_size",
    "calculate_kelly_fraction",
    "calculate_risk_reward_ratio",
    "optimize_portfolio_weights",
    "calculate_portfolio_metrics",
]
