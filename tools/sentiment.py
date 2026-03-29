from __future__ import annotations
import random


POSITIVE_WORDS = {
    "beat", "growth", "upside", "strong", "upgrade", "surge", "profit", "record", "improves"
}
NEGATIVE_WORDS = {
    "miss", "downgrade", "lawsuit", "risk", "weak", "decline", "loss", "pressure", "cuts"
}


def generate_mock_headlines(symbol: str, seed: int):
    rng = random.Random(seed + 17)
    templates = [
        f"{symbol} posts strong quarterly growth and record margins",
        f"Analysts debate {symbol} after mixed guidance and cost pressure",
        f"{symbol} faces lawsuit concerns but product demand remains strong",
        f"{symbol} receives upgrade as revenue outlook improves",
        f"{symbol} shares decline after weak short-term forecast",
    ]
    rng.shuffle(templates)
    return templates[:3]


def headline_sentiment_proxy(headlines):
    score = 0.0
    for h in headlines:
        words = set(h.lower().replace("-", " ").split())
        score += len(words & POSITIVE_WORDS)
        score -= len(words & NEGATIVE_WORDS)
    return max(-1.0, min(1.0, score / 5.0))
