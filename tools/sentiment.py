from __future__ import annotations
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


# Initialize VADER sentiment analyzer
_sia = SentimentIntensityAnalyzer()


def analyze_headline_sentiment(headline: str) -> dict:
    """
    Analyze sentiment of a single headline using VADER.
    
    Returns:
        dict with compound, pos, neu, neg scores
    """
    scores = _sia.polarity_scores(headline)
    return scores


def aggregate_sentiments(headlines: list) -> dict:
    """
    Aggregate sentiment scores across multiple headlines.
    
    Args:
        headlines: list of headline strings
        
    Returns:
        dict with aggregated scores and sentiment summary
    """
    if not headlines:
        return {
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "sentiment_summary": "neutral",
            "headline_count": 0,
        }
    
    compound_scores = []
    pos_scores = []
    neg_scores = []
    neu_scores = []
    
    for headline in headlines:
        scores = analyze_headline_sentiment(headline)
        compound_scores.append(scores['compound'])
        pos_scores.append(scores['pos'])
        neg_scores.append(scores['neg'])
        neu_scores.append(scores['neu'])
    
    avg_compound = sum(compound_scores) / len(compound_scores)
    avg_positive = sum(pos_scores) / len(pos_scores)
    avg_negative = sum(neg_scores) / len(neg_scores)
    avg_neutral = sum(neu_scores) / len(neu_scores)
    
    # Determine overall sentiment
    if avg_compound >= 0.05:
        sentiment_summary = "bullish"
    elif avg_compound <= -0.05:
        sentiment_summary = "bearish"
    else:
        sentiment_summary = "neutral"
    
    return {
        "compound": float(avg_compound),
        "positive": float(avg_positive),
        "negative": float(avg_negative),
        "neutral": float(avg_neutral),
        "sentiment_summary": sentiment_summary,
        "headline_count": len(headlines),
        "raw_scores": compound_scores,
    }


def headline_sentiment_proxy(headlines):
    """
    Legacy function for backward compatibility.
    Returns normalized sentiment score from -1 to 1.
    """
    if not headlines:
        return 0.0
    
    aggregated = aggregate_sentiments(headlines)
    # compound score is already in -1 to 1 range
    return aggregated["compound"]
