import os
import requests
from datetime import datetime, timedelta


def fetch_newsapi_headlines(symbol: str, days: int = 7, api_key: str = None):
    api_key = api_key or os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Missing NEWSAPI_KEY")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": symbol,  # you can improve later with company name
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "apiKey": api_key,
    }

    r = requests.get(url, params=params)
    r.raise_for_status()

    data = r.json()

    articles = data.get("articles", [])

    headlines = []
    for a in articles:
        headlines.append({
            "headline": a["title"],
            "source": a["source"]["name"],
            "url": a["url"],
            "datetime": a["publishedAt"],
        })

    return headlines