"""
news_sentiment.py
=================
Fetches news for a stock symbol and scores sentiment.

What it does:
  1. Resolves ticker → company name  (yfinance, free)
  2. Fetches headlines               (NewsAPI free tier)
  3. Scores each headline            (VADER + finance lexicon)
  4. Detects macro crisis keywords   (WAR, INFLATION etc.)
  5. Returns a clean result dict

Install:
  pip install newsapi-python yfinance vaderSentiment

Get a free NewsAPI key at: https://newsapi.org/register

Run:
  python3 news_sentiment.py
"""

import os
from datetime import datetime, timezone
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf


# ==============================
# Config — set your key here or
# use:  export NEWSAPI_KEY=your_key
# ==============================

NEWS_API_KEY = os.getenv("NEWSAPI_KEY", "")


# ==============================
# Finance-specific word boosts
# VADER treats these as neutral — we correct that
# ==============================

FINANCE_LEXICON = {
    # Bullish
    "beat":        2.0,  "beats":       2.0,
    "outperform":  2.0,  "upgrade":     2.0,
    "upgraded":    2.0,  "bullish":     2.5,
    "breakout":    1.5,  "surge":       1.5,
    "soar":        1.5,  "rally":       1.5,
    "record":      1.0,  "profit":      1.0,
    "buyback":     1.0,  "dividend":    0.8,
    # Bearish
    "miss":       -2.0,  "misses":     -2.0,
    "downgrade":  -2.0,  "downgraded": -2.0,
    "bearish":    -2.5,  "plunge":     -2.0,
    "crash":      -2.5,  "slump":      -1.5,
    "loss":       -1.0,  "lawsuit":    -1.5,
    "probe":      -1.5,  "fraud":      -3.0,
    "bankruptcy": -3.0,  "recall":     -1.5,
    "layoffs":    -1.0,  "warning":    -1.5,
}

# Crisis keywords — if found in headlines, we flag the macro environment
CRISIS_KEYWORDS = {
    "WAR":           ["war", "invasion", "airstrike", "missile", "sanctions", "nato"],
    "TRADE_WAR":     ["tariff", "trade war", "import duty", "export ban"],
    "INFLATION":     ["inflation", "cpi", "rate hike", "stagflation"],
    "RECESSION":     ["recession", "gdp contraction", "market crash", "bank collapse"],
    "PANDEMIC":      ["pandemic", "outbreak", "lockdown", "epidemic"],
    "ENERGY_CRISIS": ["oil shortage", "energy crisis", "opec cut", "fuel shortage"],
}


# ==============================
# Setup (done once at module load)
# ==============================

newsapi   = NewsApiClient(api_key=NEWS_API_KEY)
analyzer  = SentimentIntensityAnalyzer()
analyzer.lexicon.update(FINANCE_LEXICON)   # Apply finance boosts


# ==============================
# Step 1: Resolve company name
# ==============================

def resolve_company_name(symbol: str) -> str:
    """
    Gets the full company name for a ticker using yfinance.
    Falls back to the raw symbol if lookup fails.

    Why: "Apple Inc" gets better NewsAPI results than "AAPL"
    """
    try:
        info = yf.Ticker(symbol).info
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol


# ==============================
# Step 2: Fetch headlines
# ==============================

def fetch_headlines(symbol: str, company_name: str, limit: int = 20) -> list[dict]:
    """
    Fetches recent news headlines for a stock.

    Searches for both company name AND ticker symbol to maximise
    coverage. e.g. query = '"Apple Inc" OR "AAPL"'

    Returns list of dicts with title, source, published_at, age_hours.
    Returns [] on any error — never crashes.
    """
    query = f'"{company_name}" OR "{symbol}"'

    try:
        response = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=limit
        )
    except Exception as e:
        print(f"[NewsAPI] Fetch error: {e}")
        return []

    headlines = []
    now = datetime.now(timezone.utc)

    for article in response.get("articles", []):
        title  = (article.get("title")  or "").strip()
        source = article.get("source", {}).get("name", "Unknown")
        pub_at = article.get("publishedAt", "")

        # Skip deleted/removed articles
        if not title or title == "[Removed]":
            continue

        # Parse age
        try:
            published_at = datetime.fromisoformat(pub_at.replace("Z", "+00:00"))
            age_hours    = round((now - published_at).total_seconds() / 3600, 1)
        except Exception:
            published_at = now
            age_hours    = 0.0

        headlines.append({
            "title":        title,
            "source":       source,
            "published_at": pub_at,
            "age_hours":    age_hours,
        })

    return headlines


# ==============================
# Step 3: Score sentiment
# ==============================

def score_sentiment(headlines: list[dict]) -> dict:
    """
    Scores each headline with VADER (finance-tuned).

    Thresholds used (tighter than VADER defaults):
      compound >= +0.15  → POSITIVE
      compound <= -0.15  → NEGATIVE
      else               → NEUTRAL

    Why tighter? Financial language is often understated.
    "Miss" alone is not a strong word in general English,
    but our finance lexicon boosts it to -2.0.

    Returns aggregated counts and a weighted average score.
    Time decay is applied: older articles count less.
      weight = 0.5 ^ (age_hours / 24)
    """
    if not headlines:
        return {
            "positive": 0, "negative": 0, "neutral": 0,
            "sentiment_score": 0.0, "label": "NEUTRAL",
            "scored_headlines": []
        }

    positive = negative = neutral = 0
    weighted_sum   = 0.0
    total_weight   = 0.0
    scored_headlines = []

    for news in headlines:
        title     = news["title"]
        age_hours = news.get("age_hours", 0.0)

        scores   = analyzer.polarity_scores(title)
        compound = scores["compound"]

        # Time decay: 24h old = 0.5x weight, 48h = 0.25x
        weight        = 0.5 ** (age_hours / 24.0)
        weighted_sum += compound * weight
        total_weight += weight

        if compound >= 0.15:
            label = "POSITIVE"
            positive += 1
        elif compound <= -0.15:
            label = "NEGATIVE"
            negative += 1
        else:
            label = "NEUTRAL"
            neutral += 1

        scored_headlines.append({
            **news,
            "compound": round(compound, 3),
            "label":    label,
        })

    sentiment_score = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

    # Overall label
    if sentiment_score >= 0.35:
        overall_label = "STRONGLY_POSITIVE"
    elif sentiment_score >= 0.15:
        overall_label = "POSITIVE"
    elif sentiment_score <= -0.35:
        overall_label = "STRONGLY_NEGATIVE"
    elif sentiment_score <= -0.15:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    return {
        "positive":        positive,
        "negative":        negative,
        "neutral":         neutral,
        "sentiment_score": sentiment_score,
        "label":           overall_label,
        "scored_headlines": scored_headlines,
    }


# ==============================
# Step 4: Detect crisis keywords
# ==============================

def detect_crisis(headlines: list[dict]) -> dict:
    """
    Scans all headlines for macro crisis keywords.

    Returns the dominant crisis type if found, else None.
    Also returns which headlines triggered it.

    This is a lightweight keyword scan — not ML-based.
    Good enough to catch major events (war, rate shocks etc.)
    """
    crisis_hits = {ct: [] for ct in CRISIS_KEYWORDS}

    for news in headlines:
        text = news["title"].lower()
        for crisis_type, keywords in CRISIS_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    crisis_hits[crisis_type].append(news["title"])
                    break  # one hit per article per crisis type

    # Find the most-hit crisis type
    dominant = max(crisis_hits, key=lambda ct: len(crisis_hits[ct]))

    if len(crisis_hits[dominant]) >= 2:
        return {
            "crisis_detected": True,
            "crisis_type":     dominant,
            "hit_count":       len(crisis_hits[dominant]),
            "examples":        crisis_hits[dominant][:2],   # show 2 examples
        }

    return {
        "crisis_detected": False,
        "crisis_type":     None,
        "hit_count":       0,
        "examples":        [],
    }


# ==============================
# Main function
# ==============================

def get_news_sentiment(symbol: str, timeframe: str = "SHORT_TERM") -> dict:
    """
    Full news sentiment analysis for a stock symbol.

    Args:
      symbol : Ticker e.g. "AAPL"
      limit  : Max headlines to fetch (NewsAPI free: max 100/day total)

    Returns dict with:
      symbol, company_name, headline_count,
      sentiment (scores + label + breakdown),
      crisis (type if detected),
      top_headlines (top 5 with scores)
    """
    # Step 1
    company_name = resolve_company_name(symbol)
    limit = 20  if timeframe == "SHORT_TERM"  else 50

    # Step 2
    headlines = fetch_headlines(symbol, company_name, limit=limit)

    if not headlines:
        return {
            "symbol":         symbol,
            "company_name":   company_name,
            "headline_count": 0,
            "sentiment":      {"label": "NEUTRAL", "sentiment_score": 0.0,
                               "positive": 0, "negative": 0, "neutral": 0},
            "crisis":         {"crisis_detected": False, "crisis_type": None},
            "top_headlines":  [],
            "note":           "No headlines found — check API key or try again later",
        }

    # Step 3
    sentiment = score_sentiment(headlines)

    # Step 4
    crisis = detect_crisis(headlines)

    # Top 5 most impactful headlines
    # Sort by absolute compound score — strongest signal first
    top_headlines = sorted(
        sentiment["scored_headlines"],
        key=lambda h: abs(h["compound"]),
        reverse=True
    )[:5]

    return {
        "symbol":         symbol,
        "company_name":   company_name,
        "headline_count": len(headlines),
        "sentiment": {
            "label":           sentiment["label"],
            "sentiment_score": sentiment["sentiment_score"],
            "positive":        sentiment["positive"],
            "negative":        sentiment["negative"],
            "neutral":         sentiment["neutral"],
        },
        "crisis":        crisis,
        "top_headlines": top_headlines,
    }


# ==============================
# Run directly to test
# ==============================

if __name__ == "__main__":

    test_symbols = ["INFY.NS"]

    for symbol in test_symbols:
        result = get_news_sentiment(symbol)
        print("HELLO")
        print(result)
        print("HI")

        print(f"\n{'='*55}")
        print(f"  {result['symbol']}  —  {result['company_name']}")
        print(f"{'='*55}")
        print(f"  Headlines fetched  : {result['headline_count']}")
        print(f"  Sentiment label    : {result['sentiment']['label']}")
        print(f"  Sentiment score    : {result['sentiment']['sentiment_score']:+.3f}")
        print(f"  Breakdown          : "
              f"{result['sentiment']['positive']} pos / "
              f"{result['sentiment']['neutral']} neu / "
              f"{result['sentiment']['negative']} neg")

        crisis = result["crisis"]
        if crisis["crisis_detected"]:
            print(f"\n  ⚠  CRISIS DETECTED: {crisis['crisis_type']}")
            print(f"     Triggered by {crisis['hit_count']} headlines")
            for ex in crisis["examples"]:
                print(f"     → {ex}")
        else:
            print(f"\n  Crisis             : None detected")

        print(f"\n  Top Headlines (by signal strength):")
        for h in result["top_headlines"]:
            icon = "↑" if h["label"] == "POSITIVE" else ("↓" if h["label"] == "NEGATIVE" else "→")
            print(f"    {icon} [{h['compound']:+.3f}] {h['title'][:65]}")
            print(f"         {h['source']}  •  {h['age_hours']}h ago")