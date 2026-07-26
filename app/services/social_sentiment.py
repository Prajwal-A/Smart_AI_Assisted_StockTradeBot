"""
social_sentiment.py
===================
Social/retail investor sentiment using RSS feeds only.
Zero API keys. Zero setup. Runs immediately.

Sources:
  1. Reddit subreddit RSS   — r/wallstreetbets, r/investing, r/stocks
                              Reddit blocks API but RSS still works publicly
  2. Seeking Alpha RSS      — Retail investor articles
  3. Yahoo Finance RSS      — Community discussion

What it does:
  1. Fetches posts mentioning the ticker from all sources
  2. Scores each with VADER (finance-tuned lexicon)
  3. Detects general market mood from WSB hot posts
  4. Returns one clean result dict — same structure as get_news_sentiment()

Install:
  pip install feedparser vaderSentiment requests

Run:
  python3 social_sentiment.py
"""

import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ==============================
# RSS feed templates
# {symbol} replaced at runtime
# ==============================

STOCK_RSS_FEEDS = {
    "WSB":             "https://www.reddit.com/r/wallstreetbets/search.rss?q={symbol}&sort=new&restrict_sr=1",
    "r_investing":     "https://www.reddit.com/r/investing/search.rss?q={symbol}&sort=new&restrict_sr=1",
    "r_stocks":        "https://www.reddit.com/r/stocks/search.rss?q={symbol}&sort=new&restrict_sr=1",
    "SeekingAlpha":    "https://seekingalpha.com/api/sa/combined/{symbol}.xml",
    "YahooFinance":    "https://finance.yahoo.com/rss/headline?s={symbol}",
}

# Market mood — not stock specific, reflects overall retail sentiment
MOOD_RSS_FEEDS = {
    "WSB_Hot":         "https://www.reddit.com/r/wallstreetbets/hot.rss",
    "WSB_New":         "https://www.reddit.com/r/wallstreetbets/.rss",
    "r_investing_hot": "https://www.reddit.com/r/investing/hot.rss",
}

FINANCE_LEXICON = {
    "beat":        2.0,  "beats":       2.0,
    "outperform":  2.0,  "upgrade":     2.0,
    "upgraded":    2.0,  "bullish":     2.5,
    "breakout":    1.5,  "surge":       1.5,
    "soar":        1.5,  "rally":       1.5,
    "record":      1.0,  "profit":      1.0,
    "buyback":     1.0,  "dividend":    0.8,
    "miss":       -2.0,  "misses":     -2.0,
    "downgrade":  -2.0,  "downgraded": -2.0,
    "bearish":    -2.5,  "plunge":     -2.0,
    "crash":      -2.5,  "slump":      -1.5,
    "loss":       -1.0,  "lawsuit":    -1.5,
    "probe":      -1.5,  "fraud":      -3.0,
    "bankruptcy": -3.0,  "layoffs":    -1.0,
    "warning":    -1.5,
    # Social/retail specific terms
    "moon":        2.0,  "mooning":     2.0,
    "rocket":      1.5,  "squeeze":     1.5,
    "hodl":        1.5,  "oversold":    1.5,
    "yolo":        1.0,  "calls":       0.8,
    "puts":       -0.8,  "dump":       -2.0,
    "short":      -1.0,  "rug":        -2.5,
    "overbought": -1.5,  "bagholder":  -2.0,
}

# Setup VADER once at module load
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(FINANCE_LEXICON)


# ==============================
# Helper: fetch one RSS feed
# ==============================

def _fetch_rss(url: str, source_name: str, timeout: int = 10) -> list:
    """
    Fetches and parses a single RSS feed.
    Returns raw feedparser entries, or [] on any error.

    Uses a browser User-Agent because Reddit blocks
    Python's default requests agent.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingBot/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 429:
            print(f"[RSS] {source_name}: rate limited")
            return []
        if response.status_code == 404:
            return []
        if response.status_code != 200:
            print(f"[RSS] {source_name}: HTTP {response.status_code}")
            return []
        parsed = feedparser.parse(response.text)
        return parsed.entries or []
    except Exception as e:
        print(f"[RSS] {source_name}: {e}")
        return []


# ==============================
# Helper: parse entry age
# ==============================

def _parse_age(entry) -> float:
    """Returns how many hours old this RSS entry is."""
    try:
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        else:
            return 0.0
        return (datetime.now(timezone.utc) - published).total_seconds() / 3600
    except Exception:
        return 0.0


# ==============================
# Helper: relevance check
# ==============================

def _is_relevant(text: str, symbol: str) -> bool:
    """
    True if text mentions the ticker.
    Matches both AAPL and $AAPL style.
    """
    text_lower   = text.lower()
    symbol_lower = symbol.lower()
    return symbol_lower in text_lower or f"${symbol_lower}" in text_lower


# ==============================
# Step 1: Fetch stock posts
# ==============================

def fetch_social_posts(symbol: str, max_age_hours: int = 48) -> list[dict]:
    """
    Fetches posts mentioning the ticker from social RSS feeds.

    Filters:
      - Must mention ticker symbol or $ticker
      - Must be within max_age_hours (default 48h)
      - Duplicates removed by URL hash

    Returns list of dicts with: text, source, age_hours, url
    """
    posts    = []
    seen_ids = set()

    for source_name, url_template in STOCK_RSS_FEEDS.items():
        url     = url_template.format(symbol=symbol.upper())
        entries = _fetch_rss(url, source_name)

        for entry in entries:
            title   = getattr(entry, "title",   "").strip()
            summary = getattr(entry, "summary", "").strip()
            url_out = getattr(entry, "link",    "").strip()

            if not title:
                continue

            full_text = f"{title} {summary[:200]}"

            if not _is_relevant(full_text, symbol):
                continue

            age_hours = _parse_age(entry)
            if age_hours > max_age_hours:
                continue

            post_id = hashlib.md5(url_out.encode()).hexdigest()
            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            posts.append({
                "text":      full_text.strip(),
                "source":    source_name,
                "age_hours": round(age_hours, 1),
                "url":       url_out,
            })

    print(f"[Social] {symbol}: {len(posts)} relevant posts found")
    return posts


# ==============================
# Step 2: Market mood
# ==============================

def fetch_market_mood(max_age_hours: int = 24) -> dict:
    """
    Gauges overall retail investor mood from WSB and r/investing hot posts.
    Not stock-specific — tells you if the crowd is fearful or greedy.

    Returns: mood_label (GREED/NEUTRAL/FEAR), mood_score, post_count
    """
    scores = []

    for source_name, url in MOOD_RSS_FEEDS.items():
        entries = _fetch_rss(url, source_name)
        for entry in entries:
            title     = getattr(entry, "title",   "").strip()
            summary   = getattr(entry, "summary", "").strip()
            age_hours = _parse_age(entry)

            if not title or age_hours > max_age_hours:
                continue

            text   = f"{title} {summary[:150]}"
            result = analyzer.polarity_scores(text)
            scores.append(result["compound"])

    if not scores:
        return {"mood_label": "UNKNOWN", "mood_score": 0.0, "post_count": 0}

    avg = round(sum(scores) / len(scores), 3)

    if avg >= 0.15:
        mood_label = "GREED"
    elif avg <= -0.15:
        mood_label = "FEAR"
    else:
        mood_label = "NEUTRAL"

    return {"mood_label": mood_label, "mood_score": avg, "post_count": len(scores)}


# ==============================
# Step 3: Score posts
# ==============================

def score_posts(posts: list[dict]) -> dict:
    """
    Scores all posts with VADER + time decay.

    Time decay formula (same as news_sentiment.py):
      weight = 0.5 ^ (age_hours / 24)
      24h old = 0.5x weight | 48h = 0.25x weight

    Returns aggregated sentiment dict.
    """
    if not posts:
        return {
            "positive": 0, "negative": 0, "neutral": 0,
            "sentiment_score": 0.0, "label": "NEUTRAL",
            "scored_posts": [],
        }

    positive = negative = neutral = 0
    weighted_sum = 0.0
    total_weight = 0.0
    scored_posts = []

    for post in posts:
        text      = post.get("text", "")
        age_hours = post.get("age_hours", 0.0)

        if not text:
            continue

        result   = analyzer.polarity_scores(text)
        compound = result["compound"]

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

        scored_posts.append({**post, "compound": round(compound, 3), "label": label})

    sentiment_score = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0

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
        "positive": positive, "negative": negative, "neutral": neutral,
        "sentiment_score": sentiment_score, "label": overall_label,
        "scored_posts": scored_posts,
    }


# ==============================
# Step 4: Per-source breakdown
# ==============================

def _source_breakdown(scored_posts: list[dict]) -> dict:
    """
    Shows average sentiment per source.
    Useful to see if WSB and SeekingAlpha disagree.
    """
    sources = {}
    for post in scored_posts:
        src = post["source"]
        if src not in sources:
            sources[src] = {"count": 0, "score_sum": 0.0}
        sources[src]["count"]     += 1
        sources[src]["score_sum"] += post["compound"]

    breakdown = {}
    for src, data in sources.items():
        avg = round(data["score_sum"] / data["count"], 3) if data["count"] > 0 else 0.0
        breakdown[src] = {
            "count":     data["count"],
            "avg_score": avg,
            "label":     "POSITIVE" if avg >= 0.15 else ("NEGATIVE" if avg <= -0.15 else "NEUTRAL"),
        }
    return breakdown


# ==============================
# Main function
# ==============================

def get_social_sentiment(symbol: str) -> dict:
    """
    Full social sentiment for a stock symbol via RSS only.
    No API keys. No setup.

    Args:
      symbol : Ticker e.g. "AAPL"

    Returns dict with:
      symbol, post_count, sentiment, market_mood,
      source_breakdown, top_posts
    """
    posts       = fetch_social_posts(symbol, max_age_hours=48)
    market_mood = fetch_market_mood(max_age_hours=24)

    if not posts:
        return {
            "symbol":           symbol,
            "post_count":       0,
            "sentiment":        {"label": "NEUTRAL", "sentiment_score": 0.0,
                                 "positive": 0, "negative": 0, "neutral": 0},
            "market_mood":      market_mood,
            "source_breakdown": {},
            "top_posts":        [],
            "note":             "No posts found for this symbol in last 48h",
        }

    scored    = score_posts(posts)
    breakdown = _source_breakdown(scored["scored_posts"])

    # Top 5 by absolute signal strength
    top_posts = sorted(
        scored["scored_posts"],
        key=lambda p: abs(p["compound"]),
        reverse=True
    )[:5]

    return {
        "symbol":     symbol,
        "post_count": len(posts),
        "sentiment": {
            "label":           scored["label"],
            "sentiment_score": scored["sentiment_score"],
            "positive":        scored["positive"],
            "negative":        scored["negative"],
            "neutral":         scored["neutral"],
        },
        "market_mood":      market_mood,
        "source_breakdown": breakdown,
        "top_posts": [
            {
                "text":      p["text"][:120],
                "source":    p["source"],
                "compound":  p["compound"],
                "label":     p["label"],
                "age_hours": p["age_hours"],
            }
            for p in top_posts
        ],
    }


# ==============================
# Run directly to test
# ==============================

if __name__ == "__main__":

    test_symbols = ["AAPL", "TSLA"]

    for symbol in test_symbols:
        result = get_social_sentiment(symbol)

        print(f"\n{'='*55}")
        print(f"  SOCIAL SENTIMENT: {result['symbol']}")
        print(f"{'='*55}")

        print(f"\n  Posts analysed  : {result['post_count']}")

        s = result["sentiment"]
        print(f"\n  Overall label   : {s['label']}")
        print(f"  Sentiment score : {s['sentiment_score']:+.3f}")
        print(f"  Breakdown       : {s['positive']} pos / "
              f"{s['neutral']} neu / {s['negative']} neg")

        mood = result["market_mood"]
        print(f"\n  Market mood     : {mood['mood_label']}  "
              f"(score {mood['mood_score']:+.3f}, "
              f"{mood['post_count']} posts)")

        print(f"\n  Per-source breakdown:")
        for src, data in result["source_breakdown"].items():
            print(f"    {src:20s}  {data['count']:3d} posts  "
                  f"avg={data['avg_score']:+.3f}  {data['label']}")

        print(f"\n  Top Posts (strongest signal):")
        for p in result["top_posts"]:
            icon = "↑" if p["label"] == "POSITIVE" else (
                   "↓" if p["label"] == "NEGATIVE" else "→")
            print(f"    {icon} [{p['compound']:+.3f}] {p['text'][:65]}")
            print(f"         {p['source']}  •  {p['age_hours']}h ago")

        if "note" in result:
            print(f"\n  Note: {result['note']}")