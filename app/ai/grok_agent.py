"""
adk_agent.py
============
Three focused AI agents using Groq API + Llama 3.3 70B.

Why Groq:
  - Completely free tier, no credit card needed
  - Sign up at https://console.groq.com → API Keys → Create key
  - llama-3.3-70b-versatile: best quality on free tier
  - No daily quota issues unlike Google Gemini

Install:
  pip install groq

Set key:
  export GROQ_API_KEY=your_key_here

Three agents:

  1. extract_intent(query)
     Understands any natural language query including Hindi/mixed.
     Detects timeframe: SHORT_TERM / MEDIUM_TERM / LONG_TERM
     Input : "SBI mein 50000 long term ke liye lagao"
     Output: {symbol, company, budget, currency, quantity,
              action, timeframe}

  2. suggest_peers(symbol, company, sector, industry, country)
     Finds 3 real comparable stocks from same country + sector.
     Input : SBIN.NS, Financial Services, Banks, India
     Output: [{symbol, name, reason}]

  3. explain_and_recommend(payload, budget, currency, timeframe)
     Plain English recommendation + risk-adjusted quantity.
     Tone changes based on timeframe:
       SHORT_TERM → focuses on RSI, MACD, trend, news
       LONG_TERM  → focuses on CAGR, ROE, FCF, dividends,
                    5yr price return
     Input : full payload from data_builder + budget + timeframe
     Output: {recommendation, explanation, quantity,
              quantity_reasoning, key_points, risk_warning,
              disclaimer}

  4. validate_and_filter_peers(peers)
     Validates every ticker Groq suggests through yfinance.
     Removes hallucinated or delisted tickers.

Usage:
  from adk_agent import (
      extract_intent,
      suggest_peers,
      explain_and_recommend,
      validate_and_filter_peers,
  )
"""

import os
import re
import json
import time
from typing import Optional
from services.data_builder import build_payload
from groq import Groq
# from app.services.data_builder import build_payload


# ==============================
# Config
# ==============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# llama-3.3-70b-versatile — best quality on free tier
# llama-3.1-8b-instant    — faster, use if hitting rate limits
MODEL = "llama-3.3-70b-versatile"

DISCLAIMER = (
    "This analysis is for informational purposes only and does not "
    "constitute financial advice. Always consult a licensed financial "
    "advisor before making investment decisions."
)


# ==============================
# Core Groq caller
# ==============================

def _call_groq(
    system_prompt: str,
    user_prompt:   str,
    max_tokens:    int = 1000,
    max_retries:   int = 3,
) -> Optional[str]:
    """
    Makes a single Groq API call.
    Returns text response or None on failure.

    Auto-retries on 429 rate limit with backoff:
      Attempt 1 fails → wait 15s → attempt 2
      Attempt 2 fails → wait 30s → attempt 3
      Attempt 3 fails → return None
    """
    if not GROQ_API_KEY:
        print("[Groq] No GROQ_API_KEY set")
        return None

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model      = MODEL,
                max_tokens = max_tokens,
                messages   = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                # Tell Groq we want JSON — reduces formatting errors
                response_format = {"type": "json_object"},
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e)

            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = 15 * attempt
                print(f"[Groq] Rate limited (attempt {attempt}/{max_retries}) "
                      f"— waiting {wait}s...")
                time.sleep(wait)
                if attempt == max_retries:
                    print("[Groq] Max retries reached — using fallback")
                    return None
                continue

            print(f"[Groq] API error: {e}")
            return None

    return None


def _parse_json(text: str) -> Optional[dict]:
    """
    Safely parses JSON from Groq response.
    Groq's json_object mode usually returns clean JSON
    but we handle edge cases just in case.
    """
    if not text:
        return None
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        cleaned = cleaned.replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
        # Try to find JSON object within text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


# ==============================
# Agent 1: Intent Extraction
# ==============================

_INTENT_SYSTEM = """You are a financial query parser.
Extract structured information from stock queries in ANY language
including English, Hindi, Tamil, Kannada, mixed language etc.

RULES:
1. Return ONLY valid JSON. No explanations. No extra text.
2. Exchange suffixes:
   - Indian NSE stocks : .NS  (SBIN.NS, HDFCBANK.NS, INFY.NS, TCS.NS,
                                RELIANCE.NS, WIPRO.NS, KOTAKBANK.NS)
   - Indian BSE stocks : .BO
   - US stocks         : no suffix (AAPL, MSFT, GOOGL, TSLA, NVDA, META)
   - UK stocks         : .L
3. Currency detection:
   - ₹ / rupee / rupees / Rs / lakh / crore / inr → INR
   - $ / dollar / dollars / usd                   → USD
   - £ / pound / gbp                              → GBP
   - default if unclear                           → USD
4. action must be exactly one of: BUY, SELL, ANALYSE
5. timeframe detection:
   - "long term" / "long-term" / "sip" / "years" / "retirement" /
     "wealth" / "hold for" / "5 saal" / "future"   → LONG_TERM
   - "few months" / "6 months" / "quarterly" /
     "medium term"                                  → MEDIUM_TERM
   - everything else                               → SHORT_TERM
6. quantity: number of shares if mentioned, else null
7. budget: numeric amount if mentioned, else null
8. symbol: null if cannot be determined

Return EXACTLY this JSON:
{
  "symbol":    "TICKER",
  "company":   "Full Company Name",
  "budget":    50000.0,
  "currency":  "INR",
  "quantity":  null,
  "action":    "BUY",
  "timeframe": "SHORT_TERM"
}"""


def extract_intent(query: str) -> dict:
    """
    Understands a natural language stock query in any language.
    Detects symbol, budget, currency, quantity, action, timeframe.

    Args:
      query : Raw user input e.g. "SBI mein 50000 lagao long term"

    Returns structured dict. Falls back to safe defaults on failure.
    """
    if not GROQ_API_KEY:
        print("[Groq] No GROQ_API_KEY — using fallback")
        return _intent_fallback()

    response = _call_groq(
        system_prompt = _INTENT_SYSTEM,
        user_prompt   = f"Parse this stock query:\n\n{query}",
        max_tokens    = 300,
    )
    result = _parse_json(response)

    if result and result.get("symbol"):
        # Sanitise timeframe
        if result.get("timeframe") not in ("SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"):
            result["timeframe"] = "SHORT_TERM"
        print(
            f"[Groq] Intent: {result.get('symbol')} | "
            f"{result.get('budget')} {result.get('currency')} | "
            f"{result.get('timeframe')} | {result.get('action')}"
        )
        return result

    print("[Groq] Intent extraction failed — using fallback")
    return _intent_fallback()


def _intent_fallback() -> dict:
    return {
        "symbol":    None,
        "company":   None,
        "budget":    None,
        "currency":  "USD",
        "quantity":  None,
        "action":    "ANALYSE",
        "timeframe": "SHORT_TERM",
    }


# ==============================
# Agent 2: Peer Suggestion
# ==============================

_PEERS_SYSTEM = """You are a stock market expert specialising in peer comparison.
Suggest exactly 3 comparable peer stocks for investment comparison.

RULES:
1. Return ONLY valid JSON. No explanations. No extra text.
2. Peers MUST be from the SAME country as the asked stock.
3. Correct exchange suffixes:
   - India NSE : .NS  Examples: HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS,
                                 AXISBANK.NS, WIPRO.NS, TCS.NS, HCLTECH.NS,
                                 TATASTEEL.NS, MARUTI.NS, BAJFINANCE.NS
   - India BSE : .BO
   - USA       : no suffix  Examples: JPM, BAC, WFC, GS, MSFT, GOOGL, META
   - UK        : .L
4. Large-cap, liquid, well-known stocks only.
5. NEVER suggest the asked stock itself.
6. NEVER suggest ETFs.
7. NEVER suggest stocks from a different country.
8. reason = one short sentence why comparable.

Good peer examples:
  SBIN.NS (India bank)  → HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS
  INFY (India IT)       → TCS.NS, WIPRO.NS, HCLTECH.NS
  AAPL (US tech)        → MSFT, GOOGL, META
  TSLA (US auto/tech)   → RIVN, GM, F
  JPM (US bank)         → BAC, WFC, GS

Return EXACTLY this JSON:
{
  "peers": [
    {"symbol": "TICKER1", "name": "Company 1", "reason": "Why comparable"},
    {"symbol": "TICKER2", "name": "Company 2", "reason": "Why comparable"},
    {"symbol": "TICKER3", "name": "Company 3", "reason": "Why comparable"}
  ]
}"""


def suggest_peers(
    symbol:   str,
    company:  str,
    sector:   str,
    industry: str,
    country:  str,
) -> list[dict]:
    """
    Suggests 3 real comparable stocks from same country and sector.

    Args:
      symbol   : e.g. "SBIN.NS"
      company  : e.g. "State Bank of India"
      sector   : e.g. "Financial Services"
      industry : e.g. "Banks - Regional"
      country  : e.g. "India"

    Returns list of [{symbol, name, reason}].
    Falls back to empty list on failure.
    """
    if not GROQ_API_KEY:
        print("[Groq] No GROQ_API_KEY — returning empty peers")
        return []

    prompt = (
        f"Find 3 peer stocks comparable to:\n\n"
        f"Symbol  : {symbol}\n"
        f"Company : {company}\n"
        f"Sector  : {sector}\n"
        f"Industry: {industry}\n"
        f"Country : {country}\n\n"
        f"Return peers from {country} in the {sector} sector only."
    )

    response = _call_groq(
        system_prompt = _PEERS_SYSTEM,
        user_prompt   = prompt,
        max_tokens    = 400,
    )
    result = _parse_json(response)

    if result and "peers" in result:
        peers = result["peers"]
        print(f"[Groq] Peers suggested: {[p['symbol'] for p in peers]}")
        return peers

    print("[Groq] Peer suggestion failed — returning empty list")
    return []


# ==============================
# Agent 3: Explain and Recommend
# ==============================

_EXPLAIN_SYSTEM = """You are a professional financial analyst assistant.
Explain stock analysis results in clear, simple language for retail investors.

RULES:
1. Return ONLY valid JSON. No explanations. No extra text.
2. Plain simple English — no jargon.
3. Be honest about risks — do not oversell.

4. Quantity calculation:
   base_qty = floor(budget / current_price)
   Risk allocation:
     EXTREME risk → 30% of budget
     HIGH risk    → 50% of budget
     NORMAL risk  → 80% of budget
     LOW risk     → 95% of budget
   If no budget → quantity = null

5. Tone based on timeframe:
   SHORT_TERM  → focus on RSI, MACD, trend, news sentiment.
                 Language: "right now", "entry point", "momentum"
   MEDIUM_TERM → balance technical and fundamental factors.
                 Language: "over the next few months"
   LONG_TERM   → focus on 3yr CAGR, ROE, FCF, 5yr price return,
                 dividends. IGNORE short-term RSI/MACD noise.
                 Language: "long-term compounding", "business quality",
                 "hold for years", "patient investor"

6. key_points: exactly 3 bullet points
7. risk_warning: honest, specific, 1-2 sentences
8. recommendation must be one of:
   STRONG BUY / BUY / HOLD / SELL / STRONG SELL / AVOID

Return EXACTLY this JSON:
{
  "recommendation":      "BUY",
  "explanation":         "2-3 sentence plain English explanation",
  "quantity":            47,
  "quantity_reasoning":  "At INR 1047/share, 50% of INR 50000 = INR 25000 / 1047 = 23 shares (HIGH risk)",
  "key_points":          ["Point 1", "Point 2", "Point 3"],
  "risk_warning":        "Specific risk warning",
  "disclaimer":          "This analysis is for informational purposes only and does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions."
}"""


def explain_and_recommend(
    payload:   dict,
    budget:    Optional[float],
    currency:  str = "USD",
    timeframe: str = "SHORT_TERM",
) -> dict:
    """
    Explains the full analysis in plain English with quantity advice.
    Tone changes based on timeframe.

    Args:
      payload   : Full dict from data_builder.build_payload()
      budget    : User's budget e.g. 50000.0
      currency  : "INR" / "USD" etc.
      timeframe : "SHORT_TERM" / "MEDIUM_TERM" / "LONG_TERM"

    Returns dict with recommendation, explanation, quantity etc.
    Falls back to structured defaults if API fails.
    """
    if not GROQ_API_KEY:
        print("[Groq] No GROQ_API_KEY — using fallback")
        return _explain_fallback(payload, budget, currency)

    tech = payload.get("technical",      {})
    news = payload.get("news",           {})
    soc  = payload.get("social",         {})
    fund = payload.get("fundamentals",   {})
    conf = payload.get("conflict",       {})
    wp   = payload.get("weight_profile", {})

    # Build helper strings safely
    def _pct(val, mult=100):
        return f"{val * mult:+.1f}%" if val is not None else "N/A"

    def _val(val, fmt=".2f"):
        return format(val, fmt) if val is not None else "N/A"

    summary = f"""
STOCK ANALYSIS [{timeframe}]
============================
Stock         : {payload.get('symbol')} — {payload.get('company_name')}
Current Price : {payload.get('current_price')} {currency}
Budget        : {budget} {currency}

TECHNICAL {"(primary)" if timeframe == "SHORT_TERM" else "(secondary for long-term)"}
  Signal     : {tech.get('signal')} (strength {tech.get('strength')}/100)
  Trend      : {tech.get('trend')}
  Momentum   : {tech.get('momentum')} (RSI {tech.get('rsi')})
  Risk Level : {tech.get('risk_level')}
  SMA 50/200 : {tech.get('sma_50')} / {tech.get('sma_200')}
  BB Position: {tech.get('bb_position')}
  Volatility : {tech.get('volatility')}

NEWS
  Sentiment  : {news.get('signal')} (score {news.get('sentiment_score')})
  Headlines  : {news.get('headline_count')}
  Crisis     : {news.get('crisis', {}).get('type', 'None detected')}
  Earnings   : {'WITHIN 5 DAYS — be cautious' if news.get('earnings_window') else 'Not imminent'}

SOCIAL
  Sentiment  : {soc.get('signal')} (score {soc.get('sentiment_score')})
  Market Mood: {soc.get('market_mood', {}).get('mood_label', 'UNKNOWN')}

FUNDAMENTALS {"(primary for long-term)" if timeframe == "LONG_TERM" else "(supporting)"}
  Health       : {fund.get('health_label')} ({fund.get('health_score')}/100)
  Revenue YoY  : {_pct(fund.get('annual_revenue_growth'))} [{fund.get('revenue_period', '')}]
  Qtr Trend    : {fund.get('revenue_trend')}
  Margin       : {_pct(fund.get('profit_margin'), 100)} [{fund.get('profit_margin_label')}]
  Debt/Equity  : {_val(fund.get('debt_to_equity'))} [{fund.get('debt_label')}]
  P/E Ratio    : {_val(fund.get('pe_ratio'))} [{fund.get('pe_label')}]
  Earnings     : {' / '.join(fund.get('earnings_record', [])) or 'N/A'} [{fund.get('earnings_label')}]
  3yr CAGR     : {_pct(fund.get('cagr_3yr'))} [{fund.get('cagr_label', 'N/A')}]
  5yr Price Ret: {_pct(fund.get('price_return_5yr'))} p.a. [{fund.get('price_return_label', 'N/A')}]
  ROE          : {_pct(fund.get('roe'))} [{fund.get('roe_label', 'N/A')}]
  Free Cash    : {fund.get('fcf_label', 'N/A')}
  Dividend     : {_pct(fund.get('dividend_yield'))} | {fund.get('dividend_consistency', 'N/A')} | Growing: {fund.get('dividend_growing')}

DECISION
  Final Signal  : {payload.get('final_signal')}
  Confidence    : {payload.get('final_confidence')}/100
  Weight Profile: {wp.get('name')} — {wp.get('reason')}
  Conflict      : {conf.get('conflict_type')} [{conf.get('conflict_severity')}]

KEY NOTES:
{chr(10).join(f"  • {n}" for n in payload.get('notes', [])[:5])}
"""

    timeframe_hint = {
        "SHORT_TERM":  "Focus on RSI, MACD, trend and news for a short-term trade.",
        "MEDIUM_TERM": "Balance technical signals with fundamental health.",
        "LONG_TERM":   "Focus on 3yr CAGR, ROE, FCF, 5yr price return, dividends. Ignore RSI/MACD.",
    }.get(timeframe, "")

    response = _call_groq(
        system_prompt = _EXPLAIN_SYSTEM,
        user_prompt   = (
            f"Explain this stock analysis and give a recommendation.\n"
            f"{timeframe_hint}\n\n{summary}"
        ),
        max_tokens = 1000,
    )
    result = _parse_json(response)

    if result and "recommendation" in result:
        result["disclaimer"] = DISCLAIMER   # always hardcode
        result["timeframe"]  = timeframe
        print(
            f"[Groq] Recommendation: {result.get('recommendation')} | "
            f"Qty: {result.get('quantity')} | "
            f"Timeframe: {timeframe}"
        )
        return result

    print("[Groq] Explanation failed — using fallback")
    return _explain_fallback(payload, budget, currency)


def _explain_fallback(
    payload:  dict,
    budget:   Optional[float],
    currency: str,
) -> dict:
    """Fallback when Groq is unavailable — math-based quantity, no AI."""
    tech  = payload.get("technical", {})
    price = payload.get("current_price")
    risk  = tech.get("risk_level", "NORMAL")

    quantity   = None
    qty_reason = "Budget not provided"

    if budget and price and price > 0:
        alloc = {"EXTREME": 0.30, "HIGH": 0.50, "NORMAL": 0.80, "LOW": 0.95}.get(risk, 0.80)
        quantity   = int((budget * alloc) / price)
        qty_reason = (
            f"At {currency} {price:,.2f}/share, "
            f"{alloc:.0%} of {currency} {budget:,.0f} = "
            f"{currency} {budget*alloc:,.0f} / {price:,.2f} = "
            f"{quantity} shares ({risk} risk)"
        )

    return {
        "recommendation":    payload.get("final_signal", "HOLD"),
        "explanation":       (
            f"{payload.get('symbol')} signal is {payload.get('final_signal')} "
            f"with {payload.get('final_confidence')}/100 confidence. "
            f"Fundamentals: {payload.get('fundamentals', {}).get('health_label', 'unknown')}."
        ),
        "quantity":           quantity,
        "quantity_reasoning": qty_reason,
        "key_points":         payload.get("notes", ["Analysis complete"])[:3],
        "risk_warning":       f"Risk level is {risk}. Invest only what you can afford to lose.",
        "timeframe":          payload.get("timeframe", "SHORT_TERM"),
        "disclaimer":         DISCLAIMER,
    }


# ==============================
# Peer validation
# ==============================

def validate_and_filter_peers(peers: list[dict]) -> list[dict]:
    """
    Validates each ticker Groq suggests via yfinance.
    Removes hallucinated or delisted tickers before analysis.
    Always call this before passing peers to compare_builder.
    """
    import yfinance as yf
    valid = []
    for peer in peers:
        symbol = peer.get("symbol", "")
        if not symbol:
            continue
        try:
            info = yf.Ticker(symbol).info
            if info.get("regularMarketPrice") or info.get("currentPrice"):
                valid.append(peer)
                print(f"[Groq] Peer valid   : {symbol}")
            else:
                print(f"[Groq] Peer invalid : {symbol} (no price) — skipping")
        except Exception:
            print(f"[Groq] Peer invalid : {symbol} (error) — skipping")
    return valid


# ==============================
# Run directly to test all 3 agents
# ==============================

if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if not GROQ_API_KEY:
        print("\nNo GROQ_API_KEY found.")
        print("Get a free key at : https://console.groq.com")
        print("Then run          : export GROQ_API_KEY=your_key_here")
        exit()

    # ── Test 1: Intent extraction ─────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Intent Extraction")
    print("="*60)

    queries = [
        "Should I buy AAPL with $5000?",
        "SBI mein 50000 rupaye lagao",
        "Is TSLA worth buying long term? I want 10 shares",
        "Should I sell my HDFC Bank shares?",
        "Tell me about Infosys for next 6 months with 1 lakh",
        "RELIANCE mein 5 saal ke liye invest karna chahta hoon",
    ]

    for q in queries:
        r = extract_intent(q)
        print(f"\n  Query     : {q}")
        print(f"  Symbol    : {str(r.get('symbol') or 'None'):<15s}  "
              f"Budget: {r.get('budget')} {r.get('currency')}")
        print(f"  Timeframe : {str(r.get('timeframe') or 'None'):<15s}  "
              f"Action: {r.get('action')}/n/n")

    # ── Test 2: Peer suggestion ───────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: Peer Suggestion (country-aware)")
    print("="*60)

    cases = [
        ("SBIN.NS", "State Bank of India", "Financial Services", "Banks - Regional",    "India"),
        ("AAPL",    "Apple Inc",           "Technology",         "Consumer Electronics","United States"),
        ("INFY",    "Infosys",             "Technology",         "IT Services",         "India"),
    ]

    for symbol, company, sector, industry, country in cases:
        peers = suggest_peers(symbol, company, sector, industry, country)
        valid = validate_and_filter_peers(peers)
        print(f"\n  {symbol} ({country} / {sector})")
        print(f"  Groq suggested : {[p['symbol'] for p in peers]}")
        print(f"  After validate : {[p['symbol'] for p in valid]}")
        for p in valid:
            print(f"    → {p['symbol']:<15s}  {p['name']:<25s}  {p['reason']}")

    # ── Test 3: Explain and recommend ────────────────────────────
    print("\n" + "="*60)
    print("TEST 3: Explain and Recommend (all 3 timeframes)")
    print("="*60)

    async def test_explain():
        test_cases = [
            ("SBIN.NS", "State Bank of India", 50000.0, "INR", "SHORT_TERM"),
            ("SBIN.NS", "State Bank of India", 50000.0, "INR", "LONG_TERM"),
            ("AAPL",    "Apple Inc",           5000.0,  "USD", "SHORT_TERM"),
        ]

        for symbol, company, budget, currency, timeframe in test_cases:
            print(f"\n  Fetching data for {symbol} [{timeframe}]...")
            try:
                payload = await build_payload(symbol, company)
            except Exception as e:
                print(f"  data_builder error: {e}")
                continue

            result = explain_and_recommend(
                payload   = payload,
                budget    = budget,
                currency  = currency,
                timeframe = timeframe,
            )

            print(f"\n  {'─'*50}")
            print(f"  {symbol}  [{timeframe}]  Budget: {currency} {budget:,.0f}")
            print(f"  {'─'*50}")
            print(f"  Recommendation  : {result.get('recommendation')}")
            print(f"  Quantity        : {result.get('quantity')} shares")
            print(f"  Qty Reasoning   : {result.get('quantity_reasoning')}")
            print(f"\n  Explanation:")
            print(f"    {result.get('explanation')}")
            print(f"\n  Key Points:")
            for pt in result.get("key_points", []):
                print(f"    • {pt}")
            print(f"\n  Risk Warning    : {result.get('risk_warning')}")
            print(f"  Disclaimer      : {result.get('disclaimer')[:60]}...")

    asyncio.run(test_explain())