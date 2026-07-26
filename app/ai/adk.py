"""
adk_agent.py
============
Three focused AI agents using Google ADK + Gemini 2.0 Flash.

Free tier:
  - Get key at https://aistudio.google.com/apikey
  - No credit card needed
  - Gemini 2.5 Flash-Lite: 1000 requests/day free (gemini-2.0 retired Mar 2026)

Install:
  pip install google-adk

Set key:
  export GOOGLE_API_KEY=your_key_here

Four functions:

  1. extract_intent(query)
     Understands any natural language query including Hindi/mixed.
     Now also detects timeframe: SHORT_TERM / MEDIUM_TERM / LONG_TERM
     Input : "SBI mein 50000 long term ke liye lagao"
     Output: {symbol, company, budget, currency, quantity,
              action, timeframe}

  2. suggest_peers(symbol, company, sector, industry, country)
     Finds 3 real comparable stocks from same country + sector.
     Input : SBIN.NS, Financial Services, Banks, India
     Output: [{symbol, name, reason}]

  3. explain_and_recommend(payload, budget, currency, timeframe)
     Plain English recommendation + risk-adjusted quantity.
     Explanation tone changes based on timeframe:
       SHORT_TERM → focuses on RSI, MACD, trend, news
       LONG_TERM  → focuses on CAGR, ROE, FCF, dividends,
                    5yr price return
     Input : full payload + budget + timeframe
     Output: {recommendation, explanation, quantity,
              quantity_reasoning, key_points, risk_warning,
              disclaimer}

  4. validate_and_filter_peers(peers)
     Validates every ticker ADK suggests through yfinance.
     Removes hallucinated or delisted tickers before analysis.

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
import uuid
import asyncio
from typing import Optional

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part


# ==============================
# Config
# ==============================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL          = "gemini-2.5-flash"   # 1000 RPD free, gemini-2.0 retired Mar 3 2026
APP_NAME       = "trading_assistant"

DISCLAIMER = (
    "This analysis is for informational purposes only and does not "
    "constitute financial advice. Always consult a licensed financial "
    "advisor before making investment decisions."
)


# ==============================
# Core ADK runner
# ==============================

async def _run_agent(
    agent:       LlmAgent,
    prompt:      str,
    max_retries: int = 5,
) -> Optional[str]:
    """
    Runs an ADK LlmAgent with a single prompt.
    Creates a fresh stateless session per call.
    Returns final text response or None on failure.

    Auto-retries on 429 rate limit errors with exponential backoff:
      Attempt 1 fails → wait 15s → attempt 2
      Attempt 2 fails → wait 30s → attempt 3
      Attempt 3 fails → return None
    """
    import asyncio as _asyncio

    for attempt in range(1, max_retries + 1):
        try:
            session_service = InMemorySessionService()
            session_id      = str(uuid.uuid4())
            user_id         = "trading_user"

            await session_service.create_session(
                app_name   = APP_NAME,
                user_id    = user_id,
                session_id = session_id,
            )

            runner = Runner(
                agent           = agent,
                app_name        = APP_NAME,
                session_service = session_service,
            )

            message = Content(
                role  = "user",
                parts = [Part(text=prompt)],
            )

            final_response = None

            async for event in runner.run_async(
                user_id     = user_id,
                session_id  = session_id,
                new_message = message,
            ):
                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response = event.content.parts[0].text
                        break

            return final_response

        except Exception as e:
            error_str = str(e)

            # Handle 429 rate limit — wait and retry
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_seconds = 15 * attempt   # 15s, 30s, 45s
                print(f"[ADK] Rate limited (attempt {attempt}/{max_retries}) "
                      f"— waiting {wait_seconds}s before retry...")
                await _asyncio.sleep(wait_seconds)
                if attempt == max_retries:
                    print("[ADK] Max retries reached — using fallback")
                    return None
                continue

            # Any other error — fail immediately
            print(f"[ADK] Agent error: {e}")
            return None

    return None


def _parse_json(text: str) -> Optional[dict]:
    """
    Safely extracts JSON from Gemini response.
    Handles markdown code fences Gemini sometimes adds.
    """
    if not text:
        return None
    try:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
        cleaned = cleaned.replace("```", "").strip()
        return json.loads(cleaned)
    except Exception:
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

_intent_agent = LlmAgent(
    name        = "intent_agent",
    model       = MODEL,
    description = "Extracts structured data from stock queries in any language",
    instruction = """You are a financial query parser.
Extract structured information from stock queries in ANY language
including English, Hindi, Tamil, mixed language etc.

CRITICAL RULES:
1. Return ONLY valid JSON. No explanations. No markdown. No code blocks.
2. Exchange suffixes:
   - Indian NSE stocks : .NS  (SBIN.NS, HDFCBANK.NS, INFY.NS, TCS.NS)
   - Indian BSE stocks : .BO
   - US stocks         : no suffix (AAPL, MSFT, GOOGL, TSLA, NVDA)
   - UK stocks         : .L
   - German stocks     : .DE
3. Currency detection:
   - ₹ / rupee / rupees / Rs / lakh / crore / inr → INR
   - $ / dollar / dollars / usd                   → USD
   - £ / pound / gbp                              → GBP
   - default                                      → USD
4. action must be exactly: BUY, SELL, or ANALYSE
5. timeframe detection:
   - "long term" / "long-term" / "sip" / "years" / "retirement"
     "wealth" / "hold for" / "future" / "5 saal" → LONG_TERM
   - "few months" / "6 months" / "quarterly"     → MEDIUM_TERM
   - everything else                              → SHORT_TERM
6. If quantity not mentioned → null
7. If budget not mentioned → null
8. If symbol cannot be determined → null

Return EXACTLY this JSON and nothing else:
{
  "symbol":    "TICKER.SUFFIX",
  "company":   "Full Company Name",
  "budget":    50000.0,
  "currency":  "INR",
  "quantity":  null,
  "action":    "BUY",
  "timeframe": "SHORT_TERM"
}"""
)


def extract_intent(query: str) -> dict:
    """
    Understands a natural language stock query in any language.
    Now also detects timeframe intent.

    Args:
      query : Raw user input

    Returns:
      {symbol, company, budget, currency, quantity, action, timeframe}

    Fallback: safe defaults with timeframe=SHORT_TERM if API fails.
    """
    if not GOOGLE_API_KEY:
        print("[ADK] No GOOGLE_API_KEY — using fallback")
        return _intent_fallback()

    prompt = f"Parse this stock query and return JSON only:\n\n{query}"

    try:
        response = asyncio.run(_run_agent(_intent_agent, prompt))
        result   = _parse_json(response)

        if result and result.get("symbol"):
            # Ensure timeframe always has a valid value
            if result.get("timeframe") not in ("SHORT_TERM", "MEDIUM_TERM", "LONG_TERM"):
                result["timeframe"] = "SHORT_TERM"
            print(
                f"[ADK] Intent: symbol={result.get('symbol')} | "
                f"budget={result.get('budget')} {result.get('currency')} | "
                f"timeframe={result.get('timeframe')} | "
                f"action={result.get('action')}"
            )
            return result

    except Exception as e:
        print(f"[ADK] Intent agent error: {e}")

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

_peers_agent = LlmAgent(
    name        = "peers_agent",
    model       = MODEL,
    description = "Suggests 3 real comparable stocks from same country and sector",
    instruction = """You are a stock market expert specialising in peer comparison.

Given a stock, suggest exactly 3 comparable peer stocks.

CRITICAL RULES:
1. Return ONLY valid JSON. No explanations. No markdown. No code blocks.
2. Peers MUST be from the SAME country as the asked stock.
3. Use correct exchange suffixes:
   - India NSE : .NS  (HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS,
                        WIPRO.NS, TCS.NS, TATASTEEL.NS)
   - India BSE : .BO
   - USA       : no suffix  (JPM, BAC, WFC, MSFT, GOOGL, NVDA)
   - UK        : .L
   - Germany   : .DE
4. Choose well-known, LIQUID, large-cap stocks only.
5. NEVER suggest the asked stock itself as a peer.
6. NEVER suggest ETFs unless no individual stocks exist.
7. NEVER suggest stocks from a different country.
8. reason = one short sentence explaining why comparable.

Examples of good peers:
  SBIN.NS (India banking)  → HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS
  INFY (India IT)          → TCS.NS, WIPRO.NS, HCLTECH.NS
  AAPL (US tech)           → MSFT, GOOGL, META
  JPM (US banking)         → BAC, WFC, GS

Return EXACTLY this JSON and nothing else:
{
  "peers": [
    {"symbol": "TICKER1", "name": "Company Name 1", "reason": "Why comparable"},
    {"symbol": "TICKER2", "name": "Company Name 2", "reason": "Why comparable"},
    {"symbol": "TICKER3", "name": "Company Name 3", "reason": "Why comparable"}
  ]
}"""
)


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

    Returns list of dicts: [{symbol, name, reason}]
    Falls back to empty list if API fails.
    """
    if not GOOGLE_API_KEY:
        print("[ADK] No GOOGLE_API_KEY — returning empty peers")
        return []

    prompt = (
        f"Find 3 peer stocks comparable to:\n\n"
        f"Symbol  : {symbol}\n"
        f"Company : {company}\n"
        f"Sector  : {sector}\n"
        f"Industry: {industry}\n"
        f"Country : {country}\n\n"
        f"Return peers from {country} only, in the {sector} sector."
    )

    try:
        response = asyncio.run(_run_agent(_peers_agent, prompt))
        result   = _parse_json(response)

        if result and "peers" in result:
            peers = result["peers"]
            print(f"[ADK] Peers suggested: {[p['symbol'] for p in peers]}")
            return peers

    except Exception as e:
        print(f"[ADK] Peers agent error: {e}")

    return []


# ==============================
# Agent 3: Explain and Recommend
# ==============================

_explain_agent = LlmAgent(
    name        = "explain_agent",
    model       = MODEL,
    description = "Explains stock analysis in plain English with quantity recommendation",
    instruction = """You are a professional financial analyst assistant.
Explain stock analysis results clearly for everyday retail investors.

CRITICAL RULES:
1. Return ONLY valid JSON. No explanations. No markdown. No code blocks.
2. Plain simple English — no jargon.
3. Be honest about risks — do not oversell.

4. Quantity calculation (IMPORTANT):
   base_qty = floor(budget / current_price)
   Risk adjustment:
     EXTREME risk → use 30% of budget
     HIGH risk    → use 50% of budget
     NORMAL risk  → use 80% of budget
     LOW risk     → use 95% of budget
   If no budget provided → quantity = null

5. Explanation tone based on timeframe:
   SHORT_TERM → focus on RSI, MACD, trend, news sentiment,
                recent momentum. Language: "right now", "entry point",
                "short-term momentum"
   MEDIUM_TERM → balance technical and fundamental factors.
                Language: "over the next few months", "medium-term hold"
   LONG_TERM  → focus on 3yr CAGR, ROE, FCF, 5yr price return,
                dividends, P/E vs growth. Ignore short-term RSI/MACD.
                Language: "long-term compounding", "business quality",
                "hold for years"

6. key_points: exactly 3 bullet points — most important observations
7. risk_warning: honest, specific, 1-2 sentences
8. recommendation: one of:
   STRONG BUY / BUY / HOLD / SELL / STRONG SELL / AVOID

Return EXACTLY this JSON and nothing else:
{
  "recommendation":      "BUY",
  "explanation":         "2-3 sentence plain English explanation",
  "quantity":            47,
  "quantity_reasoning":  "At INR 1047 per share, 50% of INR 50000 budget = INR 25000 / 1047 = 23 shares (HIGH risk)",
  "key_points":          ["Point 1", "Point 2", "Point 3"],
  "risk_warning":        "Specific risk warning",
  "disclaimer":          "This analysis is for informational purposes only and does not constitute financial advice. Always consult a licensed financial advisor before making investment decisions."
}"""
)


def explain_and_recommend(
    payload:   dict,
    budget:    Optional[float],
    currency:  str = "USD",
    timeframe: str = "SHORT_TERM",
) -> dict:
    """
    Explains the full analysis in plain English.
    Tone and focus change based on timeframe.

    Args:
      payload   : Full dict from data_builder.build_payload()
      budget    : User's budget e.g. 50000.0
      currency  : "INR" or "USD" etc.
      timeframe : "SHORT_TERM" / "MEDIUM_TERM" / "LONG_TERM"

    Returns dict with recommendation, explanation, quantity etc.
    Falls back to structured defaults if API fails.
    """
    if not GOOGLE_API_KEY:
        print("[ADK] No GOOGLE_API_KEY — using fallback explanation")
        return _explain_fallback(payload, budget, currency)

    tech = payload.get("technical",     {})
    news = payload.get("news",          {})
    soc  = payload.get("social",        {})
    fund = payload.get("fundamentals",  {})
    conf = payload.get("conflict",      {})
    wp   = payload.get("weight_profile",{})

    # ── Build analysis summary ─────────────────────────────────
    # Short/medium term: technical + news dominate the summary
    # Long term: fundamentals section is extended

    rev_str = (
        f"{fund.get('annual_revenue_growth', 0) * 100:+.1f}% YoY "
        f"({fund.get('revenue_period', '')})"
        if fund.get("annual_revenue_growth") is not None else "N/A"
    )
    margin_str = (
        f"{fund.get('profit_margin', 0) * 100:.1f}%"
        if fund.get("profit_margin") is not None else "N/A"
    )

    # Long-term extras
    cagr_str = (
        f"{fund.get('cagr_3yr', 0) * 100:+.1f}% (3yr)"
        if fund.get("cagr_3yr") is not None else "N/A"
    )
    price_ret_str = (
        f"{fund.get('price_return_5yr', 0) * 100:+.1f}% p.a."
        if fund.get("price_return_5yr") is not None else "N/A"
    )
    roe_str = (
        f"{fund.get('roe', 0) * 100:.1f}%"
        if fund.get("roe") is not None else "N/A"
    )
    fcf_label   = fund.get("fcf_label",            "UNKNOWN")
    div_cons    = fund.get("dividend_consistency",  "NONE")
    div_yield   = fund.get("dividend_yield")
    div_str     = f"{div_yield * 100:.2f}%" if div_yield else "None"
    div_growing = fund.get("dividend_growing")

    summary = f"""
STOCK ANALYSIS — {timeframe}
============================
Stock         : {payload.get('symbol')} — {payload.get('company_name')}
Current Price : {payload.get('current_price')} {currency}
User Budget   : {budget} {currency}
Timeframe     : {timeframe}

TECHNICAL ({"primary signal" if timeframe == "SHORT_TERM" else "secondary signal for long-term"})
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
  Earnings   : {'⚠ WITHIN 5 DAYS — be cautious' if news.get('earnings_window') else 'Not imminent'}

SOCIAL
  Sentiment  : {soc.get('signal')} (score {soc.get('sentiment_score')})
  Market Mood: {soc.get('market_mood', {}).get('mood_label', 'UNKNOWN')}

FUNDAMENTALS ({"primary signal for long-term" if timeframe == "LONG_TERM" else "supporting signal"})
  Health     : {fund.get('health_label')} ({fund.get('health_score')}/100)
  Revenue YoY: {rev_str} [Trend: {fund.get('revenue_trend')}]
  Margin     : {margin_str} [{fund.get('profit_margin_label')}]
  Debt/Equity: {fund.get('debt_to_equity')} [{fund.get('debt_label')}]
  P/E        : {fund.get('pe_ratio')} [{fund.get('pe_label')}]
  Earnings   : {' / '.join(fund.get('earnings_record', [])) or 'N/A'} [{fund.get('earnings_label')}]
  3yr CAGR   : {cagr_str} [{fund.get('cagr_label', 'N/A')}]
  5yr Price  : {price_ret_str} [{fund.get('price_return_label', 'N/A')}]
  ROE        : {roe_str} [{fund.get('roe_label', 'N/A')}]
  Free CF    : {fcf_label}
  Dividend   : {div_str} | Consistency: {div_cons} | Growing: {div_growing}

DECISION
  Final Signal  : {payload.get('final_signal')}
  Confidence    : {payload.get('final_confidence')}/100
  Weight Profile: {wp.get('name')} — {wp.get('reason')}
  Conflict      : {conf.get('conflict_type')} [{conf.get('conflict_severity')}]

KEY OBSERVATIONS:
{chr(10).join(f"  • {n}" for n in payload.get('notes', [])[:6])}
"""

    timeframe_instruction = {
        "SHORT_TERM":  "Focus on RSI, MACD, trend, and news for a short-term trade.",
        "MEDIUM_TERM": "Balance technical signals with fundamental health for a medium-term hold.",
        "LONG_TERM":   "Focus on 3yr CAGR, ROE, FCF, 5yr price return, and dividends. Ignore short-term RSI/MACD noise.",
    }.get(timeframe, "")

    prompt = (
        f"Explain this stock analysis and give a recommendation.\n"
        f"{timeframe_instruction}\n"
        f"Return JSON only.\n\n{summary}"
    )

    try:
        response = asyncio.run(_run_agent(_explain_agent, prompt))
        result   = _parse_json(response)

        if result and "recommendation" in result:
            # Always hardcode disclaimer — never trust AI to include it
            result["disclaimer"] = DISCLAIMER
            result["timeframe"]  = timeframe
            print(
                f"[ADK] Recommendation: {result.get('recommendation')} | "
                f"Qty: {result.get('quantity')} | "
                f"Timeframe: {timeframe}"
            )
            return result

    except Exception as e:
        print(f"[ADK] Explain agent error: {e}")

    return _explain_fallback(payload, budget, currency)


def _explain_fallback(
    payload:   dict,
    budget:    Optional[float],
    currency:  str,
) -> dict:
    """Fallback when ADK is unavailable — basic calculation without AI."""
    tech  = payload.get("technical", {})
    price = payload.get("current_price")
    risk  = tech.get("risk_level", "NORMAL")

    quantity   = None
    qty_reason = "Budget not provided"

    if budget and price and price > 0:
        multipliers = {
            "EXTREME": 0.30,
            "HIGH":    0.50,
            "NORMAL":  0.80,
            "LOW":     0.95,
        }
        allocation = multipliers.get(risk, 0.80)
        quantity   = int((budget * allocation) / price)
        qty_reason = (
            f"At {currency} {price:,.2f} per share, "
            f"{allocation:.0%} of {currency} {budget:,.0f} budget "
            f"= {currency} {budget * allocation:,.0f} / {price:,.2f} "
            f"= {quantity} shares ({risk} risk)"
        )

    return {
        "recommendation":    payload.get("final_signal", "HOLD"),
        "explanation":       (
            f"{payload.get('symbol')} has a {payload.get('final_signal')} signal "
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
    Validates each ticker ADK suggests through yfinance.
    Removes hallucinated or delisted symbols before analysis.
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
                print(f"[ADK] Peer valid   : {symbol}")
            else:
                print(f"[ADK] Peer invalid : {symbol} (no price) — skipping")
        except Exception:
            print(f"[ADK] Peer invalid : {symbol} (error) — skipping")
    return valid


# ==============================
# Run directly to test all 3 agents
# ==============================

if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])

    if not GOOGLE_API_KEY:
        print("\nNo GOOGLE_API_KEY found.")
        print("Get a free key at : https://aistudio.google.com/apikey")
        print("Then run          : export GOOGLE_API_KEY=your_key_here")
        exit()

    # ── Test 1: Intent extraction ─────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: Intent Extraction (with timeframe detection)")
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
              f"Action: {r.get('action')}")

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
        print(f"  ADK suggested  : {[p['symbol'] for p in peers]}")
        print(f"  After validate : {[p['symbol'] for p in valid]}")
        for p in valid:
            print(f"    → {p['symbol']:15s}  {p['name']:25s}  {p['reason']}")

    # ── Test 3: Explain and recommend ────────────────────────────
    # Uses a real data_builder payload so the explanation is
    # based on actual live data, not mock data
    print("\n" + "="*60)
    print("TEST 3: Explain and Recommend (all 3 timeframes)")
    print("="*60)

    async def test_explain():
        from services.data_builder import build_payload

        test_cases = [
            ("SBIN.NS", "State Bank of India", 50000.0,  "INR", "SHORT_TERM"),
            ("SBIN.NS", "State Bank of India", 50000.0,  "INR", "LONG_TERM"),
            ("AAPL",    "Apple Inc",           5000.0,   "USD", "SHORT_TERM"),
        ]

        for symbol, company, budget, currency, timeframe in test_cases:
            print(f"\n  Fetching payload for {symbol} [{timeframe}]...")
            try:
                payload = await build_payload(symbol, company)
            except Exception as e:
                print(f"  data_builder failed: {e}")
                continue

            print(f"  Passing to ADK explain agent...")
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