"""
compare_builder.py
==================
Runs full analysis for asked stock + ADK-provided peers
concurrently and returns a ranked comparison.

Single responsibility:
  - Receives: asked stock + list of peers (from adk_agent.py)
  - Runs: data_builder.build_payload() for all stocks at once
  - Returns: ComparisonResult with StockCards ranked by signal

This file does NOT discover peers — that is adk_agent.py's job.
This file does NOT parse user queries — that is adk_agent.py's job.
This file does NOT explain results — that is adk_agent.py's job.

Usage:
  from compare_builder import build_comparison

  result = await build_comparison(
      asked_symbol  = "SBIN.NS",
      asked_company = "State Bank of India",
      peers = [
          {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
          {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
          {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
      ]
  )
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional
from data_builder import build_payload


# ==============================
# Output dataclasses
# ==============================

@dataclass
class StockCard:
    """
    Summary card for one stock shown in the comparison view.
    User reads this to decide which stock to analyse further.
    full_payload is stored silently — passed to ADK when user
    clicks Simplify with AI.
    """
    symbol:          str
    company:         str
    current_price:   Optional[float]
    currency:        str
    final_signal:    str            # BUY / SELL / HOLD / STRONG_BUY etc.
    confidence:      int            # 0–100
    risk_level:      str            # LOW / NORMAL / HIGH / EXTREME
    health_label:    str            # STRONG / HEALTHY / WEAK / DISTRESSED
    trend:           str
    rsi:             Optional[float]
    revenue_growth:  Optional[float]
    revenue_trend:   str            # ACCELERATING / STABLE / DECELERATING
    earnings_record: list[str]      # ["BEAT","BEAT","MISS","BEAT"]
    earnings_label:  str            # CONSISTENT_BEATER / MIXED / CONSISTENT_MISSER
    top_note:        str            # single most important observation
    is_asked_stock:  bool           # True for the stock user asked about
    full_payload:    dict           # passed to ADK when selected


@dataclass
class ComparisonResult:
    asked_symbol:  str
    asked_company: str
    cards:         list[StockCard]  # asked stock first, peers ranked after
    best_symbol:   str              # symbol with strongest signal + confidence
    errors:        list[str]        # symbols that failed analysis


# ==============================
# Signal rank for sorting peers
# ==============================

SIGNAL_RANK = {
    "STRONG_BUY":              6,
    "BUY":                     5,
    "POTENTIAL_REVERSAL_BUY":  4,
    "HOLD":                    3,
    "AVOID_TRADE":             2,
    "POTENTIAL_REVERSAL_SELL": 2,
    "SELL":                    1,
    "STRONG_SELL":             0,
}


# ==============================
# Helpers
# ==============================

def _get_currency(symbol: str) -> str:
    """Gets currency for a ticker from yfinance. Falls back to USD."""
    try:
        return yf.Ticker(symbol).info.get("currency", "USD")
    except Exception:
        return "USD"


def _extract_top_note(payload: dict) -> str:
    """
    Picks the single most important note from the payload.
    Priority order: crisis > earnings window > conflict >
                    distressed > accelerating > first note
    """
    notes = payload.get("notes", [])
    if not notes:
        return "No notable observations"

    priority_keywords = [
        "crisis", "earnings window", "conflict",
        "distressed", "accelerating", "warning"
    ]
    for keyword in priority_keywords:
        for note in notes:
            if keyword in note.lower():
                return note

    return notes[0]


def _build_card(
    payload:       dict,
    is_asked:      bool,
    currency:      str,
) -> StockCard:
    """Converts a full payload dict into a StockCard."""
    tech = payload.get("technical",    {})
    fund = payload.get("fundamentals", {})

    return StockCard(
        symbol         = payload.get("symbol",           ""),
        company        = payload.get("company_name",     ""),
        current_price  = payload.get("current_price"),
        currency       = currency,
        final_signal   = payload.get("final_signal",     "HOLD"),
        confidence     = payload.get("final_confidence",  0),
        risk_level     = tech.get("risk_level",          "UNKNOWN"),
        health_label   = fund.get("health_label",        "UNKNOWN"),
        trend          = tech.get("trend",               "UNKNOWN"),
        rsi            = tech.get("rsi"),
        revenue_growth = fund.get("annual_revenue_growth"),
        revenue_trend  = fund.get("revenue_trend",       "UNKNOWN"),
        earnings_record= fund.get("earnings_record",     []),
        earnings_label = fund.get("earnings_label",      "UNKNOWN"),
        top_note       = _extract_top_note(payload),
        is_asked_stock = is_asked,
        full_payload   = payload,
    )


# ==============================
# Ticker validation
# ==============================

def validate_ticker(symbol: str) -> bool:
    """
    Checks if a ticker is valid and tradeable via yfinance.
    Filters out hallucinated or delisted tickers from ADK.
    """
    try:
        info = yf.Ticker(symbol).info
        return bool(
            info.get("regularMarketPrice") or
            info.get("currentPrice") or
            info.get("navPrice")
        )
    except Exception:
        return False


# ==============================
# Main function
# ==============================

async def build_comparison(
    asked_symbol:  str,
    asked_company: str,
    peers:         list[dict],
) -> ComparisonResult:
    """
    Runs full analysis for asked stock + peers concurrently.

    Args:
      asked_symbol  : Ticker the user asked about e.g. "SBIN.NS"
      asked_company : Company name e.g. "State Bank of India"
      peers         : List of {"symbol": ..., "name": ...}
                      Provided by adk_agent.suggest_peers()
                      Already validated before being passed here

    Returns ComparisonResult:
      - asked stock card always first
      - peer cards sorted: best signal first, then confidence
    """
    errors = []

    # Build all tasks — asked stock + peers
    all_stocks = [{"symbol": asked_symbol, "name": asked_company, "is_asked": True}]
    for p in peers:
        all_stocks.append({
            "symbol":   p["symbol"],
            "name":     p.get("name", p["symbol"]),
            "is_asked": False,
        })

    print(f"\n[Compare] Running analysis for "
          f"{[s['symbol'] for s in all_stocks]}...")

    # Run all concurrently
    async def _run_one(stock: dict):
        try:
            payload = await build_payload(stock["symbol"], stock["name"])
            currency = _get_currency(stock["symbol"])
            card = _build_card(payload, stock["is_asked"], currency)
            return card, None
        except Exception as e:
            return None, f"{stock['symbol']}: {str(e)}"

    results = await asyncio.gather(
        *[_run_one(s) for s in all_stocks],
        return_exceptions=True
    )

    # Separate cards from errors
    asked_cards = []
    peer_cards  = []

    for result in results:
        if isinstance(result, Exception):
            errors.append(str(result))
            continue
        card, error = result
        if error:
            errors.append(error)
        elif card:
            if card.is_asked_stock:
                asked_cards.append(card)
            else:
                peer_cards.append(card)

    # Sort peers: best signal first, then highest confidence
    peer_cards.sort(
        key=lambda c: (SIGNAL_RANK.get(c.final_signal, 3), c.confidence),
        reverse=True
    )

    all_cards = asked_cards + peer_cards

    # Find best overall
    best_symbol = asked_symbol
    if all_cards:
        # Best = highest combined score of signal strength × confidence
        # This prevents a low-confidence STRONG BUY beating a high-confidence BUY
        def _score(card):
            sig_rank = SIGNAL_RANK.get(card.final_signal, 3)
            # Weight: signal rank × 100 + confidence
            # So a BUY at 80 confidence (580) beats STRONG BUY at 20 confidence (620)
            # but STRONG BUY at 70 (670) beats BUY at 80 (580)
            return sig_rank * 100 + card.confidence

        best = max(all_cards, key=_score)
        best_symbol = best.symbol

    return ComparisonResult(
        asked_symbol  = asked_symbol,
        asked_company = asked_company,
        cards         = all_cards,
        best_symbol   = best_symbol,
        errors        = errors,
    )


# ==============================
# Serializer — converts result to
# clean dict for API response
# ==============================

def comparison_to_dict(result: ComparisonResult) -> dict:
    """
    Converts ComparisonResult to a JSON-serializable dict
    for the FastAPI response.

    Note: full_payload is excluded from each card in the
    API response — it's large and only needed internally
    for ADK. The frontend gets a clean summary per stock.
    """
    cards = []
    for card in result.cards:
        cards.append({
            "symbol":          card.symbol,
            "company":         card.company,
            "current_price":   card.current_price,
            "currency":        card.currency,
            "final_signal":    card.final_signal,
            "confidence":      card.confidence,
            "risk_level":      card.risk_level,
            "health_label":    card.health_label,
            "trend":           card.trend,
            "rsi":             card.rsi,
            "revenue_growth":  round(card.revenue_growth * 100, 1)
                               if card.revenue_growth is not None else None,
            "revenue_trend":   card.revenue_trend,
            "earnings_record": card.earnings_record,
            "earnings_label":  card.earnings_label,
            "top_note":        card.top_note,
            "is_asked_stock":  card.is_asked_stock,
        })

    return {
        "asked_symbol":  result.asked_symbol,
        "asked_company": result.asked_company,
        "best_symbol":   result.best_symbol,
        "cards":         cards,
        "errors":        result.errors,
    }


# ==============================
# Pretty print for terminal testing
# ==============================

def print_comparison(result: ComparisonResult):
    print(f"\n{'='*65}")
    print(f"  COMPARISON: {result.asked_symbol} vs peers")
    print(f"{'='*65}")

    for i, card in enumerate(result.cards):
        label = "  ★ YOUR STOCK" if card.is_asked_stock else f"  #{i} ALTERNATIVE"
        print(f"\n{label}")
        print(f"  {'─'*55}")

        price_str = (f"{card.currency} {card.current_price:,.2f}"
                     if card.current_price else "N/A")
        print(f"  {card.symbol:15s}  {card.company}")
        print(f"  Price      : {price_str}")
        print(f"  Signal     : {card.final_signal:20s}  "
              f"Confidence: {card.confidence}/100")
        print(f"  Risk       : {card.risk_level:20s}  "
              f"Health: {card.health_label}")
        print(f"  Trend      : {card.trend}")
        print(f"  RSI        : {card.rsi:.1f}"
              if card.rsi else "  RSI        : N/A")

        rev = card.revenue_growth
        print(f"  Revenue    : {rev*100:+.1f}% YoY [{card.revenue_trend}]"
              if rev is not None else "  Revenue    : N/A")

        er = card.earnings_record
        print(f"  Earnings   : {' / '.join(er)}  [{card.earnings_label}]"
              if er else "  Earnings   : N/A")

        print(f"  Key note   : {card.top_note}")

    print(f"\n{'─'*65}")
    print(f"  Best signal  : {result.best_symbol}")

    if result.errors:
        print(f"\n  Errors:")
        for e in result.errors:
            print(f"    ✗  {e}")

    print(f"\n  → Select a stock and click 'Simplify with AI'")
    print(f"    for plain English recommendation + quantity advice.")