"""
payload_builder.py
==================
Combines all 4 layers into one unified payload for ADK.

Layers:
  1. technical_analysis.py  — price action, trend, momentum
  2. news_sentiment.py      — news sentiment + crisis detection
  3. social_sentiment.py    — retail mood via RSS
  4. fundamentals.py        — company financial health

What this file does:
  1. Runs all 4 layers concurrently
  2. Selects a weight profile based on market context
  3. Detects conflicts between layers
  4. Applies fundamentals health modifier to confidence
  5. Computes one final signal + confidence score
  6. Returns one clean payload dict ready for ADK

All 5 files must be in the same folder:
  technical_analysis.py
  news_sentiment.py
  social_sentiment.py
  fundamentals.py
  payload_builder.py

Run:
  python3 payload_builder.py --symbol AAPL --company Apple
  python3 payload_builder.py --symbol TSLA --company Tesla --json
"""

import asyncio
import argparse
import json
from datetime import datetime, timezone

from technical_services import get_technical_snapshot
from news_service     import get_news_sentiment
from social_sentiment   import get_social_sentiment
from fundamental       import get_fundamentals


# ==============================
# Weight profiles
# ==============================

WEIGHT_PROFILES = {
    "LONG_TERM": {
        "technical":    0.15,
        "news":         0.10,
        "social":       0.05,
        "fundamentals": 0.70,
        "reason": "Long-term investment — fundamentals dominate"
    },
    "CRISIS": {
        "technical": 0.20,
        "news":      0.55,
        "social":    0.15,
        "fundamentals": 0.10,
        "reason": "Active macro crisis — news sentiment dominates"
    },
    "EARNINGS_WINDOW": {
        "technical": 0.20,
        "news":      0.45,
        "social":    0.15,
        "fundamentals": 0.20,
        "reason": "Earnings imminent — fundamentals and news dominate"
    },
    "HIGH_SOCIAL_BUZZ": {
        "technical": 0.30,
        "news":      0.20,
        "social":    0.35,
        "fundamentals": 0.15,
        "reason": "High retail buzz — social sentiment weighted higher"
    },
    "LOW_NEWS_NORMAL": {
        "technical": 0.55,
        "news":      0.15,
        "social":    0.10,
        "fundamentals": 0.20,
        "reason": "Quiet news environment — technical and fundamentals lead"
    },
    "DEFAULT": {
        "technical": 0.45,
        "news":      0.25,
        "social":    0.15,
        "fundamentals": 0.15,
        "reason": "Balanced weighting across all four layers"
    },
}


# ==============================
# Signal → numeric score
# ==============================

SIGNAL_SCORES = {
    # Technical
    "STRONG_BUY":              1.0,
    "BUY":                     0.6,
    "POTENTIAL_REVERSAL_BUY":  0.4,
    "HOLD":                    0.0,
    "POTENTIAL_REVERSAL_SELL": -0.4,
    "SELL":                   -0.6,
    "STRONG_SELL":             -1.0,
    "AVOID_TRADE":             -0.8,
    # News
    "BULLISH":                  0.6,
    "NEUTRAL":                  0.0,
    "BEARISH":                 -0.6,
    "CRISIS":                  -0.8,
    "EARNINGS_WINDOW":          0.0,
    # Social
    "STRONGLY_POSITIVE":        0.8,
    "POSITIVE":                 0.5,
    "STRONGLY_NEGATIVE":       -0.8,
    "NEGATIVE":                -0.5,
    # Fundamentals
    "STRONG":                   0.6,
    "HEALTHY":                  0.3,
    "WEAK":                    -0.3,
    "DISTRESSED":              -0.6,
    "UNKNOWN":                  0.0,
}


# ==============================
# Step 1: Select weight profile
# ==============================

def select_weight_profile(
    news_signal:       str,
    earnings_window:   bool,
    social_post_count: int,
    market_mood:       str,
) -> dict:
    """
    Picks the most appropriate weight profile.

    Priority:
      1. Crisis active          → CRISIS
      2. Earnings within 5 days → EARNINGS_WINDOW
      3. High social buzz       → HIGH_SOCIAL_BUZZ
      4. Quiet news period      → LOW_NEWS_NORMAL
      5. Everything else        → DEFAULT
    """
    if news_signal == "CRISIS":
        return {"profile_name": "CRISIS",          **WEIGHT_PROFILES["CRISIS"]}
    if earnings_window:
        return {"profile_name": "EARNINGS_WINDOW", **WEIGHT_PROFILES["EARNINGS_WINDOW"]}
    if social_post_count >= 15 and market_mood in ("GREED", "FEAR"):
        return {"profile_name": "HIGH_SOCIAL_BUZZ", **WEIGHT_PROFILES["HIGH_SOCIAL_BUZZ"]}
    if news_signal == "NEUTRAL" and social_post_count < 5:
        return {"profile_name": "LOW_NEWS_NORMAL",  **WEIGHT_PROFILES["LOW_NEWS_NORMAL"]}
    return     {"profile_name": "DEFAULT",           **WEIGHT_PROFILES["DEFAULT"]}


# ==============================
# Step 2: Detect conflicts
# ==============================

def detect_conflict(
    tech_signal:    str,
    news_signal:    str,
    social_label:   str,
    fund_health:    str,
) -> dict:
    """
    Detects disagreements between layers.

    Conflicts reduce final confidence — a strong recommendation
    requires signals to roughly agree, not contradict each other.

    Returns conflict_detected, conflict_type, conflict_severity.
    """
    tech_score   = SIGNAL_SCORES.get(tech_signal,  0.0)
    news_score   = SIGNAL_SCORES.get(news_signal,  0.0)
    social_score = SIGNAL_SCORES.get(social_label, 0.0)
    fund_score   = SIGNAL_SCORES.get(fund_health,  0.0)

    conflicts = []

    if tech_score > 0.3 and news_score < -0.3:
        conflicts.append("Technical bullish but news bearish")
    elif tech_score < -0.3 and news_score > 0.3:
        conflicts.append("Technical bearish but news bullish")

    if tech_score > 0.3 and social_score < -0.3:
        conflicts.append("Technical bullish but social bearish")
    elif tech_score < -0.3 and social_score > 0.3:
        conflicts.append("Technical bearish but social bullish")

    if tech_score > 0.3 and fund_score < -0.3:
        conflicts.append("Technical bullish but fundamentals weak")
    elif tech_score < -0.3 and fund_score > 0.3:
        conflicts.append("Technical bearish but fundamentals strong")

    if news_score > 0.3 and fund_score < -0.3:
        conflicts.append("News positive but fundamentals weak")

    if not conflicts:
        return {
            "conflict_detected": False,
            "conflict_type":     "NONE",
            "conflict_severity": "NONE",
        }

    severity = "HIGH" if len(conflicts) >= 2 else "MEDIUM"

    return {
        "conflict_detected": True,
        "conflict_type":     " | ".join(conflicts),
        "conflict_severity": severity,
    }


# ==============================
# Step 3: Compute final signal
# ==============================

def compute_final_signal(
    tech_signal:    str,
    tech_strength:  int,
    news_signal:    str,
    social_label:   str,
    fund_health:    str,
    fund_score:     int,
    weight_profile: dict,
    conflict:       dict,
) -> dict:
    """
    Combines all 4 layer signals into one final signal.

    Formula:
      raw_score = (tech  * tech_weight)
                + (news  * news_weight)
                + (social* social_weight)
                + (fund  * fund_weight)

    Confidence:
      Base = tech_strength / 100
      Fundamentals modifier:
        STRONG     +10  → business health confirms the signal
        HEALTHY    +5
        WEAK       -10  → business is struggling, reduce conviction
        DISTRESSED -20  → serious red flag regardless of chart
      Conflict penalty:
        HIGH conflict   → 50% confidence reduction
        MEDIUM conflict → 25% confidence reduction
        No conflict     → 20% confidence boost (all layers agree)
    """
    tech_score   = SIGNAL_SCORES.get(tech_signal,  0.0)
    news_score   = SIGNAL_SCORES.get(news_signal,  0.0)
    social_score = SIGNAL_SCORES.get(social_label, 0.0)
    fund_score_s = SIGNAL_SCORES.get(fund_health,  0.0)

    tw = weight_profile["technical"]
    nw = weight_profile["news"]
    sw = weight_profile["social"]
    fw = weight_profile["fundamentals"]

    raw_score = (
        (tech_score   * tw) +
        (news_score   * nw) +
        (social_score * sw) +
        (fund_score_s * fw)
    )

    # Base confidence from technical strength
    base_confidence = tech_strength / 100.0

    # Fundamentals health modifier
    health_mod = {
        "STRONG":     +0.10,
        "HEALTHY":    +0.05,
        "WEAK":       -0.10,
        "DISTRESSED": -0.20,
        "UNKNOWN":     0.00,
    }.get(fund_health, 0.0)

    confidence = base_confidence + health_mod

    # Conflict modifier
    if not conflict["conflict_detected"]:
        confidence = confidence * 1.20   # boost — all layers agree
    elif conflict["conflict_severity"] == "HIGH":
        confidence = confidence * 0.50   # heavy penalty
    else:
        confidence = confidence * 0.75   # medium penalty

    confidence = round(max(0.0, min(confidence, 1.0)), 2)

    # Map raw score to signal
    if raw_score >= 0.55:
        final_signal = "STRONG_BUY"
    elif raw_score >= 0.25:
        final_signal = "BUY"
    elif raw_score <= -0.55:
        final_signal = "STRONG_SELL"
    elif raw_score <= -0.25:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    # Override: high conflict → never give STRONG signal
    if conflict["conflict_severity"] == "HIGH":
        if final_signal == "STRONG_BUY":   final_signal = "BUY"
        if final_signal == "STRONG_SELL":  final_signal = "SELL"

    # Override: distressed fundamentals on a BUY → downgrade
    if fund_health == "DISTRESSED" and final_signal in ("BUY", "STRONG_BUY"):
        final_signal = "HOLD"

    # Override: avoid trade if technical says so
    if tech_signal == "AVOID_TRADE":
        final_signal = "AVOID_TRADE"
        confidence   = 0.1

    return {
        "final_signal":     final_signal,
        "final_confidence": round(confidence * 100),   # 0–100
        "raw_score":        round(raw_score, 3),
    }


# ==============================
# Step 4: Collect all notes
# ==============================

def build_notes(
    tech_snapshot,
    news_result:    dict,
    social_result:  dict,
    fund_snapshot,
    conflict:       dict,
    weight_profile: dict,
) -> list[str]:
    """
    Collects all notable observations from every layer.
    These become the bullet points in ADK's plain English explanation.
    Ordered: fundamentals summary first, then technical, news, social, conflict.
    """
    notes = []

    # Fundamentals summary goes first — it's the business context
    if fund_snapshot:
        summary = getattr(fund_snapshot, "summary", "")
        if summary:
            notes.append(f"Company health: {summary}")
        notes.extend(getattr(fund_snapshot, "notes", []))

    # Technical notes
    if tech_snapshot:
        notes.extend(getattr(tech_snapshot, "notes", []))
        risk = getattr(tech_snapshot, "risk_level", "")
        if risk in ("HIGH", "EXTREME"):
            notes.append(f"Volatility is {risk} — consider smaller position size")

    # News / crisis notes
    crisis = news_result.get("crisis", {})
    if crisis.get("crisis_detected"):
        notes.append(
            f"⚠ Macro crisis detected: {crisis['crisis_type']} "
            f"({crisis['hit_count']} headlines)"
        )

    # Social / market mood notes
    mood = social_result.get("market_mood", {})
    if mood.get("mood_label") in ("GREED", "FEAR"):
        notes.append(
            f"Retail market mood is {mood['mood_label']} "
            f"(score {mood.get('mood_score', 0):+.3f})"
        )

    # Conflict notes
    if conflict["conflict_detected"]:
        notes.append(
            f"Signal conflict [{conflict['conflict_severity']}]: "
            f"{conflict['conflict_type']}"
        )
        notes.append("Conflicting signals detected — wait for alignment before acting")

    # Weight profile note
    notes.append(
        f"Weight profile used: {weight_profile['profile_name']} — "
        f"{weight_profile['reason']}"
    )

    return notes


# ==============================
# Main builder
# ==============================

async def build_payload(symbol: str, company_name: str, timeframe: str = "SHORT_TERM") -> dict:
    """
    Runs all 4 layers concurrently and combines into one ADK-ready payload.

    Args:
      symbol       : Ticker e.g. "AAPL"
      company_name : e.g. "Apple"

    Returns one dict with everything ADK needs.
    """
    print(f"\n[Payload] Starting full analysis for {symbol}...")

    loop = asyncio.get_running_loop()

    # All 4 layers run at the same time
    # Technical is async natively
    # News, social, fundamentals are sync — run in executor so they don't block
    tech_task   = get_technical_snapshot(symbol)
    news_task   = loop.run_in_executor(None, get_news_sentiment,  symbol)
    social_task = loop.run_in_executor(None, get_social_sentiment, symbol)
    fund_task   = loop.run_in_executor(None, get_fundamentals,    symbol)

    tech_snapshot, news_result, social_result, fund_snapshot = await asyncio.gather(
        tech_task, news_task, social_task, fund_task,
        return_exceptions=True
    )

    # Safe unwrap — a failed layer returns defaults, not a crash
    if isinstance(tech_snapshot, Exception) or tech_snapshot is None:
        print(f"[Payload] Technical layer failed: {tech_snapshot}")
        tech_snapshot = None
    if isinstance(news_result,   Exception):
        print(f"[Payload] News layer failed: {news_result}")
        news_result   = {}
    if isinstance(social_result, Exception):
        print(f"[Payload] Social layer failed: {social_result}")
        social_result = {}
    if isinstance(fund_snapshot, Exception):
        print(f"[Payload] Fundamentals layer failed: {fund_snapshot}")
        fund_snapshot = None

    # ── Extract values from each layer ───────────────────────────

    # Technical
    tech_signal   = getattr(tech_snapshot, "signal",          "HOLD")     if tech_snapshot else "HOLD"
    tech_strength = getattr(tech_snapshot, "signal_strength",  50)        if tech_snapshot else 50
    tech_trend    = getattr(tech_snapshot, "trend",            "UNKNOWN")  if tech_snapshot else "UNKNOWN"
    tech_momentum = getattr(tech_snapshot, "momentum",         "UNKNOWN")  if tech_snapshot else "UNKNOWN"
    tech_rsi      = getattr(tech_snapshot, "rsi",              None)       if tech_snapshot else None
    tech_risk     = getattr(tech_snapshot, "risk_level",       "UNKNOWN")  if tech_snapshot else "UNKNOWN"
    current_price = getattr(tech_snapshot, "current_price",    None)       if tech_snapshot else None

    # News
    news_sentiment    = news_result.get("sentiment", {})
    news_signal       = news_sentiment.get("label",            "NEUTRAL")
    news_score        = news_sentiment.get("sentiment_score",  0.0)
    news_headline_cnt = news_result.get("headline_count",      0)
    crisis_data       = news_result.get("crisis",              {})
    earnings_window   = news_result.get("earnings_window",     False)

    # Social
    social_sentiment  = social_result.get("sentiment", {})
    social_label      = social_sentiment.get("label",           "NEUTRAL")
    social_score      = social_sentiment.get("sentiment_score", 0.0)
    social_post_count = social_result.get("post_count",         0)
    market_mood       = social_result.get("market_mood",        {}).get("mood_label", "UNKNOWN")

    # Fundamentals
    fund_health   = getattr(fund_snapshot, "health_label",  "UNKNOWN") if fund_snapshot else "UNKNOWN"
    fund_score    = getattr(fund_snapshot, "health_score",   50)       if fund_snapshot else 50

    # ── Decision logic ────────────────────────────────────────────

    news_signal_for_profile = "CRISIS" if crisis_data.get("crisis_detected") else news_signal

    weight_profile = select_weight_profile(
        news_signal       = news_signal_for_profile,
        earnings_window   = earnings_window,
        social_post_count = social_post_count,
        market_mood       = market_mood,
    )

     # Long term overrides weight profile
    if timeframe == "LONG_TERM":
        weight_profile = WEIGHT_PROFILES["LONG_TERM"]

    conflict = detect_conflict(tech_signal, news_signal, social_label, fund_health)

    final = compute_final_signal(
        tech_signal    = tech_signal,
        tech_strength  = tech_strength,
        news_signal    = news_signal,
        social_label   = social_label,
        fund_health    = fund_health,
        fund_score     = fund_score,
        weight_profile = weight_profile,
        conflict       = conflict,
    )

    notes = build_notes(
        tech_snapshot, news_result, social_result,
        fund_snapshot, conflict, weight_profile
    )

    # ── Assemble payload ──────────────────────────────────────────

    payload = {
        "symbol":        symbol,
        "company_name":  company_name,
        "analysed_at":   datetime.now(timezone.utc).isoformat(),
        "current_price": current_price,

        # Layer 1
        "technical": {
            "signal":      tech_signal,
            "strength":    tech_strength,
            "trend":       tech_trend,
            "momentum":    tech_momentum,
            "rsi":         tech_rsi,
            "risk_level":  tech_risk,
            "sma_50":      getattr(tech_snapshot, "sma_50",         None) if tech_snapshot else None,
            "sma_200":     getattr(tech_snapshot, "sma_200",        None) if tech_snapshot else None,
            "macd":        getattr(tech_snapshot, "macd_histogram", None) if tech_snapshot else None,
            "bb_position": getattr(tech_snapshot, "bb_position",    None) if tech_snapshot else None,
            "volatility":  getattr(tech_snapshot, "volatility",     None) if tech_snapshot else None,
        },

        # Layer 2
        "news": {
            "signal":          news_signal,
            "sentiment_score": news_score,
            "headline_count":  news_headline_cnt,
            "crisis": {
                "detected":  crisis_data.get("crisis_detected", False),
                "type":      crisis_data.get("crisis_type"),
                "hit_count": crisis_data.get("hit_count", 0),
                "examples":  crisis_data.get("examples", []),
            },
            "earnings_window": earnings_window,
            "top_headlines":   news_result.get("top_headlines", []),
        },

        # Layer 3
        "social": {
            "signal":           social_label,
            "sentiment_score":  social_score,
            "post_count":       social_post_count,
            "market_mood":      social_result.get("market_mood",        {}),
            "source_breakdown": social_result.get("source_breakdown",   {}),
            "top_posts":        social_result.get("top_posts",          []),
        },

        # Layer 4
        "fundamentals": {
            "health_label":           fund_health,
            "health_score":           fund_score,
            "annual_revenue_growth":  getattr(fund_snapshot, "annual_revenue_growth",    None)      if fund_snapshot else None,
            "quarterly_revenue_growth": getattr(fund_snapshot, "quarterly_revenue_growth", None)    if fund_snapshot else None,
            "revenue_period":         getattr(fund_snapshot, "revenue_period_label",     "Unknown") if fund_snapshot else "Unknown",
            "quarterly_period":       getattr(fund_snapshot, "quarterly_period_label",   "Unknown") if fund_snapshot else "Unknown",
            "revenue_growth_label":   getattr(fund_snapshot, "revenue_growth_label",     "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "revenue_trend":          getattr(fund_snapshot, "revenue_trend",            "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "profit_margin":          getattr(fund_snapshot, "profit_margin",            None)      if fund_snapshot else None,
            "profit_margin_label":    getattr(fund_snapshot, "profit_margin_label",      "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "debt_to_equity":         getattr(fund_snapshot, "debt_to_equity",           None)      if fund_snapshot else None,
            "debt_label":             getattr(fund_snapshot, "debt_label",               "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "pe_ratio":               getattr(fund_snapshot, "pe_ratio",                 None)      if fund_snapshot else None,
            "pe_label":               getattr(fund_snapshot, "pe_label",                 "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "earnings_record":        getattr(fund_snapshot, "earnings_record",          [])        if fund_snapshot else [],
            "avg_surprise_pct":       getattr(fund_snapshot, "avg_surprise_pct",         None)      if fund_snapshot else None,
            "earnings_label":         getattr(fund_snapshot, "earnings_label",           "UNKNOWN") if fund_snapshot else "UNKNOWN",
            "summary":                getattr(fund_snapshot, "summary",                  "")        if fund_snapshot else "",
        },

        # Decision
        "weight_profile": {
            "name":    weight_profile["profile_name"],
            "weights": {
                "technical":    weight_profile["technical"],
                "news":         weight_profile["news"],
                "social":       weight_profile["social"],
                "fundamentals": weight_profile["fundamentals"],
            },
            "reason":  weight_profile["reason"],
        },
        "conflict": conflict,

        # Final output
        "final_signal":     final["final_signal"],
        "final_confidence": final["final_confidence"],
        "raw_score":        final["raw_score"],
        "notes":            notes,
    }

    print(f"[Payload] ✓ {symbol} — {final['final_signal']} "
          f"(confidence {final['final_confidence']}/100 | "
          f"fundamentals: {fund_health})")

    return payload


# ==============================
# Pretty print
# ==============================

def print_payload(p: dict):
    print(f"\n{'='*60}")
    print(f"  FULL ANALYSIS: {p['symbol']}  —  {p.get('company_name','')}")
    print(f"  Price        : ${p.get('current_price', 'N/A')}")
    print(f"  Analysed at  : {p.get('analysed_at','')}")
    print(f"{'='*60}")

    t = p["technical"]
    print(f"\n  LAYER 1 — TECHNICAL")
    print(f"    Signal    : {t['signal']}  (strength {t['strength']}/100)")
    print(f"    Trend     : {t['trend']}")
    print(f"    Momentum  : {t['momentum']}  (RSI {t['rsi']})")
    print(f"    Risk      : {t['risk_level']}")

    n = p["news"]
    print(f"\n  LAYER 2 — NEWS")
    print(f"    Signal    : {n['signal']}  (score {n['sentiment_score']:+.3f})")
    print(f"    Headlines : {n['headline_count']}")
    c = n["crisis"]
    print(f"    Crisis    : {c['type'] if c['detected'] else 'None'}")
    print(f"    Earnings  : {'⚠ Within 5 days' if n['earnings_window'] else 'Not imminent'}")

    s = p["social"]
    print(f"\n  LAYER 3 — SOCIAL")
    print(f"    Signal    : {s['signal']}  (score {s['sentiment_score']:+.3f})")
    print(f"    Posts     : {s['post_count']}")
    print(f"    Mkt Mood  : {s['market_mood'].get('mood_label','UNKNOWN')}")

    f = p["fundamentals"]
    print(f"\n  LAYER 4 — FUNDAMENTALS")
    print(f"    Health    : {f['health_label']}  ({f['health_score']}/100)")
    rev = f['annual_revenue_growth']
    print(f"    Revenue   : {rev*100:+.1f}%  [{f['revenue_growth_label']}]  ({f['revenue_period']})"
          if rev is not None else f"    Revenue   : N/A")
    qrev = f['quarterly_revenue_growth']
    print(f"    Qtr trend : {qrev*100:+.1f}%  [{f['revenue_trend']}]  ({f['quarterly_period']})"
          if qrev is not None else f"    Qtr trend : N/A")
    pm = f['profit_margin']
    print(f"    Margin    : {pm*100:.1f}%  [{f['profit_margin_label']}]"
          if pm is not None else f"    Margin    : N/A")
    dte = f['debt_to_equity']
    print(f"    D/E       : {dte:.2f}  [{f['debt_label']}]"
          if dte is not None else f"    D/E       : N/A")
    pe = f['pe_ratio']
    print(f"    P/E       : {pe:.1f}  [{f['pe_label']}]"
          if pe is not None else f"    P/E       : N/A")
    er = f['earnings_record']
    print(f"    Earnings  : {' / '.join(er)}  [{f['earnings_label']}]"
          if er else f"    Earnings  : N/A")

    wp = p["weight_profile"]
    print(f"\n  WEIGHT PROFILE: {wp['name']}")
    w = wp["weights"]
    print(f"    Tech {w['technical']:.0%} / News {w['news']:.0%} / "
          f"Social {w['social']:.0%} / Fundamentals {w['fundamentals']:.0%}")
    print(f"    {wp['reason']}")

    co = p["conflict"]
    print(f"\n  CONFLICT : {co['conflict_type']}  [{co['conflict_severity']}]")

    print(f"\n  {'='*45}")
    print(f"  FINAL SIGNAL   : {p['final_signal']}")
    print(f"  CONFIDENCE     : {p['final_confidence']}/100")
    print(f"  RAW SCORE      : {p['raw_score']:+.3f}")
    print(f"  {'='*45}")

    print(f"\n  NOTES:")
    for note in p["notes"]:
        print(f"    •  {note}")


# ==============================
# Run directly
# ==============================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="AAPL",  help="Ticker e.g. AAPL")
    parser.add_argument("--company", default="Apple", help="Company name e.g. Apple")
    parser.add_argument("--json",    action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    payload = await build_payload(args.symbol, args.company)

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print_payload(payload)


if __name__ == "__main__":
    asyncio.run(main())