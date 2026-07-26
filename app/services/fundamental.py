"""
fundamental.py
==============
Fetches and analyses company financial health using yfinance.
Completely free. No API key needed.

Supports two timeframes:
  SHORT_TERM  — current default, focuses on recent performance
  LONG_TERM   — adds 3-year CAGR, dividend history, extended scoring

What it fetches (both timeframes):
  1. Revenue growth     — annual YoY + quarterly YoY trend
  2. Profit margin      — net profit margin
  3. Debt to equity     — balance sheet strength
  4. P/E ratio          — valuation
  5. Earnings history   — last 4 quarters beat/miss

Additional for LONG_TERM:
  6. 3-year revenue CAGR    — sustained business growth
  7. 5-year price return    — actual investor return (always available)
                              more useful than 5yr revenue CAGR which
                              yfinance rarely has enough data for
  8. Dividend history       — consistency + growth
  9. Return on Equity (ROE) — management efficiency
  10. Free Cash Flow        — real cash generation

Run:
  python3 fundamental.py --symbol AAPL
  python3 fundamental.py --symbol AAPL --timeframe LONG_TERM
  python3 fundamental.py --symbol SBIN.NS --timeframe LONG_TERM
"""

import argparse
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Optional


# ==============================
# Output dataclass
# — added long-term fields at bottom
# — all existing fields unchanged
# ==============================

@dataclass
class FundamentalsSnapshot:
    symbol: str
    timeframe: str                         # SHORT_TERM / LONG_TERM

    # Revenue — short term
    annual_revenue_growth: Optional[float]
    quarterly_revenue_growth: Optional[float]
    revenue_period_label: str
    quarterly_period_label: str
    revenue_growth_label: str
    revenue_trend: str

    # Profitability
    profit_margin: Optional[float]
    profit_margin_label: str

    # Debt
    debt_to_equity: Optional[float]
    debt_label: str

    # Valuation
    pe_ratio: Optional[float]
    pe_label: str

    # Earnings history
    earnings_record: list[str]
    avg_surprise_pct: Optional[float]
    earnings_label: str

    # ── Long-term additions ──────────────────────────────────────
    cagr_3yr: Optional[float]             # 3-year revenue CAGR
    price_return_5yr: Optional[float]     # 5-year stock price CAGR (actual investor return)
    price_return_label: str               # STRONG / MODERATE / FLAT / NEGATIVE
    cagr_label: str                        # STRONG / MODERATE / FLAT / DECLINING
    dividend_yield: Optional[float]        # e.g. 0.014 = 1.4%
    dividend_consistency: str              # CONSISTENT / IRREGULAR / NONE
    dividend_growing: Optional[bool]       # True if dividends growing YoY
    roe: Optional[float]                   # Return on Equity e.g. 0.32 = 32%
    roe_label: str                         # EXCELLENT / GOOD / WEAK / NEGATIVE
    free_cash_flow: Optional[float]        # Most recent annual FCF in raw number
    fcf_label: str                         # STRONG / POSITIVE / NEGATIVE
    # ────────────────────────────────────────────────────────────

    # Overall
    health_label: str
    health_score: int
    summary: str
    notes: list[str] = field(default_factory=list)


# ==============================
# Existing labelling functions
# — unchanged from your original
# ==============================

def _label_revenue_growth(growth: Optional[float]) -> str:
    if growth is None:   return "UNKNOWN"
    if growth >= 0.15:   return "STRONG"
    elif growth >= 0.05: return "MODERATE"
    elif growth >= 0.0:  return "FLAT"
    else:                return "DECLINING"


def _label_revenue_trend(annual: Optional[float], quarterly: Optional[float]) -> str:
    if annual is None or quarterly is None:
        return "UNKNOWN"
    diff = quarterly - annual
    if diff > 0.03:    return "ACCELERATING"
    elif diff < -0.03: return "DECELERATING"
    else:              return "STABLE"


def _label_profit_margin(margin: Optional[float]) -> str:
    if margin is None:   return "UNKNOWN"
    if margin >= 0.20:   return "STRONG"
    elif margin >= 0.10: return "HEALTHY"
    elif margin >= 0.02: return "THIN"
    else:                return "NEGATIVE"


def _label_debt(dte: Optional[float]) -> str:
    if dte is None:   return "UNKNOWN"
    if dte < 0.3:     return "LOW"
    elif dte < 1.0:   return "MODERATE"
    elif dte < 2.0:   return "HIGH"
    else:             return "DANGEROUS"


def _label_pe(pe: Optional[float]) -> str:
    if pe is None or pe <= 0: return "UNKNOWN"
    if pe < 12:               return "CHEAP"
    elif pe < 25:             return "FAIR"
    elif pe < 50:             return "EXPENSIVE"
    else:                     return "EXTREME"


def _label_earnings(record: list[str]) -> str:
    if not record: return "UNKNOWN"
    beats     = record.count("BEAT")
    total     = len(record)
    beat_rate = beats / total
    if beat_rate >= 0.75:   return "CONSISTENT_BEATER"
    elif beat_rate <= 0.25: return "CONSISTENT_MISSER"
    else:                   return "MIXED"


# ==============================
# New long-term labelling
# ==============================

def _label_cagr(cagr: Optional[float]) -> str:
    """
    Labels multi-year revenue CAGR.
    Higher bar than single-year growth because
    sustaining growth over 3+ years is harder.
    """
    if cagr is None:    return "UNKNOWN"
    if cagr >= 0.15:    return "STRONG"      # 15%+ sustained — excellent
    elif cagr >= 0.08:  return "MODERATE"    # 8–15% — solid
    elif cagr >= 0.02:  return "FLAT"        # 2–8% — barely growing
    else:               return "DECLINING"   # negative — shrinking business


def _label_price_return(ret: Optional[float]) -> str:
    """
    Labels 5-year annualised price return (CAGR).
    This is the actual return an investor would have received
    holding the stock for 5 years — more complete than revenue
    CAGR because it captures margin expansion, buybacks,
    and market sentiment changes too.

    Benchmarks:
      S&P 500 historical average ≈ 10% annualised
      So 15%+ = outperforming, 10-15% = market rate, <10% = underperforming
    """
    if ret is None:    return "UNKNOWN"
    if ret >= 0.15:    return "STRONG"       # 15%+ p.a. — strong outperformer
    elif ret >= 0.10:  return "MODERATE"     # 10–15% — market rate
    elif ret >= 0.0:   return "FLAT"         # 0–10% — underperforming
    else:              return "NEGATIVE"     # negative — destroyed value


def _label_roe(roe: Optional[float]) -> str:
    """
    Return on Equity — how efficiently management
    uses shareholder money to generate profit.
    Buffett threshold: consistently above 15% = quality business.
    """
    if roe is None:    return "UNKNOWN"
    if roe >= 0.20:    return "EXCELLENT"    # 20%+ — exceptional
    elif roe >= 0.15:  return "GOOD"         # 15–20% — quality
    elif roe >= 0.08:  return "WEAK"         # 8–15% — mediocre
    else:              return "NEGATIVE"     # negative — destroying value


def _label_fcf(fcf: Optional[float]) -> str:
    """
    Free Cash Flow — real cash the business generates
    after maintaining/growing its asset base.
    More reliable than earnings which can be manipulated.
    """
    if fcf is None:  return "UNKNOWN"
    if fcf > 0:      return "POSITIVE"
    else:            return "NEGATIVE"


# ==============================
# Existing data fetchers
# — unchanged from your original
# ==============================

def _get_revenue_growth(ticker) -> tuple[Optional[float], Optional[float], str, str]:
    annual_growth    = None
    quarterly_growth = None
    annual_label     = "Annual period unknown"
    quarterly_label  = "Quarterly period unknown"

    try:
        annual = ticker.financials
        if annual is not None and not annual.empty:
            for idx in annual.index:
                if "revenue" in str(idx).lower():
                    row = annual.loc[idx]
                    if len(row) >= 2:
                        recent_val  = row.iloc[0]
                        prior_val   = row.iloc[1]
                        recent_date = annual.columns[0]
                        prior_date  = annual.columns[1]
                        annual_label = (
                            f"FY{pd.Timestamp(recent_date).year} "
                            f"vs FY{pd.Timestamp(prior_date).year}"
                        )
                        if (prior_val and prior_val != 0
                                and not pd.isna(prior_val)
                                and not pd.isna(recent_val)):
                            annual_growth = round(
                                (recent_val - prior_val) / abs(prior_val), 4
                            )
                    break
    except Exception:
        pass

    try:
        quarterly = ticker.quarterly_financials
        if quarterly is not None and not quarterly.empty:
            for idx in quarterly.index:
                if "revenue" in str(idx).lower():
                    row = quarterly.loc[idx]
                    if len(row) >= 5:
                        q_recent = row.iloc[0]
                        q_prior  = row.iloc[4]
                        recent_qdate = quarterly.columns[0]
                        prior_qdate  = quarterly.columns[4]
                        def _quarter_str(ts):
                            t = pd.Timestamp(ts)
                            q = (t.month - 1) // 3 + 1
                            return f"Q{q}-{t.year}"
                        quarterly_label = (
                            f"{_quarter_str(recent_qdate)} "
                            f"vs {_quarter_str(prior_qdate)}"
                        )
                        if (q_prior and q_prior != 0
                                and not pd.isna(q_prior)
                                and not pd.isna(q_recent)):
                            quarterly_growth = round(
                                (q_recent - q_prior) / abs(q_prior), 4
                            )
                    break
    except Exception:
        pass

    return annual_growth, quarterly_growth, annual_label, quarterly_label


def _get_profit_margin(info: dict) -> Optional[float]:
    margin = info.get("profitMargins")
    if margin is None or (isinstance(margin, float) and margin != margin):
        return None
    return round(float(margin), 4)


def _get_debt_to_equity(info: dict) -> Optional[float]:
    dte = info.get("debtToEquity")
    if dte is None or (isinstance(dte, float) and dte != dte):
        return None
    return round(float(dte) / 100, 4)


def _get_pe_ratio(info: dict) -> Optional[float]:
    pe = info.get("trailingPE")
    if pe is None or (isinstance(pe, float) and pe != pe):
        return None
    return round(float(pe), 2)


def _get_earnings_history(ticker) -> tuple[list[str], Optional[float]]:
    try:
        history = ticker.earnings_history
        if history is None or history.empty:
            return [], None
        record    = []
        surprises = []
        for _, row in history.head(4).iterrows():
            estimate = row.get("epsEstimate")
            actual   = row.get("epsActual")
            surprise = row.get("surprisePercent")
            if pd.isna(estimate) or pd.isna(actual):
                continue
            record.append("BEAT" if actual >= estimate else "MISS")
            if not pd.isna(surprise):
                surprises.append(float(surprise) * 100)
        avg_surprise = round(sum(surprises) / len(surprises), 2) if surprises else None
        return record, avg_surprise
    except Exception:
        return [], None


# ==============================
# New long-term data fetchers
# ==============================

def _get_revenue_cagr(ticker) -> tuple[Optional[float], str]:
    """
    Calculates 3-year revenue CAGR from annual financials.

    CAGR formula: (recent / oldest) ^ (1/years) - 1

    Uses iloc[0] vs iloc[3] — 4 years of data is the
    maximum yfinance reliably provides, giving us a clean
    3-year comparison.

    Returns (cagr_3yr, period_label)
    e.g. (0.124, "FY2025 vs FY2022")
    """
    cagr_3yr     = None
    period_label = "3yr period unknown"

    try:
        annual = ticker.financials
        if annual is None or annual.empty:
            return None, period_label

        for idx in annual.index:
            if "revenue" in str(idx).lower():
                row = annual.loc[idx].dropna()

                if len(row) >= 4:
                    recent      = float(row.iloc[0])
                    oldest      = float(row.iloc[3])
                    recent_date = annual.columns[0]
                    oldest_date = annual.columns[3]

                    period_label = (
                        f"FY{pd.Timestamp(recent_date).year} "
                        f"vs FY{pd.Timestamp(oldest_date).year}"
                    )

                    if oldest > 0:
                        cagr_3yr = round((recent / oldest) ** (1/3) - 1, 4)
                break

    except Exception:
        pass

    return cagr_3yr, period_label


def _get_price_return_5yr(ticker) -> Optional[float]:
    """
    Calculates 5-year annualised stock price return (CAGR).

    Unlike revenue CAGR which needs 5 years of financial statements
    (rarely available), price history goes back as far as needed.

    Formula: (price_now / price_5yr_ago) ^ (1/5) - 1

    Why this is better than 5yr revenue CAGR:
      - Always available for any listed stock
      - Captures the full picture — revenue growth, margin expansion,
        buybacks, dividend reinvestment, sentiment changes
      - It's the actual return an investor would have received
      - Directly comparable across different sectors and countries

    Returns annualised return as decimal e.g. 0.18 = 18% p.a.
    Returns None if less than 4.5 years of history available.
    """
    try:
        hist = ticker.history(period="5y")

        if hist is None or hist.empty:
            return None

        # Need at least 4.5 years of data for a meaningful 5yr calculation
        years_available = len(hist) / 252   # 252 trading days per year
        if years_available < 4.5:
            return None

        price_start = float(hist["Close"].iloc[0])
        price_end   = float(hist["Close"].iloc[-1])

        if price_start <= 0:
            return None

        # Annualised CAGR over actual period
        actual_years = years_available
        cagr = (price_end / price_start) ** (1 / actual_years) - 1
        return round(cagr, 4)

    except Exception:
        return None


def _get_dividend_data(ticker, info: dict) -> tuple[Optional[float], str, Optional[bool]]:
    """
    Fetches dividend yield, consistency, and growth direction.

    dividend_yield        : e.g. 0.014 = 1.4%
    dividend_consistency  : CONSISTENT / IRREGULAR / NONE
    dividend_growing      : True if last dividend > dividend 2 years ago

    Why dividends matter for long-term:
      Consistent dividend payers are usually mature, profitable businesses.
      Growing dividends signal management confidence in future earnings.
      Many long-term investors specifically target dividend compounders.
    """
    dividend_yield       = None
    dividend_consistency = "NONE"
    dividend_growing     = None

    try:
        # Yield from info
        dy = info.get("dividendYield")
        if dy and not (isinstance(dy, float) and dy != dy):
            dividend_yield = round(float(dy), 4)

        # History from ticker
        divs = ticker.dividends
        if divs is None or divs.empty:
            return dividend_yield, "NONE", None

        # Check consistency: paid in each of last 3 years?
        divs.index = pd.to_datetime(divs.index, utc=True)
        now        = pd.Timestamp.now(tz="UTC")

        years_paid = set()
        for date in divs.index:
            years_paid.add(date.year)

        current_year   = now.year
        last_3_years   = {current_year - 1, current_year - 2, current_year - 3}
        years_covered  = last_3_years.intersection(years_paid)

        if len(years_covered) >= 3:
            dividend_consistency = "CONSISTENT"
        elif len(years_covered) >= 1:
            dividend_consistency = "IRREGULAR"
        else:
            dividend_consistency = "NONE"

        # Check if dividends are growing
        recent_divs = divs.sort_index()
        if len(recent_divs) >= 8:
            recent_avg    = recent_divs.iloc[-4:].mean()   # last 4 payments
            prior_avg     = recent_divs.iloc[-8:-4].mean() # prior 4 payments
            if prior_avg > 0:
                dividend_growing = bool(recent_avg > prior_avg)

    except Exception:
        pass

    return dividend_yield, dividend_consistency, dividend_growing


def _get_roe(info: dict) -> Optional[float]:
    """
    Return on Equity from yfinance info.
    returnOnEquity is already a decimal in yfinance.
    """
    roe = info.get("returnOnEquity")
    if roe is None or (isinstance(roe, float) and roe != roe):
        return None
    return round(float(roe), 4)


def _get_free_cash_flow(ticker) -> Optional[float]:
    """
    Most recent annual Free Cash Flow from cashflow statement.

    FCF = Operating Cash Flow - Capital Expenditure
    yfinance provides this directly in the cashflow statement.
    """
    try:
        cf = ticker.cashflow
        if cf is None or cf.empty:
            return None

        # Try direct FCF row first
        for idx in cf.index:
            if "free cash flow" in str(idx).lower():
                val = cf.loc[idx].iloc[0]
                if not pd.isna(val):
                    return float(val)

        # Calculate from operating CF - capex
        op_cf  = None
        capex  = None

        for idx in cf.index:
            idx_lower = str(idx).lower()
            if "operating" in idx_lower and "cash" in idx_lower:
                val = cf.loc[idx].iloc[0]
                if not pd.isna(val):
                    op_cf = float(val)
            if "capital expenditure" in idx_lower or "capex" in idx_lower:
                val = cf.loc[idx].iloc[0]
                if not pd.isna(val):
                    capex = float(val)

        if op_cf is not None and capex is not None:
            return op_cf - abs(capex)   # capex is usually negative in yfinance

    except Exception:
        pass

    return None


# ==============================
# Health score
# — short term unchanged
# — long term uses different weights
# ==============================

def _compute_health_score(
    revenue_label:  str,
    margin_label:   str,
    debt_label:     str,
    earnings_label: str,
    revenue_trend:  str,
    timeframe:      str = "SHORT_TERM",
    # Long-term extras
    cagr_label:          str = "UNKNOWN",
    price_return_label:  str = "UNKNOWN",
    roe_label:           str = "UNKNOWN",
    fcf_label:           str = "UNKNOWN",
    div_consistency:     str = "NONE",
) -> tuple[int, str]:
    """
    SHORT_TERM scoring (unchanged):
      Revenue growth  25 pts
      Profit margin   25 pts
      Debt level      25 pts
      Earnings hist   25 pts
      Trend ±5 pts

    LONG_TERM scoring (fundamentals-heavy):
      3yr CAGR        20 pts  — sustained revenue growth
      5yr price ret   15 pts  — actual investor return track record
      Profit margin   20 pts  — profitability
      ROE             20 pts  — management quality
      Debt level      15 pts  — survivability
      FCF              5 pts  — real cash generation
      Dividends        5 pts  — shareholder returns
      Trend ±5 pts
    """
    score = 0

    if timeframe == "LONG_TERM":
        score += {"STRONG": 20, "MODERATE": 14, "FLAT": 6,
                  "DECLINING": 0, "UNKNOWN": 8}.get(cagr_label, 8)

        score += {"STRONG": 15, "MODERATE": 10, "FLAT": 4,
                  "NEGATIVE": 0, "UNKNOWN": 6}.get(price_return_label, 6)

        score += {"STRONG": 20, "HEALTHY": 14, "THIN": 6,
                  "NEGATIVE": 0, "UNKNOWN": 10}.get(margin_label, 10)

        score += {"EXCELLENT": 20, "GOOD": 15, "WEAK": 5,
                  "NEGATIVE": 0, "UNKNOWN": 8}.get(roe_label, 8)

        score += {"LOW": 15, "MODERATE": 10, "HIGH": 4,
                  "DANGEROUS": 0, "UNKNOWN": 7}.get(debt_label, 7)

        score += {"POSITIVE": 5, "NEGATIVE": 0, "UNKNOWN": 2}.get(fcf_label, 2)

        score += {"CONSISTENT": 5, "IRREGULAR": 2, "NONE": 0}.get(div_consistency, 0)

        if revenue_trend == "ACCELERATING":   score += 5
        elif revenue_trend == "DECELERATING": score -= 5

    else:
        # Short-term weights — original unchanged
        score += {"STRONG": 25, "MODERATE": 18, "FLAT": 10,
                  "DECLINING": 0, "UNKNOWN": 12}.get(revenue_label, 12)

        score += {"STRONG": 25, "HEALTHY": 18, "THIN": 8,
                  "NEGATIVE": 0, "UNKNOWN": 12}.get(margin_label, 12)

        score += {"LOW": 25, "MODERATE": 18, "HIGH": 8,
                  "DANGEROUS": 0, "UNKNOWN": 12}.get(debt_label, 12)

        score += {"CONSISTENT_BEATER": 25, "MIXED": 15,
                  "CONSISTENT_MISSER": 0, "UNKNOWN": 12}.get(earnings_label, 12)

        if revenue_trend == "ACCELERATING":   score += 5
        elif revenue_trend == "DECELERATING": score -= 5

    score = max(0, min(score, 100))

    if score >= 75:   label = "STRONG"
    elif score >= 50: label = "HEALTHY"
    elif score >= 25: label = "WEAK"
    else:             label = "DISTRESSED"

    return score, label


# ==============================
# Summary builder
# — adds long-term context when timeframe=LONG_TERM
# ==============================

def _build_summary(snap: FundamentalsSnapshot) -> str:
    """One plain English sentence for ADK to use directly."""
    parts = []

    health_intro = {
        "STRONG":     "financially strong",
        "HEALTHY":    "financially healthy",
        "WEAK":       "showing financial weakness",
        "DISTRESSED": "in financial distress",
    }.get(snap.health_label, "of mixed financial health")

    parts.append(health_intro)

    if snap.timeframe == "LONG_TERM":
        # Long-term summary focuses on CAGR, price return, and ROE
        if snap.cagr_3yr is not None:
            parts.append(f"3-year revenue CAGR of {snap.cagr_3yr*100:+.1f}%")
        if snap.price_return_5yr is not None:
            parts.append(f"{snap.price_return_5yr*100:+.1f}% annualised price return over 5 years")
        if snap.roe is not None:
            parts.append(f"{snap.roe*100:.1f}% return on equity")
        if snap.profit_margin is not None:
            parts.append(f"{snap.profit_margin*100:.1f}% profit margin")
        if snap.debt_label in ("LOW", "MODERATE"):
            parts.append("manageable debt")
        elif snap.debt_label in ("HIGH", "DANGEROUS"):
            parts.append(f"concerning debt (D/E: {snap.debt_to_equity:.2f})")
        if snap.dividend_consistency == "CONSISTENT":
            growing_str = ", growing" if snap.dividend_growing else ""
            parts.append(f"consistent dividend payer{growing_str}")
        if snap.fcf_label == "POSITIVE":
            parts.append("positive free cash flow")
    else:
        # Short-term summary — original logic unchanged
        if snap.annual_revenue_growth is not None:
            pct = snap.annual_revenue_growth * 100
            dir = "growing" if pct >= 0 else "declining"
            parts.append(f"revenue {dir} {abs(pct):.1f}% YoY ({snap.revenue_period_label})")
        if snap.revenue_trend == "ACCELERATING":
            parts.append("momentum accelerating in recent quarters")
        elif snap.revenue_trend == "DECELERATING":
            parts.append("but growth momentum is slowing in recent quarters")
        if snap.profit_margin is not None:
            parts.append(f"{snap.profit_margin*100:.1f}% profit margin")
        if snap.debt_label in ("LOW", "MODERATE"):
            parts.append("manageable debt")
        elif snap.debt_label in ("HIGH", "DANGEROUS"):
            parts.append(f"high debt (D/E: {snap.debt_to_equity:.2f})")
        if snap.earnings_label == "CONSISTENT_BEATER":
            beats = snap.earnings_record.count("BEAT")
            total = len(snap.earnings_record)
            avg   = snap.avg_surprise_pct
            ep    = f", avg +{avg:.1f}% surprise" if avg else ""
            parts.append(f"beaten estimates {beats}/{total} recent quarters{ep}")
        elif snap.earnings_label == "CONSISTENT_MISSER":
            parts.append("consistently missed earnings estimates")

    return f"{snap.symbol} is {', '.join(parts)}."


# ==============================
# Main function
# — timeframe parameter added
# — all existing logic unchanged
# ==============================

def get_fundamentals(
    symbol:    str,
    timeframe: str = "SHORT_TERM",
) -> FundamentalsSnapshot:
    """
    Fetches and analyses fundamental data for a stock.

    Args:
      symbol    : Ticker e.g. "AAPL" or "SBIN.NS"
      timeframe : "SHORT_TERM" (default) or "LONG_TERM"

    SHORT_TERM → same as before, no changes
    LONG_TERM  → adds CAGR, ROE, FCF, dividends
                 uses fundamentals-heavy scoring
    """
    print(f"[Fundamentals] Fetching data for {symbol} [{timeframe}]...")

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}
    except Exception as e:
        print(f"[Fundamentals] yfinance init error: {e}")
        ticker = None
        info   = {}

    # ── Fetch existing metrics (unchanged) ───────────────────────
    if ticker:
        annual_growth, quarterly_growth, annual_label, quarterly_label = _get_revenue_growth(ticker)
        earnings_record, avg_surprise = _get_earnings_history(ticker)
    else:
        annual_growth = quarterly_growth = None
        annual_label  = quarterly_label  = "Unavailable"
        earnings_record = []
        avg_surprise    = None

    profit_margin  = _get_profit_margin(info)
    debt_to_equity = _get_debt_to_equity(info)
    pe_ratio       = _get_pe_ratio(info)

    # ── Fetch long-term metrics ───────────────────────────────────
    cagr_3yr         = None
    cagr_3yr_label   = "3yr period unknown"
    price_return_5yr = None
    dividend_yield   = None
    dividend_consistency = "NONE"
    dividend_growing = None
    roe              = None
    fcf              = None

    if timeframe == "LONG_TERM" and ticker:
        print(f"[Fundamentals] Fetching long-term metrics...")
        cagr_3yr, cagr_3yr_label              = _get_revenue_cagr(ticker)
        price_return_5yr                       = _get_price_return_5yr(ticker)
        dividend_yield, dividend_consistency, dividend_growing = _get_dividend_data(ticker, info)
        roe                                    = _get_roe(info)
        fcf                                    = _get_free_cash_flow(ticker)

    # ── Labels ───────────────────────────────────────────────────
    revenue_label  = _label_revenue_growth(annual_growth)
    revenue_trend  = _label_revenue_trend(annual_growth, quarterly_growth)
    margin_label   = _label_profit_margin(profit_margin)
    debt_label     = _label_debt(debt_to_equity)
    pe_label       = _label_pe(pe_ratio)
    earnings_label = _label_earnings(earnings_record)
    cagr_label     = _label_cagr(cagr_3yr)
    roe_label      = _label_roe(roe)
    fcf_label      = _label_fcf(fcf)

    # ── Health score ─────────────────────────────────────────────
    price_return_label = _label_price_return(price_return_5yr)

    health_score, health_label = _compute_health_score(
        revenue_label      = revenue_label,
        margin_label       = margin_label,
        debt_label         = debt_label,
        earnings_label     = earnings_label,
        revenue_trend      = revenue_trend,
        timeframe          = timeframe,
        cagr_label         = cagr_label,
        price_return_label = price_return_label,
        roe_label          = roe_label,
        fcf_label          = fcf_label,
        div_consistency    = dividend_consistency,
    )

    # ── Notes ────────────────────────────────────────────────────
    notes = []

    if timeframe == "LONG_TERM":
        # Long-term specific notes
        if cagr_3yr is not None:
            if cagr_3yr >= 0.15:
                notes.append(f"Strong 3-year revenue CAGR of {cagr_3yr*100:+.1f}% — sustained growth")
            elif cagr_3yr < 0:
                notes.append(f"3-year revenue CAGR is negative ({cagr_3yr*100:+.1f}%) — shrinking business")
        if roe_label == "EXCELLENT":
            notes.append(f"ROE of {roe*100:.1f}% — management is generating strong returns")
        elif roe_label == "NEGATIVE":
            notes.append("Negative ROE — company is destroying shareholder value")
        if dividend_consistency == "CONSISTENT" and dividend_growing:
            notes.append("Consistent and growing dividends — strong shareholder returns")
        elif dividend_consistency == "NONE" and pe_ratio and pe_ratio > 30:
            notes.append("No dividend + high P/E — pure growth bet, higher risk")
        if fcf_label == "NEGATIVE":
            notes.append("Negative free cash flow — burning cash, investigate sustainability")
        if price_return_5yr is not None:
            if price_return_5yr >= 0.15:
                notes.append(
                    f"5-year annualised price return of {price_return_5yr*100:+.1f}% "
                    f"— strong long-term outperformer"
                )
            elif price_return_5yr < 0:
                notes.append(
                    f"5-year annualised price return is negative ({price_return_5yr*100:+.1f}%) "
                    f"— stock has destroyed value over 5 years"
                )
    else:
        # Short-term notes — original unchanged
        if revenue_trend == "DECELERATING":
            notes.append(
                f"Revenue growth decelerating — annual {annual_growth*100:+.1f}% "
                f"but recent quarter only {quarterly_growth*100:+.1f}%"
                if annual_growth and quarterly_growth else
                "Revenue growth decelerating in recent quarters"
            )
        if revenue_label == "DECLINING":
            notes.append("Revenue declining YoY — investigate before buying")
        if margin_label == "NEGATIVE":
            notes.append("Company is unprofitable — elevated risk")
        if debt_label == "DANGEROUS":
            notes.append(f"Dangerous debt levels (D/E: {debt_to_equity:.2f}) — vulnerable in downturns")
        if earnings_label == "CONSISTENT_MISSER":
            notes.append("Consistently misses earnings — management guidance unreliable")
        if pe_label == "EXTREME":
            notes.append(f"P/E of {pe_ratio} is extremely high — growth already priced in")
        if revenue_trend == "ACCELERATING":
            notes.append("Revenue momentum accelerating — business is gaining speed")

    if health_label in ("STRONG", "HEALTHY") and not notes:
        notes.append("Fundamentals are solid and support the signal")

    # ── Assemble snapshot ─────────────────────────────────────────
    snapshot = FundamentalsSnapshot(
        symbol                   = symbol,
        timeframe                = timeframe,
        annual_revenue_growth    = annual_growth,
        quarterly_revenue_growth = quarterly_growth,
        revenue_period_label     = annual_label,
        quarterly_period_label   = quarterly_label,
        revenue_growth_label     = revenue_label,
        revenue_trend            = revenue_trend,
        profit_margin            = profit_margin,
        profit_margin_label      = margin_label,
        debt_to_equity           = debt_to_equity,
        debt_label               = debt_label,
        pe_ratio                 = pe_ratio,
        pe_label                 = pe_label,
        earnings_record          = earnings_record,
        avg_surprise_pct         = avg_surprise,
        earnings_label           = earnings_label,
        cagr_3yr                 = cagr_3yr,
        price_return_5yr         = price_return_5yr,
        price_return_label       = price_return_label,
        cagr_label               = cagr_label,
        dividend_yield           = dividend_yield,
        dividend_consistency     = dividend_consistency,
        dividend_growing         = dividend_growing,
        roe                      = roe,
        roe_label                = roe_label,
        free_cash_flow           = fcf,
        fcf_label                = fcf_label,
        health_label             = health_label,
        health_score             = health_score,
        summary                  = "",
        notes                    = notes,
    )

    snapshot.summary = _build_summary(snapshot)

    print(f"[Fundamentals] {symbol}: {health_label} ({health_score}/100) | "
          f"Revenue {annual_label} | Trend: {revenue_trend}"
          + (f" | CAGR-3yr: {cagr_3yr*100:+.1f}%" if cagr_3yr else ""))

    return snapshot


# ==============================
# Run directly to test
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="AAPL",       help="Ticker symbol")
    parser.add_argument("--timeframe", default="SHORT_TERM",
                        choices=["SHORT_TERM", "LONG_TERM"],  help="Analysis timeframe")
    args = parser.parse_args()

    snap = get_fundamentals(args.symbol, args.timeframe)

    print(f"\n{'='*60}")
    print(f"  FUNDAMENTALS: {snap.symbol}  [{snap.timeframe}]")
    print(f"{'='*60}")

    print(f"\n  REVENUE")
    if snap.annual_revenue_growth is not None:
        print(f"    Annual growth   : {snap.annual_revenue_growth*100:+.1f}%  "
              f"[{snap.revenue_growth_label}]  ({snap.revenue_period_label})")
    else:
        print(f"    Annual growth   : N/A")
    if snap.quarterly_revenue_growth is not None:
        print(f"    Quarterly (YoY) : {snap.quarterly_revenue_growth*100:+.1f}%  "
              f"({snap.quarterly_period_label})")
    print(f"    Trend           : {snap.revenue_trend}")

    if snap.timeframe == "LONG_TERM":
        print(f"\n  LONG-TERM GROWTH")
        print(f"    3-yr CAGR      : "
              + (f"{snap.cagr_3yr*100:+.1f}%  [{snap.cagr_label}]"
                 if snap.cagr_3yr is not None else "N/A"))
        print(f"    5-yr price ret : "
              + (f"{snap.price_return_5yr*100:+.1f}% p.a.  [{snap.price_return_label}]"
                 if snap.price_return_5yr is not None else "N/A"))

        print(f"\n  QUALITY METRICS")
        print(f"    ROE        : "
              + (f"{snap.roe*100:.1f}%  [{snap.roe_label}]"
                 if snap.roe is not None else "N/A"))
        print(f"    Free CF    : "
              + (f"{snap.free_cash_flow:,.0f}  [{snap.fcf_label}]"
                 if snap.free_cash_flow is not None else f"N/A  [{snap.fcf_label}]"))

        print(f"\n  DIVIDENDS")
        print(f"    Yield      : "
              + (f"{snap.dividend_yield*100:.2f}%"
                 if snap.dividend_yield else "None"))
        print(f"    Consistency: {snap.dividend_consistency}")
        print(f"    Growing    : {snap.dividend_growing}")

    print(f"\n  PROFITABILITY")
    print(f"    Profit margin : "
          + (f"{snap.profit_margin*100:.1f}%  [{snap.profit_margin_label}]"
             if snap.profit_margin is not None else "N/A"))

    print(f"\n  DEBT")
    print(f"    Debt/Equity   : "
          + (f"{snap.debt_to_equity:.2f}  [{snap.debt_label}]"
             if snap.debt_to_equity is not None else "N/A"))

    print(f"\n  VALUATION")
    print(f"    P/E ratio     : "
          + (f"{snap.pe_ratio:.1f}  [{snap.pe_label}]"
             if snap.pe_ratio is not None else "N/A"))

    print(f"\n  EARNINGS HISTORY")
    if snap.earnings_record:
        print(f"    Record        : {' / '.join(snap.earnings_record)}")
        print(f"    Avg surprise  : "
              + (f"{snap.avg_surprise_pct:+.1f}%" if snap.avg_surprise_pct else "N/A"))
        print(f"    Label         : {snap.earnings_label}")
    else:
        print(f"    Record        : N/A")

    print(f"\n  OVERALL")
    print(f"    Health score  : {snap.health_score}/100")
    print(f"    Health label  : {snap.health_label}")

    print(f"\n  SUMMARY:")
    print(f"    {snap.summary}")

    if snap.notes:
        print(f"\n  NOTES:")
        for note in snap.notes:
            print(f"    • {note}")