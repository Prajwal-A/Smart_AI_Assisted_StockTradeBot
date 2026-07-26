"""
risk_service.py
---------------
Deterministic Risk Scoring Engine.
Consumes a technical snapshot and outputs a structured risk score with breakdown.

Same inputs → same score. Always.
"""

from typing import TypedDict


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RiskFactor(TypedDict):
    factor: str
    penalty: int
    detail: str


class RiskSnapshot(TypedDict):
    symbol: str
    risk_score: int
    risk_category: str  # LOW | MODERATE | HIGH | VERY_HIGH
    breakdown: list[RiskFactor]
    summary: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_BANDS = [
    (0,  25,  "LOW"),
    (26, 50,  "MODERATE"),
    (51, 70,  "HIGH"),
    (71, 100, "VERY_HIGH"),
]

MAX_SCORE = 100


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _score_rsi(rsi: float | None) -> RiskFactor | None:
    """Penalise overbought and oversold RSI."""
    if rsi is None:
        return None

    if rsi > 80:
        return RiskFactor(
            factor="RSI Extremely Overbought",
            penalty=35,
            detail=f"RSI at {rsi:.1f} — well above the 80 danger threshold",
        )
    if rsi > 70:
        return RiskFactor(
            factor="RSI Overbought",
            penalty=20,
            detail=f"RSI at {rsi:.1f} — above the 70 overbought threshold",
        )
    if rsi < 30:
        return RiskFactor(
            factor="RSI Oversold",
            penalty=15,
            detail=f"RSI at {rsi:.1f} — below 30, potential downside continuation risk",
        )
    # Healthy 40–60 zone: no penalty
    return None


def _score_trend(trend: str | None) -> RiskFactor | None:
    """Penalise weak or negative trend conditions."""
    if trend is None:
        return None

    mapping = {
        "STRONG_DOWNTREND": (25, "Strong downtrend detected — persistent selling pressure"),
        "DOWNTREND":        (15, "Downtrend detected — price making lower highs/lows"),
        "SIDEWAYS":         (10, "Sideways trend — no clear directional momentum"),
        "UPTREND":          (0,  ""),
        "STRONG_UPTREND":   (-5, "Strong uptrend — slight risk reduction from momentum"),
    }

    penalty, detail = mapping.get(trend, (0, ""))

    if penalty == 0 and trend == "UPTREND":
        return None  # No penalty, no need to surface

    if penalty <= 0 and trend == "STRONG_UPTREND":
        # We still want to show this as a positive factor
        return RiskFactor(
            factor="Trend (Positive)",
            penalty=penalty,  # negative value reduces score
            detail=detail,
        )

    if penalty == 0:
        return None

    return RiskFactor(
        factor="Trend",
        penalty=penalty,
        detail=f"{trend.replace('_', ' ').title()} — {detail}",
    )


def _score_moving_averages(
    price: float | None,
    sma_50: float | None,
    sma_200: float | None,
) -> RiskFactor | None:
    """Penalise price trading below key moving averages (capped, not additive)."""
    if price is None:
        return None

    below_50  = sma_50  is not None and price < sma_50
    below_200 = sma_200 is not None and price < sma_200

    if below_50 and below_200:
        detail_parts = []
        if sma_50:
            detail_parts.append(f"SMA50 ₹{sma_50:,.2f}")
        if sma_200:
            detail_parts.append(f"SMA200 ₹{sma_200:,.2f}")
        return RiskFactor(
            factor="Below Both Moving Averages",
            penalty=25,
            detail=f"Price ₹{price:,.2f} below {' and '.join(detail_parts)}",
        )

    if below_200:
        return RiskFactor(
            factor="Below SMA 200",
            penalty=15,
            detail=f"Price ₹{price:,.2f} below long-term SMA200 ₹{sma_200:,.2f}",
        )

    if below_50:
        return RiskFactor(
            factor="Below SMA 50",
            penalty=10,
            detail=f"Price ₹{price:,.2f} below medium-term SMA50 ₹{sma_50:,.2f}",
        )

    return None


def _score_volatility(volatility: float | None) -> RiskFactor | None:
    """Penalise high 20-day price volatility."""
    if volatility is None:
        return None

    # volatility is expected as a decimal fraction, e.g. 0.038 = 3.8%
    pct = volatility * 100

    if pct > 5:
        return RiskFactor(
            factor="Extreme Volatility",
            penalty=25,
            detail=f"20-day volatility at {pct:.1f}% — very high price swings",
        )
    if pct > 3:
        return RiskFactor(
            factor="High Volatility",
            penalty=15,
            detail=f"20-day volatility at {pct:.1f}% — elevated price swings",
        )

    # Below 3% — no penalty
    return None


# ---------------------------------------------------------------------------
# Risk category lookup
# ---------------------------------------------------------------------------

def _get_risk_category(score: int) -> str:
    for low, high, label in RISK_BANDS:
        if low <= score <= high:
            return label
    return "VERY_HIGH"


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------

def _build_summary(factors: list[RiskFactor], category: str) -> str:
    """Build a one-line human-readable summary from top risk factors."""
    if not factors:
        return "No significant risk factors detected. Conditions appear stable."

    # Pick top 2 positive-penalty factors (ignore negative/bonus ones)
    top = [f for f in factors if f["penalty"] > 0][:2]

    if not top:
        return "Slight tailwind from strong trend. Overall risk appears low."

    descriptions = " and ".join(f["factor"].lower() for f in top)
    category_str = category.replace("_", " ").title()
    return f"{category_str} risk — elevated by {descriptions}."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_risk(technical_snapshot: dict) -> RiskSnapshot:
    """
    Calculate a deterministic risk score from a technical snapshot.

    Parameters
    ----------
    technical_snapshot : dict
        Output from technical_service.py. Expected keys:
            symbol       : str
            current_price: float
            rsi          : float | None
            sma_50       : float | None
            sma_200      : float | None
            volatility   : float | None   (decimal, e.g. 0.038 for 3.8%)
            trend        : str  (STRONG_UPTREND | UPTREND | SIDEWAYS | DOWNTREND | STRONG_DOWNTREND)

    Returns
    -------
    RiskSnapshot
    """
    symbol = technical_snapshot.get("symbol", "UNKNOWN")
    price  = technical_snapshot.get("current_price")
    rsi    = technical_snapshot.get("rsi")
    sma_50 = technical_snapshot.get("sma_50")
    sma_200= technical_snapshot.get("sma_200")
    vol    = technical_snapshot.get("volatility")
    trend  = technical_snapshot.get("trend")

    # Run each scoring function
    raw_factors: list[RiskFactor | None] = [
        _score_rsi(rsi),
        _score_trend(trend),
        _score_moving_averages(price, sma_50, sma_200),
        _score_volatility(vol),
    ]

    # Filter out None (no penalty)
    factors: list[RiskFactor] = [f for f in raw_factors if f is not None]

    # Sum penalties (allow negative factors to reduce score slightly)
    raw_score = sum(f["penalty"] for f in factors)

    # Clamp between 0 and MAX_SCORE
    risk_score = max(0, min(raw_score, MAX_SCORE))

    category = _get_risk_category(risk_score)
    summary  = _build_summary(factors, category)

    return RiskSnapshot(
        symbol=symbol,
        risk_score=risk_score,
        risk_category=category,
        breakdown=factors,
        summary=summary,
    )