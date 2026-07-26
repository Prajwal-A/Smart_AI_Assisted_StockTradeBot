import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Optional


# ==============================
# Data Classes
# ==============================

@dataclass
class TechnicalSnapshot:
    symbol: str
    current_price: float
    trend: str
    momentum: str
    signal: str
    signal_strength: int          # 0–100 composite confidence score
    rsi: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    bb_position: Optional[float]  # 0.0 = at lower band, 1.0 = at upper band
    volatility: Optional[float]
    volume_confirmed: bool
    risk_level: str
    notes: list[str]


# ==============================
# Historical Data (Non-blocking)
# ==============================

async def get_historical_data(symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data. Uses 2y by default to ensure enough history
    for SMA-200 (~200 trading days) with buffer.
    """
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            None,
            lambda: yf.Ticker(symbol).history(period=period)
        )
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None


# ==============================
# Indicators
# ==============================

def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    return data["Close"].rolling(window=window).mean()


def calculate_ema(data: pd.DataFrame, window: int) -> pd.Series:
    return data["Close"].ewm(span=window, adjust=False).mean()


def calculate_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's RSI using EWM smoothing."""
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (macd_line, signal_line, histogram).
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(macd, signal_period)
    Histogram = MACD - Signal
    """
    ema_fast = data["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_band, lower_band)."""
    sma = data["Close"].rolling(window=window).mean()
    std = data["Close"].rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calculate_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualised historical volatility."""
    returns = data["Close"].pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — measures price range volatility."""
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def is_volume_confirmed(data: pd.DataFrame, window: int = 20) -> bool:
    """
    True if today's volume is above the rolling average —
    confirms that price moves have participation behind them.
    """
    avg_volume = data["Volume"].rolling(window=window).mean()
    latest_vol = data["Volume"].iloc[-1]
    avg = avg_volume.iloc[-1]
    if np.isnan(avg) or avg == 0:
        return False
    return bool(latest_vol > avg)


# ==============================
# Trend Detection
# ==============================

def detect_trend(current_price: float, sma_50: float, sma_200: float) -> str:
    """
    Uses the Golden/Death Cross framework combined with price location
    relative to both moving averages.
    """
    if np.isnan(sma_50) or np.isnan(sma_200):
        return "INSUFFICIENT_DATA"

    golden_cross = sma_50 > sma_200      # Bullish structural alignment
    death_cross = sma_50 < sma_200       # Bearish structural alignment
    price_above_50 = current_price > sma_50
    price_above_200 = current_price > sma_200

    if golden_cross:
        if price_above_50:
            return "STRONG_UPTREND"           # Price leading both MAs — full bull
        else:
            return "PULLBACK_IN_UPTREND"      # Healthy pullback, structure still bullish

    elif death_cross:
        if not price_above_50:
            return "STRONG_DOWNTREND"         # Price lagging both MAs — full bear
        else:
            return "BOUNCE_IN_DOWNTREND"      # Temporary bounce, structure still bearish

    else:
        # SMAs nearly equal — transitional / sideways
        if price_above_50 and price_above_200:
            return "SIDEWAYS_BULLISH_BIAS"
        elif not price_above_50 and not price_above_200:
            return "SIDEWAYS_BEARISH_BIAS"
        return "SIDEWAYS"


# ==============================
# RSI Momentum Classification
# ==============================

def classify_momentum(rsi: float) -> str:
    if np.isnan(rsi):
        return "UNKNOWN"
    if rsi < 20:
        return "EXTREME_OVERSOLD"
    elif rsi < 30:
        return "OVERSOLD"
    elif rsi < 40:
        return "WEAK_BEARISH"
    elif rsi <= 60:
        return "NEUTRAL"
    elif rsi <= 70:
        return "BULLISH"
    elif rsi <= 80:
        return "OVERBOUGHT"
    else:
        return "EXTREME_OVERBOUGHT"


# ==============================
# MACD Crossover Check
# ==============================

def macd_crossover(macd_line: pd.Series, signal_line: pd.Series) -> str:
    """
    Detects recent MACD crossover by comparing the last two bars.
    Returns: BULLISH_CROSS, BEARISH_CROSS, or NONE
    """
    if len(macd_line) < 2 or len(signal_line) < 2:
        return "NONE"

    prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
    curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]

    if np.isnan(prev_diff) or np.isnan(curr_diff):
        return "NONE"

    if prev_diff < 0 and curr_diff >= 0:
        return "BULLISH_CROSS"
    elif prev_diff > 0 and curr_diff <= 0:
        return "BEARISH_CROSS"
    return "NONE"


# ==============================
# Composite Signal Engine
# ==============================

def generate_signal(
    trend: str,
    momentum: str,
    rsi: float,
    macd_cross: str,
    macd_hist: float,
    bb_position: float,
    volume_confirmed: bool,
    risk_level: str
) -> tuple[str, int, list[str]]:
    """
    Multi-factor signal generation.
    Returns (signal, strength_score, notes[]).

    Signal priority:
      1. Risk override (extreme volatility)
      2. Strong confluence signals (trend + momentum + MACD + volume)
      3. Moderate signals (2-factor agreement)
      4. Reversal signals (RSI extremes)
      5. HOLD
    """
    score = 0
    notes = []

    if risk_level == "EXTREME":
        return "AVOID_TRADE", 0, ["Extreme volatility — no trade recommended"]

    # ── Trend score ──────────────────────────────────────────────
    if trend == "STRONG_UPTREND":
        score += 30
    elif trend == "PULLBACK_IN_UPTREND":
        score += 15
        notes.append("Pullback in uptrend — potential entry zone")
    elif trend == "STRONG_DOWNTREND":
        score -= 30
    elif trend == "BOUNCE_IN_DOWNTREND":
        score -= 15
        notes.append("Bounce in downtrend — potential short zone")
    elif trend in ("SIDEWAYS_BULLISH_BIAS",):
        score += 5
    elif trend in ("SIDEWAYS_BEARISH_BIAS",):
        score -= 5

    # ── RSI score ────────────────────────────────────────────────
    if not np.isnan(rsi):
        if rsi < 30:
            score += 20
            notes.append(f"RSI oversold ({rsi:.1f}) — watch for reversal")
        elif rsi < 45:
            score += 10
        elif rsi > 70:
            score -= 20
            notes.append(f"RSI overbought ({rsi:.1f}) — watch for reversal")
        elif rsi > 55:
            score -= 10

    # ── MACD score ───────────────────────────────────────────────
    if macd_cross == "BULLISH_CROSS":
        score += 20
        notes.append("MACD bullish crossover detected")
    elif macd_cross == "BEARISH_CROSS":
        score -= 20
        notes.append("MACD bearish crossover detected")
    elif not np.isnan(macd_hist):
        if macd_hist > 0:
            score += 10
        else:
            score -= 10

    # ── Bollinger Band position score ────────────────────────────
    if not np.isnan(bb_position):
        if bb_position < 0.1:
            score += 15
            notes.append("Price near lower Bollinger Band — oversold zone")
        elif bb_position > 0.9:
            score -= 15
            notes.append("Price near upper Bollinger Band — overbought zone")

    # ── Volume confirmation bonus ─────────────────────────────────
    if volume_confirmed:
        score = int(score * 1.15)   # 15% confidence boost
        notes.append("Volume confirms price move")
    else:
        notes.append("Volume not confirming — lower confidence")

    # ── Risk adjustment ───────────────────────────────────────────
    if risk_level == "HIGH":
        score = int(score * 0.8)
        notes.append("High volatility — reduced position sizing advised")
    elif risk_level == "LOW":
        notes.append("Low volatility — breakout watch")

    # ── Signal mapping from score ─────────────────────────────────
    strength = min(abs(score), 100)   # Normalise to 0–100

    if score >= 55:
        signal = "STRONG_BUY"
    elif score >= 25:
        signal = "BUY"
    elif score <= -55:
        signal = "STRONG_SELL"
    elif score <= -25:
        signal = "SELL"
    elif score >= 15 and rsi < 35:
        signal = "POTENTIAL_REVERSAL_BUY"
    elif score <= -15 and rsi > 65:
        signal = "POTENTIAL_REVERSAL_SELL"
    else:
        signal = "HOLD"

    return signal, strength, notes


# ==============================
# Risk Level Classification
# ==============================

def classify_risk(volatility: float, atr: float, current_price: float) -> str:
    """
    Primary: annualised volatility.
    Secondary: ATR as % of price (normalised range risk).
    """
    atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

    if volatility > 0.65 or atr_pct > 4.0:
        return "EXTREME"
    elif volatility > 0.45 or atr_pct > 2.5:
        return "HIGH"
    elif volatility > 0.20 or atr_pct > 1.0:
        return "NORMAL"
    else:
        return "LOW"


# ==============================
# Main Snapshot Engine
# ==============================

async def get_technical_snapshot(symbol: str) -> Optional[TechnicalSnapshot]:
    """
    Computes a full technical snapshot for a given ticker symbol.

    Indicators used:
      - SMA 50/200 (trend structure)
      - RSI 14 (momentum)
      - MACD 12/26/9 (momentum + crossover)
      - Bollinger Bands 20/2 (mean reversion context)
      - ATR 14 + Historical Volatility 20 (risk)
      - Volume vs 20-day average (confirmation)
    """
    data = await get_historical_data(symbol, period="2y")

    if data is None or len(data) < 200:
        return None

    # ── Compute indicators ────────────────────────────────────────
    data["SMA_50"] = calculate_sma(data, 50)
    data["SMA_200"] = calculate_sma(data, 200)
    data["RSI"] = calculate_rsi(data)
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = calculate_macd(data)
    data["BB_Upper"], data["BB_Mid"], data["BB_Lower"] = calculate_bollinger_bands(data)
    data["Volatility"] = calculate_volatility(data)
    data["ATR"] = calculate_atr(data)

    # ── Use last fully-populated row ─────────────────────────────
    required = ["SMA_50", "SMA_200", "RSI", "MACD", "MACD_Signal",
                "BB_Upper", "BB_Lower", "Volatility", "ATR"]
    clean = data.dropna(subset=required)

    if clean.empty:
        return None

    latest = clean.iloc[-1]

    current_price   = float(latest["Close"])
    sma_50          = float(latest["SMA_50"])
    sma_200         = float(latest["SMA_200"])
    rsi             = float(latest["RSI"])
    macd_val        = float(latest["MACD"])
    macd_signal_val = float(latest["MACD_Signal"])
    macd_hist_val   = float(latest["MACD_Hist"])
    bb_upper        = float(latest["BB_Upper"])
    bb_lower        = float(latest["BB_Lower"])
    volatility      = float(latest["Volatility"])
    atr             = float(latest["ATR"])

    # Bollinger Band position: 0.0 = at lower, 1.0 = at upper
    bb_range = bb_upper - bb_lower
    bb_position = ((current_price - bb_lower) / bb_range) if bb_range != 0 else 0.5

    # ── Derived analysis ──────────────────────────────────────────
    trend           = detect_trend(current_price, sma_50, sma_200)
    momentum        = classify_momentum(rsi)
    macd_cross      = macd_crossover(clean["MACD"], clean["MACD_Signal"])
    volume_ok       = is_volume_confirmed(data)
    risk_level      = classify_risk(volatility, atr, current_price)

    signal, strength, notes = generate_signal(
        trend=trend,
        momentum=momentum,
        rsi=rsi,
        macd_cross=macd_cross,
        macd_hist=macd_hist_val,
        bb_position=bb_position,
        volume_confirmed=volume_ok,
        risk_level=risk_level
    )

    return TechnicalSnapshot(
        symbol=symbol,
        current_price=round(current_price, 2),
        trend=trend,
        momentum=momentum,
        signal=signal,
        signal_strength=strength,
        rsi=round(rsi, 2),
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2),
        macd=round(macd_val, 4),
        macd_signal=round(macd_signal_val, 4),
        macd_histogram=round(macd_hist_val, 4),
        bb_upper=round(bb_upper, 2),
        bb_lower=round(bb_lower, 2),
        bb_position=round(bb_position, 4),
        volatility=round(volatility, 4),
        volume_confirmed=volume_ok,
        risk_level=risk_level,
        notes=notes
    )


# ==============================
# Multi-ticker Batch Runner
# ==============================

async def scan_symbols(symbols: list[str]) -> list[TechnicalSnapshot]:
    """Run snapshots concurrently across multiple tickers."""
    tasks = [get_technical_snapshot(s) for s in symbols]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":
    async def main():
        symbols = ["AAPL", "MSFT", "TSLA", "TCS.NS", "INFY.NS", "RELIANCE.NS", "COFORGE.NS"]
        snapshots = await scan_symbols(symbols)

        for snap in snapshots:
            print(f"\n{'='*50}")
            print(f"  {snap.symbol} — ${snap.current_price}")
            print(f"  Trend     : {snap.trend}")
            print(f"  Momentum  : {snap.momentum}  (RSI: {snap.rsi})")
            print(f"  MACD Hist : {snap.macd_histogram}")
            print(f"  BB Pos    : {snap.bb_position:.2%}")
            print(f"  Signal    : {snap.signal}  (Strength: {snap.signal_strength}/100)")
            print(f"  Risk      : {snap.risk_level}")
            print(f"  Volume OK : {snap.volume_confirmed}")
            print(f"  Notes     :")
            for note in snap.notes:
                print(f"    • {note}")

    asyncio.run(main())