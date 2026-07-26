import yfinance as yf
import requests


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()

    # If already contains exchange suffix, keep it
    if "." in symbol:
        return symbol

    # Try NSE first
    nse_symbol = f"{symbol}.NS"

    try:
        ticker = yf.Ticker(nse_symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return nse_symbol
    except Exception:
        pass

    # Otherwise assume US stock
    return symbol

async def get_current_price(symbol: str) -> float | None:
    normalized = normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(normalized)

        # Use 1-minute data for near real-time
        data = ticker.history(period="1d", interval="1m")

        if data.empty:
            return None

        return float(data["Close"].iloc[-1])

    except Exception as e:
        print("Price Fetch Error:", e)
        return None

async def validate_symbol(symbol: str):
    normalized = normalize_symbol(symbol)

    try:
        ticker = yf.Ticker(normalized)
        data = ticker.history(period="1d")

        if not data.empty:
            return {
                "valid": True,
                "symbol": normalized
            }

    except Exception:
        pass

     # If direct fails → search Yahoo
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
        response = requests.get(url)
        result = response.json()

        quotes = result.get("quotes", [])

        for quote in quotes:
            # if quote.get("quoteType") == "EQUITY":
                suggested_symbol = quote.get("symbol")
                # Check if the suggested symbol is valid
                if suggested_symbol:
                    # Fetch price of suggestion
                    ticker = yf.Ticker(suggested_symbol)
                    data = ticker.history(period="1d", interval="1m")

                if not data.empty:
                    current_price = float(data["Close"].iloc[-1])
                else:
                    current_price = None

                return {
                    "valid": False,
                    "suggestion": suggested_symbol,
                    "suggested_price": current_price
                }

    except Exception as e:
        print("Search error:", e)

    return {
        "valid": False,
        "suggestion": None
    }