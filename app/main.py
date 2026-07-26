"""
main.py
=======
FastAPI backend — complete, wired up with everything.

Existing endpoints (unchanged):
  POST /auth/token          — login
  POST /auth/register       — register
  GET  /portfolio/me        — holdings + unrealized PnL
  GET  /trades/me           — trade history
  POST /trade/buy           — execute buy
  POST /trade/sell          — execute sell

New endpoints (AI analysis):
  POST /analyse             — intent extraction + peer suggestion + comparison
  POST /analyse/explain     — AI plain English explanation + quantity advice
  POST /analyse/chat        — follow-up conversation

Run:
  uvicorn app.main:app --reload --port 8000

Project structure expected:
  app/
  ├── main.py                     ← this file
  ├── auth/
  │   ├── auth_services.py
  │   ├── dependencies.py
  │   └── password_utils.py
  ├── db/
  │   └── mongo.py
  ├── models/
  │   ├── user.py
  │   ├── portfolio.py
  │   ├── trade.py
  │   └── holding.py
  └── services/
      ├── market_data.py
      ├── technical_services.py
      ├── news_service.py
      ├── social_sentiment.py
      ├── fundamental.py
      ├── data_builder.py
      ├── compare_builder.py
      └── adk_agent.py
"""

import yfinance as yf

from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── Existing imports (unchanged) ─────────────────────────────
from app.auth.auth_services  import create_access_token
from app.auth.dependencies   import get_current_user
from app.auth.password_utils import hash_password, verify_password
from app.db.mongo            import db, init_db
from app.models.user         import User
from app.models.portfolio    import Portfolio
from app.models.trade        import Trade
from app.models.holding      import Holding
from app.services.market_data import validate_symbol, get_current_price

# ── New AI analysis imports ───────────────────────────────────
from app.services.grok_AIagent import (
    extract_intent,
    suggest_peers,
    validate_and_filter_peers,
    explain_and_recommend,
    continue_conversation,
)
from app.services.compare_builder import build_comparison, comparison_to_dict
from app.services.data_builder    import build_payload


# ==============================
# App setup
# ==============================

app = FastAPI(
    title       = "Trading Assistant API",
    description = "AI-powered stock analysis + portfolio management",
    version     = "2.0.0",
)

# Allow NiceGUI frontend (port 3000) to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ==============================
# Startup
# ==============================

@app.on_event("startup")
async def app_init():
    await init_db()


# ==============================
# Health check
# ==============================

@app.get("/")
def root():
    return {"message": "Trading Assistant API Running", "version": "2.0.0"}


@app.get("/db-test")
async def test_db():
    collections = await db.list_collection_names()
    return {"collections": collections}


# ==============================
# Auth endpoints (unchanged)
# ==============================

@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await User.find_one(User.username == form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/register")
async def register(username: str, password: str):
    existing = await User.find_one(User.username == username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(username=username, password=hash_password(password))
    await user.insert()

    portfolio = Portfolio(user_id=str(user.id), balance=100000.0)
    await portfolio.insert()

    return {"message": "User registered successfully with initial portfolio balance of Rs.100,000"}


@app.get("/me")
def get_me(current_user: str = Depends(get_current_user)):
    return {"message": f"You are authenticated as {current_user}"}


# ==============================
# Portfolio endpoints (unchanged)
# ==============================

@app.get("/portfolio/me")
async def get_my_portfolio(current_user: str = Depends(get_current_user)):
    user = await User.find_one(User.username == current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    portfolio = await Portfolio.find_one(Portfolio.user_id == str(user.id))
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holdings = await Holding.find(Holding.user_id == str(user.id)).to_list()

    holdings_data      = []
    total_unrealized   = 0
    total_invested     = 0

    for h in holdings:
        current_price = await get_current_price(h.symbol)
        if current_price is None:
            current_price = h.average_price
        unrealized_pnl = (current_price - h.average_price) * h.quantity
        total_unrealized += unrealized_pnl
        total_invested   += h.average_price * h.quantity
        holdings_data.append({
            "symbol":        h.symbol,
            "quantity":      h.quantity,
            "average_price": h.average_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
        })

    return {
        "balance":               portfolio.balance,
        "total_invested":        total_invested,
        "total_unrealized_pnl":  total_unrealized,
        "total_portfolio_value": portfolio.balance + total_invested + total_unrealized,
        "holdings":              holdings_data,
    }


# ==============================
# Trade endpoints (unchanged)
# ==============================

@app.get("/trades/me")
async def get_my_trades(current_user: str = Depends(get_current_user)):
    user = await User.find_one(User.username == current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    trades = await Trade.find(Trade.user_id == str(user.id)).to_list()
    return {
        "trades": [
            {
                "symbol":     t.symbol,
                "quantity":   t.quantity,
                "price":      t.price,
                "trade_type": t.trade_type,
                "status":     t.status,
                "timestamp":  t.timestamp,
            }
            for t in trades
        ]
    }


@app.post("/trade/buy")
async def buy_trade(
    symbol:       str = Body(...),
    quantity:     int = Body(...),
    current_user: str = Depends(get_current_user),
):
    symbol = symbol.strip().upper()

    is_valid = await validate_symbol(symbol)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid stock symbol.")

    price = await get_current_price(symbol)
    if price is None:
        raise HTTPException(status_code=400, detail="Unable to fetch live market price.")

    user = await User.find_one(User.username == current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    portfolio = await Portfolio.find_one(Portfolio.user_id == str(user.id))
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    total_cost = quantity * price
    if portfolio.balance < total_cost:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    portfolio.balance -= total_cost
    await portfolio.save()

    trade = Trade(user_id=str(user.id), symbol=symbol, quantity=quantity, price=price, trade_type="BUY")
    await trade.insert()

    holding = await Holding.find_one(Holding.user_id == str(user.id), Holding.symbol == symbol)
    if holding:
        total_qty    = holding.quantity + quantity
        new_avg      = (holding.average_price * holding.quantity + total_cost) / total_qty
        holding.quantity      = total_qty
        holding.average_price = new_avg
        await holding.save()
    else:
        await Holding(user_id=str(user.id), symbol=symbol, quantity=quantity, average_price=price).insert()

    return {"message": "Trade executed successfully", "remaining_balance": portfolio.balance}


@app.post("/trade/sell")
async def sell_trade(
    symbol:       str = Body(...),
    quantity:     int = Body(...),
    current_user: str = Depends(get_current_user),
):
    symbol = symbol.strip().upper()

    is_valid = await validate_symbol(symbol)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid stock symbol.")

    price = await get_current_price(symbol)
    if price is None:
        raise HTTPException(status_code=400, detail="Unable to fetch live market price.")

    user = await User.find_one(User.username == current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    portfolio = await Portfolio.find_one(Portfolio.user_id == str(user.id))
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    holding = await Holding.find_one(Holding.user_id == str(user.id), Holding.symbol == symbol)
    if not holding:
        raise HTTPException(status_code=400, detail="No holdings for this symbol")
    if holding.quantity < quantity:
        raise HTTPException(status_code=400, detail="Not enough quantity to sell")

    total_value       = quantity * price
    portfolio.balance += total_value
    await portfolio.save()

    realized_pnl      = (price - holding.average_price) * quantity
    holding.quantity  -= quantity
    if holding.quantity == 0:
        await holding.delete()
    else:
        await holding.save()

    trade = Trade(
        user_id      = str(user.id),
        symbol       = symbol,
        quantity     = quantity,
        price        = price,
        trade_type   = "SELL",
        realized_pnl = realized_pnl,
    )
    await trade.insert()

    return {"message": "Sell executed successfully", "updated_balance": portfolio.balance}


# ==============================
# Request / Response models
# for new AI analysis endpoints
# ==============================

class AnalyseRequest(BaseModel):
    query: str
    # e.g. "Should I buy SBI with ₹50000 long term?"


class ExplainRequest(BaseModel):
    symbol:    str
    company:   str
    budget:    Optional[float] = None
    currency:  str             = "INR"
    timeframe: str             = "SHORT_TERM"


class ChatRequest(BaseModel):
    user_message:         str
    conversation_history: list[dict]


# ==============================
# POST /analyse
# Full analysis flow:
#   1. ADK understands the query
#   2. yfinance fetches sector metadata
#   3. ADK suggests peers
#   4. Peers are quality-validated
#   5. compare_builder runs all 4 concurrently
#   6. Returns comparison cards
# ==============================

@app.post("/analyse/")
async def analyse(
    body:         AnalyseRequest,
    current_user: str = Depends(get_current_user),
):
    """
    Full analysis from natural language query.
    Handles English, Hindi, mixed language.

    Example:
      {"query": "Should I buy SBI with ₹50000 long term?"}

    Returns:
      Comparison of asked stock + 3 peers with signal, confidence,
      risk, health, RSI, revenue, earnings per stock.
      Also returns intent metadata (budget, currency, timeframe)
      for the frontend to pass to /analyse/explain.
    """
    # Step 1 — understand the query
    intent    = extract_intent(body.query)
    symbol    = intent.get("symbol")
    company   = intent.get("company")
    budget    = intent.get("budget")
    currency  = intent.get("currency",  "USD")
    timeframe = intent.get("timeframe", "SHORT_TERM")

    if not symbol:
        raise HTTPException(
            status_code = 400,
            detail      = (
                "Could not identify a stock symbol from your query. "
                "Please mention a stock name or ticker symbol."
            )
        )

    # Step 2 — get sector/country metadata from yfinance
    try:
        info     = yf.Ticker(symbol).info or {}
        sector   = info.get("sector",   "Unknown")
        industry = info.get("industry", "Unknown")
        country  = info.get("country",  "Unknown")
        if not company:
            company = info.get("longName") or info.get("shortName") or symbol
    except Exception:
        sector = industry = country = "Unknown"
        if not company:
            company = symbol

    # Step 3 — ADK suggests peer stocks
    raw_peers = suggest_peers(symbol, company, sector, industry, country)

    # Step 4 — validate and rank peers by quality
    # Filters: must have price, min volume, market cap ratio
    # Ranks by: health score (from fundamentals), then market cap
    peers = validate_and_filter_peers(raw_peers, asked_symbol=symbol)

    # Step 5 — run full analysis for all stocks concurrently
    comparison = await build_comparison(symbol, company, peers)

    # Step 6 — return clean response (full_payload NOT sent to frontend)
    result          = comparison_to_dict(comparison)
    result["intent"] = {
        "symbol":    symbol,
        "company":   company,
        "budget":    budget,
        "currency":  currency,
        "timeframe": timeframe,
    }

    return result


# ==============================
# POST /analyse/explain
# AI plain English explanation
# for a selected stock.
# Called when user clicks
# "Simplify with AI".
# ==============================

@app.post("/analyse/explain")
async def explain(
    body:         ExplainRequest,
    current_user: str = Depends(get_current_user),
):
    """
    AI explanation for a stock the user selected from the comparison.
    Re-runs data_builder to get fresh payload, then calls Groq.

    Returns:
      recommendation   — BUY / HOLD / SELL etc.
      explanation      — 2-3 sentences in plain English
      quantity         — how many shares to buy given budget + risk
      quantity_reasoning — why that number
      key_points       — 3 bullet points
      risk_warning     — honest risk assessment
      disclaimer       — always included
      conversation_history — pass this back to /analyse/chat
    """
    # Re-run data_builder for fresh data
    # (we don't store full_payload in the DB — privacy + size)
    payload = await build_payload(body.symbol, body.company)

    result = explain_and_recommend(
        payload   = payload,
        budget    = body.budget,
        currency  = body.currency,
        timeframe = body.timeframe,
    )

    return {
        "symbol":               body.symbol,
        "company":              body.company,
        "recommendation":       result.get("recommendation"),
        "explanation":          result.get("explanation"),
        "quantity":             result.get("quantity"),
        "quantity_reasoning":   result.get("quantity_reasoning"),
        "key_points":           result.get("key_points", []),
        "risk_warning":         result.get("risk_warning"),
        "timeframe":            result.get("timeframe"),
        "disclaimer":           result.get("disclaimer"),
        "conversation_history": result.get("conversation_history", []),
    }


# ==============================
# POST /analyse/chat
# Conversational follow-up
# after initial AI explanation.
# ==============================

@app.post("/analyse/chat")
async def chat(
    body:         ChatRequest,
    current_user: str = Depends(get_current_user),
):
    """
    Free-form follow-up questions after the initial AI explanation.
    User can ask anything:
      "explain like I'm a beginner"
      "explain like I'm 5 years old"
      "worst case scenario if I invest now?"
      "should I invest all at once or spread it?"
      "what does HIGH risk mean practically?"

    Requires:
      user_message         — the follow-up question
      conversation_history — from /analyse/explain response

    Returns:
      response             — AI answer in plain text
      conversation_history — updated history for next follow-up
    """
    if not body.conversation_history:
        raise HTTPException(
            status_code = 400,
            detail      = "No conversation history. Call /analyse/explain first."
        )

    result = continue_conversation(
        conversation_history = body.conversation_history,
        user_message         = body.user_message,
    )

    return {
        "response":             result.get("response"),
        "conversation_history": result.get("conversation_history", []),
    }