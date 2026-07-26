"""
ui.py — NiceGUI Frontend
========================
Three pages:
  /          → Stock Analysis (main)
  /portfolio → Portfolio & Trade History
  /chat      → AI Chat (follow-up after analysis)

Install:
  pip install nicegui httpx

Run alongside FastAPI:
  python3 ui.py

The UI calls your FastAPI backend at http://localhost:8000
Make sure backend is running first.

Design: Dark terminal-inspired aesthetic
  Deep charcoal background, electric green accents
  Monospace font for data, clean sans-serif for UI
  Feels like a Bloomberg terminal for indie traders
"""

import asyncio
import httpx
from nicegui import ui, app as nice_app

API_BASE = "http://localhost:8000"

# ==============================
# Shared state
# ==============================
# NiceGUI uses Python dicts as reactive state per session
state = {
    "token":                None,
    "username":             None,
    "comparison":           None,   # result from /analyse
    "selected_card":        None,   # stock user picked
    "explain_result":       None,   # result from /analyse/explain
    "conversation_history": [],     # for /analyse/chat
    "intent":               {},     # budget, currency, timeframe
}


# ==============================
# Global styles
# ==============================

def apply_styles():
    ui.add_head_html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }

      body {
        background: #0a0c0f;
        color: #e2e8f0;
        font-family: 'IBM Plex Mono', monospace;
      }

      /* Scrollbar */
      ::-webkit-scrollbar { width: 4px; }
      ::-webkit-scrollbar-track { background: #0a0c0f; }
      ::-webkit-scrollbar-thumb { background: #00ff88; border-radius: 2px; }

      /* Cards */
      .stock-card {
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 16px;
        cursor: pointer;
        transition: all 0.2s;
      }
      .stock-card:hover { border-color: #00ff88; transform: translateY(-2px); }
      .stock-card.selected { border-color: #00ff88; background: #0d1f14; }

      /* Signal badges */
      .badge-buy       { background: #0d2b1a; color: #00ff88; border: 1px solid #00ff88; }
      .badge-sell      { background: #2b0d0d; color: #ff4444; border: 1px solid #ff4444; }
      .badge-hold      { background: #2b250d; color: #ffaa00; border: 1px solid #ffaa00; }
      .badge-avoid     { background: #1a1a2b; color: #8888ff; border: 1px solid #8888ff; }
      .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
      }

      /* Risk pills */
      .risk-extreme { color: #ff4444; }
      .risk-high    { color: #ff8800; }
      .risk-normal  { color: #ffdd00; }
      .risk-low     { color: #00ff88; }

      /* Chat bubbles */
      .chat-user {
        background: #1a2030;
        border-left: 3px solid #4488ff;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 13px;
      }
      .chat-ai {
        background: #0d1f14;
        border-left: 3px solid #00ff88;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.6;
      }

      /* Input styling */
      .nicegui-input input {
        background: #111318 !important;
        border: 1px solid #1e2330 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
      }
      .nicegui-input input:focus { border-color: #00ff88 !important; }

      /* Nav */
      .nav-link {
        color: #6b7280;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: color 0.2s;
      }
      .nav-link:hover, .nav-link.active { color: #00ff88; }

      /* Section headers */  
      .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #00ff88;
        margin-bottom: 12px;
     }

      /* Ticker */
      .ticker-symbol {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #e2e8f0;
      }
      .metric-value {
        font-size: 13px;
        color: #9ca3af;
      }
      .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        color: #6b7280;
        text-transform: uppercase;
      }

      /* Glow effect for selected */
      .glow-green { box-shadow: 0 0 20px rgba(0,255,136,0.15); }

      /* Loading spinner */
      .loading-text {
        color: #00ff88;
        font-size: 12px;
        letter-spacing: 2px;
        animation: pulse 1.5s infinite;
      }
      @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

      /* PnL colors */
      .pnl-positive { color: #00ff88; }
      .pnl-negative { color: #ff4444; }
    </style>
    """)


# ==============================
# API helpers
# ==============================

async def api_post(endpoint: str, data: dict, auth: bool = True) -> dict:
    headers = {}
    if auth and state["token"]:
        headers["Authorization"] = f"Bearer {state['token']}"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{API_BASE}{endpoint}", json=data, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def api_get(endpoint: str) -> dict:
    headers = {}
    if state["token"]:
        headers["Authorization"] = f"Bearer {state['token']}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{API_BASE}{endpoint}", headers=headers)
        resp.raise_for_status()
        return resp.json()


async def api_login(username: str, password: str) -> bool:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_BASE}/auth/token",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code == 200:
            state["token"]    = resp.json()["access_token"]
            state["username"] = username
            return True
        return False


# ==============================
# Helper: signal badge HTML
# ==============================

def signal_badge(signal: str) -> str:
    s = signal.upper()
    if "BUY" in s:
        css = "badge-buy"
    elif "SELL" in s:
        css = "badge-sell"
    elif "AVOID" in s:
        css = "badge-avoid"
    else:
        css = "badge-hold"
    return f'<span class="badge {css}">{signal}</span>'


def risk_color(risk: str) -> str:
    return {
        "EXTREME": "risk-extreme",
        "HIGH":    "risk-high",
        "NORMAL":  "risk-normal",
        "LOW":     "risk-low",
    }.get(risk.upper(), "metric-value")


# ==============================
# Shared nav bar
# ==============================

def navbar(page: str = "/"):
    def _nav(label, href):
        """Render a nav link — highlighted if it matches the current page."""
        is_active = (href == page) or (page == "/chat" and href == "/chat")
        color     = "#00ff88" if is_active else "#6b7280"
        border    = "border-bottom: 2px solid #00ff88; padding-bottom: 2px;" if is_active else ""
        ui.html(
            f'<a href="{href}" style="color:{color};font-size:12px;letter-spacing:2px;'
            f'text-transform:uppercase;text-decoration:none;{border}'
            f'padding:4px 12px;transition:color 0.2s;">{label}</a>'
        )

    with ui.row().classes("w-full items-center justify-between px-6 py-4").style(
        "border-bottom: 1px solid #1e2330; background: #0a0c0f; position: sticky; top:0; z-index:100;"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.html('<a href="/" style="text-decoration:none;">'
                    '<span style="color:#00ff88;font-family:Syne,sans-serif;'
                    'font-size:18px;font-weight:800;">▲ TRADEAI</span></a>')
            if state["username"]:
                ui.html(f'<span style="color:#4b5563;font-size:11px;letter-spacing:2px;">/ {state["username"].upper()}</span>')

        with ui.row().classes("items-center gap-2"):
            _nav("ANALYSIS",  "/")
            _nav("PORTFOLIO", "/portfolio")
            if state["explain_result"]:
                _nav("AI CHAT", "/chat")
            _nav("PROFILE", "/profile")


# ==============================
# Login dialog
# ==============================

def show_login_dialog(on_success):
    with ui.dialog() as dialog, ui.card().style(
        "background:#111318;border:1px solid #1e2330;min-width:320px;"
    ):
        ui.html('<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#00ff88;margin-bottom:20px;">SIGN IN</div>')

        username = ui.input("Username").classes("w-full nicegui-input").style("margin-bottom:12px;")
        password = ui.input("Password", password=True).classes("w-full nicegui-input").style("margin-bottom:16px;")
        status   = ui.label("").style("color:#ff4444;font-size:12px;")

        async def do_login():
            ok = await api_login(username.value, password.value)
            if ok:
                dialog.close()
                on_success()
            else:
                status.text = "Invalid credentials"

        ui.button("LOGIN", on_click=do_login).props("flat").style(
            "background:#00ff88;color:#0a0c0f;font-weight:700;width:100%;"
            "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
        )

        ui.html('<div style="color:#4b5563;font-size:11px;margin-top:12px;text-align:center;">'
                'No account? <a href="/register" style="color:#00ff88;">Register</a></div>')

    dialog.open()


# ==============================
# PAGE 1: Stock Analysis
# ==============================

@ui.page("/")
async def analysis_page():
    apply_styles()

    if not state["token"]:
        show_login_dialog(lambda: ui.navigate.to("/"))
        return

    navbar("/")

    with ui.column().classes("w-full max-w-5xl mx-auto px-6 py-8 gap-6"):

        # ── Search bar ───────────────────────────────────────────
        ui.html('<div class="section-label">MARKET INTELLIGENCE</div>')

        with ui.row().classes("w-full gap-3 items-end"):
            query_input = ui.input(
                placeholder='Ask anything... "Should I buy SBI with ₹50000?" or "Is AAPL good long term?"'
            ).classes("flex-1 nicegui-input")

            analyse_btn = ui.button("ANALYSE ▶").props("flat").style(
                "background:#00ff88;color:#0a0c0f;font-weight:700;"
                "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
                "padding:0 20px;height:40px;"
            )

        status_label = ui.html("").style("font-size:12px;min-height:20px;")
        loading      = ui.html('<div class="loading-text">SCANNING MARKETS...</div>').style("display:none;")

        # ── Comparison cards area ────────────────────────────────
        cards_container = ui.column().classes("w-full gap-4")

        # ── Simplify with AI button ──────────────────────────────
        explain_btn = ui.button("✨  SIMPLIFY WITH AI").props("flat").style(
            "background:#0d2b1a;color:#00ff88;border:1px solid #00ff88;"
            "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
            "width:100%;padding:12px;display:none;"
        )

        # ── Restore previous results if any ─────────────────────
        if state["comparison"]:
            _render_cards(cards_container, explain_btn)

        # ── Analyse handler ──────────────────────────────────────
        async def do_analyse():
            q = query_input.value.strip()
            if not q:
                return

            loading.style("display:block;")
            cards_container.clear()
            explain_btn.style("display:none;")
            status_label.set_content("")

            try:
                result = await api_post("/analyse/", {"query": q})
                state["comparison"]    = result
                state["selected_card"] = None
                state["explain_result"] = None
                state["intent"] = result.get("intent", {})
                _render_cards(cards_container, explain_btn)

            except httpx.HTTPStatusError as e:
                status_label.set_content(
                    f'<span style="color:#ff4444;">Error: {e.response.text}</span>'
                )
            except Exception as e:
                status_label.set_content(
                    f'<span style="color:#ff4444;">Error: {str(e)}</span>'
                )
            finally:
                loading.style("display:none;")

        analyse_btn.on("click", do_analyse)
        query_input.on("keydown.enter", do_analyse)

        # ── Explain handler ──────────────────────────────────────
        async def do_explain():
            card = state.get("selected_card")
            if not card:
                ui.notify("Select a stock first", color="negative")
                return

            intent = state.get("intent", {})

            # Disable button + show connecting message so user knows it's working
            explain_btn.props("disabled")
            explain_btn.style(
                "background:#0d2b1a;color:#00ff88;border:1px solid #00ff88;"
                "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
                "width:100%;padding:12px;opacity:0.7;"
            )
            explain_btn.text = "⏳  CONNECTING TO AI..."
            loading.style("display:block;")

            try:
                result = await api_post("/analyse/explain", {
                    "symbol":    card["symbol"],
                    "company":   card["company"],
                    "budget":    intent.get("budget"),
                    "currency":  intent.get("currency", "INR"),
                    "timeframe": intent.get("timeframe", "SHORT_TERM"),
                })
                state["explain_result"]       = result
                state["conversation_history"] = result.get("conversation_history", [])
                ui.navigate.to("/chat")

            except Exception as e:
                ui.notify(f"Error: {str(e)}", color="negative")
                explain_btn.props(remove="disabled")
                explain_btn.text = "✨  SIMPLIFY WITH AI"
            finally:
                loading.style("display:none;")

        explain_btn.on("click", do_explain)


def _render_cards(container, explain_btn):
    """Renders stock comparison cards inside the container."""
    data = state.get("comparison", {})
    cards = data.get("cards", [])
    best  = data.get("best_symbol", "")

    container.clear()

    with container:
        # Header row
        ui.html(f'<div class="section-label">'
                f'COMPARISON — {len(cards)} STOCKS ANALYSED'
                f'{"  ·  BEST: " + best if best else ""}'
                f'</div>')

        for card in cards:
            _render_single_card(card)

    explain_btn.style("display:block;")


def _render_single_card(card: dict):
    """Renders one stock card. Clicking selects it."""
    symbol  = card["symbol"]
    is_ask  = card.get("is_asked_stock", False)
    signal  = card.get("final_signal", "HOLD")
    conf    = card.get("confidence", 0)
    risk    = card.get("risk_level", "NORMAL")
    health  = card.get("health_label", "UNKNOWN")
    price   = card.get("current_price")
    curr    = card.get("currency", "")
    rsi     = card.get("rsi")
    rev     = card.get("revenue_growth")
    trend   = card.get("revenue_trend", "")
    earn    = card.get("earnings_record", [])
    note    = card.get("top_note", "")

    is_selected = (
        state.get("selected_card") and
        state["selected_card"].get("symbol") == symbol
    )

    extra_cls = "selected glow-green" if is_selected else ""
    header_tag = "★ YOUR STOCK" if is_ask else "ALTERNATIVE"

    with ui.card().classes(f"stock-card w-full {extra_cls}") as card_el:

        # Header
        with ui.row().classes("items-center justify-between mb-3"):
            ui.html(f'<span style="font-size:10px;letter-spacing:3px;color:#4b5563;">{header_tag}</span>')
            ui.html(signal_badge(signal))

        # Symbol + company
        ui.html(f'<div class="ticker-symbol">{symbol}</div>')
        ui.html(f'<div style="color:#6b7280;font-size:11px;margin-bottom:12px;">{card.get("company","")}</div>')

        # Price
        if price:
            ui.html(f'<div style="font-size:18px;font-weight:600;color:#e2e8f0;margin-bottom:12px;">'
                    f'{curr} {price:,.2f}</div>')

        # Metrics grid
        with ui.grid(columns=3).classes("w-full gap-2"):
            # Confidence
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">CONFIDENCE</div>')
                ui.html(f'<div class="metric-value">{conf}/100</div>')

            # Risk
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">RISK</div>')
                ui.html(f'<div class="{risk_color(risk)}">{risk}</div>')

            # Health
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">HEALTH</div>')
                ui.html(f'<div class="metric-value">{health}</div>')

            # RSI
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">RSI</div>')
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                ui.html(f'<div class="metric-value">{rsi_str}</div>')

            # Revenue
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">REVENUE</div>')
                rev_str = f'{rev*100:+.1f}% [{trend[:4]}]' if rev is not None else "N/A"
                ui.html(f'<div class="metric-value">{rev_str}</div>')

            # Earnings
            with ui.column().classes("gap-0"):
                ui.html(f'<div class="metric-label">EARNINGS</div>')
                earn_str = "/".join([e[0] for e in earn[:4]]) if earn else "N/A"
                ui.html(f'<div class="metric-value" style="letter-spacing:1px;">{earn_str}</div>')

        # Key note
        if note:
            ui.html(f'<div style="margin-top:12px;padding:8px;background:#0a0c0f;'
                    f'border-radius:4px;font-size:11px;color:#6b7280;'
                    f'border-left:2px solid #1e2330;line-height:1.6;'
                    f'white-space:normal;word-wrap:break-word;">{note}</div>')

    # Click to select
    def select_card():
        state["selected_card"] = card
        ui.navigate.reload()

    card_el.on("click", select_card)


# ==============================
# PAGE 2: Portfolio
# ==============================

# ==============================
# Trivia data for portfolio page
# ==============================

TRADING_TRIVIA = [
    ("📈 PAPER TRADING",
     "Paper trading means trading with virtual money. No real risk, but all the real experience. "
     "Perfect for learning how markets work before committing real capital."),
    ("💡 DID YOU KNOW?",
     "Warren Buffett bought his first stock at age 11 — 6 shares of Cities Service at $38 each. "
     "He sold at $40 and watched it rise to $200. His first lesson: patience beats quick profits."),
    ("📊 RSI EXPLAINED",
     "RSI (Relative Strength Index) measures momentum on a 0-100 scale. "
     "Below 30 = oversold (potential buy zone). Above 70 = overbought (potential sell zone). "
     "Most traders use 14-day RSI."),
    ("🏦 WHAT IS P&L?",
     "P&L = Profit & Loss. Unrealized P&L is the gain/loss on stocks you still hold. "
     "Realized P&L is what you actually made/lost when you sell. "
     "This app shows both so you always know where you stand."),
    ("⚖️ RISK MANAGEMENT",
     "Professional traders never risk more than 1-2% of their portfolio on a single trade. "
     "If you have ₹1,00,000, that means max ₹1,000-₹2,000 per trade. "
     "This rule alone separates survivors from blown-up accounts."),
    ("📉 WHAT IS A BEAR MARKET?",
     "A bear market is when stocks fall 20%+ from recent highs. "
     "Bear markets average 9.5 months in duration historically. "
     "The best long-term investors see bear markets as sale events, not disasters."),
    ("🔍 FUNDAMENTAL VS TECHNICAL",
     "Fundamental analysis asks: Is this company healthy? (Revenue, margins, debt). "
     "Technical analysis asks: What is price momentum doing right now? "
     "The best traders use both — fundamentals to pick the stock, technicals to time entry."),
    ("💰 COMPOUNDING POWER",
     "₹10,000 invested at 15% annual return becomes ₹1,63,665 in 20 years — without adding a single rupee. "
     "This is why starting early matters more than starting big."),
    ("🎯 STOP LOSS",
     "A stop loss is a pre-set price where you automatically sell to limit losses. "
     "If you buy at ₹100, a 10% stop loss means you sell at ₹90 no matter what. "
     "It removes emotion from the most painful decision in trading."),
    ("📅 EARNINGS SEASON",
     "Every quarter, listed companies report their financial results. "
     "Stocks often move 5-15% on earnings day — up if results beat expectations, down if they miss. "
     "Our AI flags when earnings are within 5 days so you can plan accordingly."),
]

import random as _random

@ui.page("/portfolio")
async def portfolio_page():
    apply_styles()

    if not state["token"]:
        show_login_dialog(lambda: ui.navigate.to("/portfolio"))
        return

    navbar("/portfolio")

    with ui.column().classes("w-full max-w-5xl mx-auto px-6 py-8 gap-6"):

        loading = ui.html('<div class="loading-text">LOADING PORTFOLIO...</div>')

        portfolio_container = ui.column().classes("w-full gap-6")
        trivia_container    = ui.column().classes("w-full gap-4")
        trade_container     = ui.column().classes("w-full gap-4")
        trades_container    = ui.column().classes("w-full gap-4")

        async def load_portfolio():
            try:
                data = await api_get("/portfolio/me")
                loading.style("display:none;")

                with portfolio_container:
                    # ── Summary cards ─────────────────────────────────
                    ui.html('<div class="section-label">PORTFOLIO SUMMARY</div>')

                    with ui.grid(columns=4).classes("w-full gap-3"):
                        pnl_color = "#00ff88" if data["total_unrealized_pnl"] >= 0 else "#ff4444"
                        pnl_sym   = "+" if data["total_unrealized_pnl"] >= 0 else ""
                        for label, value, color in [
                            ("BALANCE",     f"₹{data['balance']:,.0f}",              "#e2e8f0"),
                            ("INVESTED",    f"₹{data['total_invested']:,.0f}",        "#4488ff"),
                            ("UNREALIZED",  f"{pnl_sym}₹{abs(data['total_unrealized_pnl']):,.0f}", pnl_color),
                            ("TOTAL VALUE", f"₹{data['total_portfolio_value']:,.0f}", "#00ff88"),
                        ]:
                            with ui.card().style(
                                "background:#111318;border:1px solid #1e2330;padding:16px;"
                            ):
                                ui.html(f'<div class="metric-label">{label}</div>')
                                ui.html(f'<div style="font-size:18px;font-weight:600;'
                                        f'color:{color};margin-top:4px;">{value}</div>')

                    # ── Holdings table ────────────────────────────────
                    if data["holdings"]:
                        ui.html('<div class="section-label" style="margin-top:24px;">HOLDINGS</div>')

                        with ui.card().style(
                            "background:#111318;border:1px solid #1e2330;padding:0;overflow:hidden;"
                        ):
                            with ui.row().style(
                                "background:#0a0c0f;padding:10px 16px;"
                                "border-bottom:1px solid #1e2330;"
                            ):
                                for col in ["SYMBOL", "QTY", "AVG PRICE", "CUR PRICE", "P&L"]:
                                    ui.html(f'<div class="metric-label" style="flex:1;">{col}</div>')

                            for h in data["holdings"]:
                                pnl     = h["unrealized_pnl"]
                                pnl_cls = "pnl-positive" if pnl >= 0 else "pnl-negative"
                                pnl_sym = "+" if pnl >= 0 else ""
                                with ui.row().style(
                                    "padding:12px 16px;border-bottom:1px solid #111318;"
                                ):
                                    ui.html(f'<div style="flex:1;font-weight:600;">{h["symbol"]}</div>')
                                    ui.html(f'<div style="flex:1;color:#9ca3af;">{h["quantity"]}</div>')
                                    ui.html(f'<div style="flex:1;color:#9ca3af;">₹{h["average_price"]:,.2f}</div>')
                                    ui.html(f'<div style="flex:1;color:#e2e8f0;">₹{h["current_price"]:,.2f}</div>')
                                    ui.html(f'<div class="{pnl_cls}" style="flex:1;">{pnl_sym}₹{abs(pnl):,.0f}</div>')
                    else:
                        ui.html('<div style="color:#4b5563;font-size:12px;margin-top:12px;">'
                                'No holdings yet. Use the Analysis page to find a stock and buy here.</div>')

                # ── Paper trading section ─────────────────────────────
                with trade_container:
                    ui.html('<div class="section-label" style="margin-top:8px;">PAPER TRADING</div>')

                    ui.html(
                        '<div style="background:#111318;border:1px solid #1e2330;border-radius:8px;'
                        'padding:12px 16px;font-size:11px;color:#6b7280;margin-bottom:12px;">'
                        'Practice buying and selling with your virtual ₹1,00,000 balance. '
                        'No real money involved — this is paper trading to help you learn.'
                        '</div>'
                    )

                    with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                        trade_symbol = ui.input(
                            placeholder="Ticker (e.g. SBIN.NS, AAPL)"
                        ).classes("nicegui-input").style("flex:2;min-width:140px;")

                        trade_qty = ui.input(
                            placeholder="Quantity"
                        ).classes("nicegui-input").style("flex:1;min-width:80px;")

                        trade_status = ui.html("").style(
                            "font-size:11px;min-height:16px;width:100%;"
                        )

                        async def do_buy():
                            sym = trade_symbol.value.strip().upper()
                            qty = trade_qty.value.strip()
                            if not sym or not qty:
                                trade_status.set_content(
                                    '<span style="color:#ff8800;">Enter symbol and quantity</span>'
                                )
                                return
                            try:
                                resp = await api_post(
                                    "/trade/buy",
                                    {"symbol": sym, "quantity": int(qty)},
                                )
                                bal = resp.get("remaining_balance", 0)
                                trade_status.set_content(
                                    f'<span style="color:#00ff88;">'
                                    f'✓ Bought {qty} × {sym} | Balance: ₹{bal:,.0f}'
                                    f'</span>'
                                )
                                trade_symbol.value = ""
                                trade_qty.value    = ""
                                # Refresh page to show updated holdings
                                await asyncio.sleep(1)
                                ui.navigate.reload()
                            except httpx.HTTPStatusError as e:
                                detail = e.response.json().get("detail", str(e))
                                trade_status.set_content(
                                    f'<span style="color:#ff4444;">✗ {detail}</span>'
                                )
                            except Exception as e:
                                trade_status.set_content(
                                    f'<span style="color:#ff4444;">✗ {str(e)}</span>'
                                )

                        async def do_sell():
                            sym = trade_symbol.value.strip().upper()
                            qty = trade_qty.value.strip()
                            if not sym or not qty:
                                trade_status.set_content(
                                    '<span style="color:#ff8800;">Enter symbol and quantity</span>'
                                )
                                return
                            try:
                                resp = await api_post(
                                    "/trade/sell",
                                    {"symbol": sym, "quantity": int(qty)},
                                )
                                bal = resp.get("updated_balance", 0)
                                trade_status.set_content(
                                    f'<span style="color:#ff4444;">'
                                    f'✓ Sold {qty} × {sym} | Balance: ₹{bal:,.0f}'
                                    f'</span>'
                                )
                                trade_symbol.value = ""
                                trade_qty.value    = ""
                                await asyncio.sleep(1)
                                ui.navigate.reload()
                            except httpx.HTTPStatusError as e:
                                detail = e.response.json().get("detail", str(e))
                                trade_status.set_content(
                                    f'<span style="color:#ff4444;">✗ {detail}</span>'
                                )
                            except Exception as e:
                                trade_status.set_content(
                                    f'<span style="color:#ff4444;">✗ {str(e)}</span>'
                                )

                        ui.button("BUY", on_click=do_buy).props("flat").style(
                            "background:#0d2b1a;color:#00ff88;border:1px solid #00ff88;"
                            "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
                            "font-weight:700;padding:0 20px;height:40px;"
                        )
                        ui.button("SELL", on_click=do_sell).props("flat").style(
                            "background:#2b0d0d;color:#ff4444;border:1px solid #ff4444;"
                            "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
                            "font-weight:700;padding:0 20px;height:40px;"
                        )

                    trade_status  # render status below buttons

                # ── Trivia cards ──────────────────────────────────────
                with trivia_container:
                    ui.html('<div class="section-label" style="margin-top:8px;">LEARN AS YOU TRADE</div>')

                    # Pick 3 random trivia items each page load
                    shown = _random.sample(TRADING_TRIVIA, min(3, len(TRADING_TRIVIA)))

                    with ui.grid(columns=1).classes("w-full gap-3"):
                        for title, body in shown:
                            with ui.card().style(
                                "background:#111318;border:1px solid #1e2330;"
                                "border-radius:8px;padding:16px;"
                            ):
                                ui.html(
                                    f'<div style="font-size:11px;letter-spacing:2px;'
                                    f'color:#00ff88;margin-bottom:8px;">{title}</div>'
                                    f'<div style="font-size:12px;color:#9ca3af;line-height:1.7;">'
                                    f'{body}</div>'
                                )

                # ── Trade history ─────────────────────────────────────
                trades_data = await api_get("/trades/me")
                trades      = trades_data.get("trades", [])

                with trades_container:
                    ui.html('<div class="section-label">TRADE HISTORY</div>')

                    if trades:
                        with ui.card().style(
                            "background:#111318;border:1px solid #1e2330;padding:0;overflow:hidden;"
                        ):
                            with ui.row().style(
                                "background:#0a0c0f;padding:10px 16px;"
                                "border-bottom:1px solid #1e2330;"
                            ):
                                for col in ["SYMBOL", "TYPE", "QTY", "PRICE", "TIME"]:
                                    ui.html(f'<div class="metric-label" style="flex:1;">{col}</div>')

                            for t in reversed(trades[-30:]):
                                t_color = "#00ff88" if t["trade_type"] == "BUY" else "#ff4444"
                                ts      = str(t.get("timestamp", ""))[:16]
                                with ui.row().style(
                                    "padding:10px 16px;border-bottom:1px solid #111318;"
                                ):
                                    ui.html(f'<div style="flex:1;font-weight:600;">{t["symbol"]}</div>')
                                    ui.html(f'<div style="flex:1;color:{t_color};">{t["trade_type"]}</div>')
                                    ui.html(f'<div style="flex:1;color:#9ca3af;">{t["quantity"]}</div>')
                                    ui.html(f'<div style="flex:1;color:#9ca3af;">₹{t["price"]:,.2f}</div>')
                                    ui.html(f'<div style="flex:1;color:#4b5563;font-size:11px;">{ts}</div>')
                    else:
                        ui.html('<div style="color:#4b5563;font-size:12px;">'
                                'No trades yet. Use the BUY/SELL panel above to practice.</div>')

            except Exception as e:
                loading.set_content(f'<span style="color:#ff4444;">Error loading portfolio: {str(e)}</span>')
                with portfolio_container:
                    ui.button("↩ GO BACK TO ANALYSIS", on_click=lambda: ui.navigate.to("/")).props("flat").style(
                        "color:#00ff88;font-family:'IBM Plex Mono',monospace;margin-top:12px;"
                    )

        await load_portfolio()


# ==============================
# PAGE 3: AI Chat
# ==============================

@ui.page("/chat")
async def chat_page():
    apply_styles()

    if not state["token"]:
        show_login_dialog(lambda: ui.navigate.to("/chat"))
        return

    if not state.get("explain_result"):
        ui.navigate.to("/")
        return

    navbar("/chat")

    result  = state["explain_result"]
    symbol  = result.get("symbol", "")
    company = result.get("company", "")
    rec     = result.get("recommendation", "")
    intent  = state.get("intent", {})

    with ui.column().classes("w-full max-w-3xl mx-auto px-6 py-8 gap-4"):

        # ── Stock header ─────────────────────────────────────────
        with ui.row().classes("items-center justify-between"):
            with ui.column().classes("gap-1"):
                ui.html(f'<div class="ticker-symbol">{symbol}</div>')
                ui.html(f'<div style="color:#6b7280;font-size:11px;">{company}</div>')
            ui.html(signal_badge(rec))

        # ── Key stats ────────────────────────────────────────────
        if result.get("quantity"):
            intent_curr = intent.get("currency", "")
            ui.html(
                f'<div style="background:#0d1f14;border:1px solid #00ff8833;'
                f'border-radius:6px;padding:12px;font-size:12px;">'
                f'<span style="color:#00ff88;">▶ SUGGESTED:</span> '
                f'{result["quantity"]} shares'
                + (f' · {intent_curr} {intent.get("budget"):,.0f} budget' if intent.get("budget") else "")
                + f'<br><span style="color:#4b5563;font-size:11px;">{result.get("quantity_reasoning","")}</span>'
                f'</div>'
            )

        # ── Chat messages ────────────────────────────────────────
        chat_container = ui.column().classes("w-full gap-2 chat-scroll-area").style(
            "max-height:50vh;overflow-y:auto;padding:4px;scroll-behavior:smooth;"
        )

        def _add_message(role: str, text: str):
            with chat_container:
                css = "chat-user" if role == "user" else "chat-ai"
                prefix = "YOU" if role == "user" else "AI"
                color  = "#4488ff" if role == "user" else "#00ff88"
                ui.html(
                    f'<div class="{css}">'
                    f'<span style="color:{color};font-size:10px;letter-spacing:2px;">{prefix}</span><br>'
                    f'{text}'
                    f'</div>'
                )

        # Show initial AI explanation
        _add_message("ai", result.get("explanation", ""))

        # Key points
        kp = result.get("key_points", [])
        if kp:
            with chat_container:
                pts = "".join(f"<div style='margin:2px 0;'>· {p}</div>" for p in kp)
                ui.html(
                    f'<div class="chat-ai">'
                    f'<span style="color:#00ff88;font-size:10px;letter-spacing:2px;">KEY POINTS</span><br>'
                    f'{pts}'
                    f'</div>'
                )

        # Risk warning
        rw = result.get("risk_warning", "")
        if rw:
            with chat_container:
                ui.html(
                    f'<div style="background:#2b0d0d;border-left:3px solid #ff4444;'
                    f'padding:10px 14px;border-radius:0 8px 8px 0;font-size:12px;">'
                    f'<span style="color:#ff4444;font-size:10px;letter-spacing:2px;">RISK</span><br>'
                    f'{rw}'
                    f'</div>'
                )

        # Disclaimer
        disc = result.get("disclaimer", "")
        if disc:
            with chat_container:
                ui.html(
                    f'<div style="color:#4b5563;font-size:10px;margin-top:8px;">{disc}</div>'
                )

        # ── Suggestion chips ─────────────────────────────────────
        ui.html('<div class="section-label" style="margin-top:8px;">QUICK QUESTIONS</div>')

        suggestions = [
            "Explain like I'm a complete beginner",
            "What is the worst case scenario?",
            "Should I invest all at once or spread it?",
            "What does this risk level mean practically?",
        ]

        with ui.row().classes("w-full flex-wrap gap-2"):
            for s in suggestions:
                chip_btn = ui.button(s).props("flat no-caps").style(
                    "background:#111318;color:#9ca3af;"
                    "border:1px solid #1e2330;font-size:11px;"
                    "padding:4px 10px;border-radius:4px;"
                )
                chip_btn.on("click", lambda s=s: send_message(s))

        # ── Input row ────────────────────────────────────────────
        sending = ui.html("").style("font-size:12px;")

        with ui.row().classes("w-full gap-2 items-end"):
            msg_input = ui.input(
                placeholder="Ask anything about this analysis..."
            ).classes("flex-1 nicegui-input")

            send_btn = ui.button("SEND ▶").props("flat").style(
                "background:#00ff88;color:#0a0c0f;font-weight:700;"
                "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
                "height:40px;padding:0 16px;"
            )

        async def send_message(msg: str = None):
            text = msg or msg_input.value.strip()
            if not text:
                return

            msg_input.value = ""
            _add_message("user", text)
            sending.set_content('<span class="loading-text">AI IS THINKING...</span>')

            try:
                resp = await api_post("/analyse/chat", {
                    "user_message":         text,
                    "conversation_history": state["conversation_history"],
                })
                ai_response = resp.get("response", "")
                state["conversation_history"] = resp.get("conversation_history", [])
                _add_message("ai", ai_response)

            except Exception as e:
                _add_message("ai", f"Error: {str(e)}")
            finally:
                sending.set_content("")

        send_btn.on("click", lambda: send_message())
        msg_input.on("keydown.enter", lambda: send_message())

        # Scroll chat to bottom
        # Auto-scroll chat container to bottom after new messages
        ui.run_javascript("""
            setTimeout(() => {
                const container = document.querySelector('.chat-scroll-area');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                } else {
                    const msgs = document.querySelectorAll('.chat-ai, .chat-user');
                    if (msgs.length) msgs[msgs.length-1].scrollIntoView({behavior:'smooth'});
                }
            }, 150);
        """)


# ==============================
# Register page (bonus)
# ==============================

@ui.page("/register")
async def register_page():
    apply_styles()

    with ui.column().classes("w-full items-center justify-center").style("min-height:100vh;"):
        with ui.card().style(
            "background:#111318;border:1px solid #1e2330;min-width:360px;padding:32px;"
        ):
            ui.html('<div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;'
                    'color:#00ff88;margin-bottom:8px;">▲ TRADEAI</div>')
            ui.html('<div style="color:#4b5563;font-size:11px;letter-spacing:2px;margin-bottom:24px;">'
                    'CREATE ACCOUNT</div>')

            username = ui.input("Username").classes("w-full nicegui-input").style("margin-bottom:12px;")
            password = ui.input("Password", password=True).classes("w-full nicegui-input").style("margin-bottom:16px;")
            status   = ui.html("").style("font-size:12px;margin-bottom:12px;")

            async def do_register():
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        f"{API_BASE}/auth/register",
                        params={"username": username.value, "password": password.value},
                    )
                    if resp.status_code == 200:
                        status.set_content('<span style="color:#00ff88;">Account created! Logging you in...</span>')
                        # Auto-login after register so user lands directly on analysis page
                        ok = await api_login(username.value, password.value)
                        if ok:
                            ui.navigate.to("/")
                        else:
                            await asyncio.sleep(1)
                            ui.navigate.to("/")
                    else:
                        detail = resp.json().get("detail", "Registration failed")
                        status.set_content(f'<span style="color:#ff4444;">{detail}</span>')

            ui.button("CREATE ACCOUNT", on_click=do_register).props("flat").style(
                "background:#00ff88;color:#0a0c0f;font-weight:700;width:100%;"
                "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
            )
            ui.html('<div style="color:#4b5563;font-size:11px;margin-top:16px;text-align:center;">'
                    'Have an account? <a href="/" style="color:#00ff88;">Sign in</a></div>')


# ==============================
# PAGE 4: Profile
# ==============================

@ui.page("/profile")
async def profile_page():
    apply_styles()

    if not state["token"]:
        show_login_dialog(lambda: ui.navigate.to("/profile"))
        return

    navbar("/profile")

    with ui.column().classes("w-full max-w-2xl mx-auto px-6 py-12 gap-6"):

        # ── Avatar + name ─────────────────────────────────────────
        with ui.column().classes("items-center gap-3").style("margin-bottom:8px;"):
            # Initials avatar
            initials = (state["username"] or "?")[0].upper()
            ui.html(
                f'<div style="width:80px;height:80px;border-radius:50%;'
                f'background:#0d2b1a;border:2px solid #00ff88;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-family:Syne,sans-serif;font-size:32px;font-weight:800;'
                f'color:#00ff88;">{initials}</div>'
            )
            ui.html(
                f'<div style="font-family:Syne,sans-serif;font-size:22px;'
                f'font-weight:800;color:#e2e8f0;">{(state["username"] or "").upper()}</div>'
            )
            ui.html('<div style="font-size:11px;letter-spacing:2px;color:#4b5563;">PAPER TRADER</div>')

        # ── Stats from portfolio ──────────────────────────────────
        stats_container = ui.column().classes("w-full gap-3")

        async def load_stats():
            try:
                data = await api_get("/portfolio/me")
                with stats_container:
                    ui.html('<div class="section-label">ACCOUNT STATS</div>')
                    with ui.grid(columns=2).classes("w-full gap-3"):
                        pnl      = data["total_unrealized_pnl"]
                        pnl_col  = "#00ff88" if pnl >= 0 else "#ff4444"
                        pnl_sym  = "+" if pnl >= 0 else ""
                        for label, value, color in [
                            ("VIRTUAL BALANCE",  f"₹{data['balance']:,.0f}",         "#e2e8f0"),
                            ("TOTAL VALUE",      f"₹{data['total_portfolio_value']:,.0f}", "#00ff88"),
                            ("INVESTED",         f"₹{data['total_invested']:,.0f}",   "#4488ff"),
                            ("UNREALIZED P&L",   f"{pnl_sym}₹{abs(pnl):,.0f}",         pnl_col),
                        ]:
                            with ui.card().style(
                                "background:#111318;border:1px solid #1e2330;padding:16px;"
                            ):
                                ui.html(f'<div class="metric-label">{label}</div>')
                                ui.html(
                                    f'<div style="font-size:16px;font-weight:600;'
                                    f'color:{color};margin-top:4px;">{value}</div>'
                                )
            except Exception:
                pass

        await load_stats()

        # ── Account info ──────────────────────────────────────────
        ui.html('<div class="section-label" style="margin-top:8px;">ACCOUNT INFO</div>')
        with ui.card().style(
            "background:#111318;border:1px solid #1e2330;padding:20px;width:100%;"
        ):
            for label, value in [
                ("Username",     state["username"] or "—"),
                ("Account type", "Paper Trading (Virtual)"),
                ("Starting balance", "₹1,00,000"),
                ("Platform",     "TradeAI — AI-Powered Paper Trading"),
            ]:
                with ui.row().style(
                    "padding:10px 0;border-bottom:1px solid #1a1f2b;"
                ):
                    ui.html(
                        f'<div style="flex:1;font-size:11px;letter-spacing:1px;'
                        f'color:#4b5563;text-transform:uppercase;">{label}</div>'
                        f'<div style="flex:2;font-size:12px;color:#9ca3af;">{value}</div>'
                    )

        # ── Logout ────────────────────────────────────────────────
        ui.html('<div class="section-label" style="margin-top:8px;">SESSION</div>')

        def do_logout():
            state["token"]                = None
            state["username"]             = None
            state["comparison"]           = None
            state["selected_card"]        = None
            state["explain_result"]       = None
            state["conversation_history"] = []
            state["intent"]               = {}
            ui.navigate.to("/")

        ui.button("LOGOUT →", on_click=do_logout).props("flat").style(
            "background:#2b0d0d;color:#ff4444;border:1px solid #ff4444;"
            "font-family:'IBM Plex Mono',monospace;letter-spacing:2px;"
            "font-weight:700;padding:0 24px;height:44px;width:100%;"
        )

        ui.html(
            '<div style="color:#4b5563;font-size:10px;text-align:center;margin-top:4px;">'
            'Logging out clears your session. Your trades and portfolio are saved.'
            '</div>'
        )


# ==============================
# Run
# ==============================

ui.run(
    host           = "0.0.0.0",
    port           = 3000,
    title          = "TradeAI",
    dark           = True,
    reload         = False,
    show           = False,
    storage_secret = "tradeai_secret_change_this",
)