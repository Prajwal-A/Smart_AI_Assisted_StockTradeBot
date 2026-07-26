from beanie import Document
from pydantic import Field
from datetime import datetime


class Trade(Document):
    user_id: str
    symbol: str
    quantity: int
    price: float
    trade_type: str  # BUY / SELL
    status: str = Field(default="OPEN")
    realized_pnl: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "trades"