from beanie import Document
from pydantic import Field


class Portfolio(Document):
    user_id: str
    balance: float = Field(default=100000.0)

    class Settings:
        name = "portfolios"