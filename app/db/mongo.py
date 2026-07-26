from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.models.user import User
from app.models.portfolio import Portfolio
from app.models.trade import Trade
from app.models.holding import Holding

MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "trading_assistant"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

async def init_db():
    await init_beanie(
        database=db,
        document_models=[User, Portfolio, Trade, Holding]
    )
