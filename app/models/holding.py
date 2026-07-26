from beanie import Document


class Holding(Document):
    user_id: str
    symbol: str
    quantity: int
    average_price: float

    class Settings:
        name = "holdings"