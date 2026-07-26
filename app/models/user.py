from beanie import Document
from pydantic import Field


class User(Document):
    username: str = Field(..., unique=True)
    password: str

    class Settings:
        name = "users"