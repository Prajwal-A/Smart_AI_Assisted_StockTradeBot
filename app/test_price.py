import asyncio
from services.market_data import (
    get_current_price,
    validate_symbol
)


async def test():

    print("Testing valid symbol...")
    result = await validate_symbol("TCS")
    print("Validation Result:", result)

    if result.get("valid"):
        price = await get_current_price(result["symbol"])
        print("Current Price:", price)

    print("\nTesting invalid symbol...")
    result = await validate_symbol("TCSX")
    print("Validation Result:", result)

    print("\nTesting US stock...")
    result = await validate_symbol("AAPL")
    print("Validation Result:", result)

    if result.get("valid"):
        price = await get_current_price(result["symbol"])
        print("Current Price:", price)


if __name__ == "__main__":
    asyncio.run(test())