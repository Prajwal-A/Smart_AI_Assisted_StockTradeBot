# # import asyncio
# # from services.technical_services import get_technical_snapshot
# # from services.claude_technical_services import scan_symbols
# # from services.risk_service import calculate_risk


# # async def test():

# #     result = await get_technical_snapshot("INFY.NS")
# #     print("Technical Snapshot:")
# #     print(result)
# #     print("\nDetailed Snapshot:")

# #     snapshots = await scan_symbols("INFY.NS")

# #     for snap in snapshots:
# #             print(f"\n{'='*50}")
# #             print(f"  {snap.symbol} — ${snap.current_price}")
# #             print(f"  Trend     : {snap.trend}")
# #             print(f"  Momentum  : {snap.momentum}  (RSI: {snap.rsi})")
# #             print(f"  MACD Hist : {snap.macd_histogram}")
# #             print(f"  BB Pos    : {snap.bb_position:.2%}")
# #             print(f"  Signal    : {snap.signal}  (Strength: {snap.signal_strength}/100)")
# #             print(f"  Risk      : {snap.risk_level}")
# #             print(f"  Volume OK : {snap.volume_confirmed}")
# #             print(f"  Notes     :")
# #             for note in snap.notes:
# #                 print(f"    • {note}")


# #     risk = calculate_risk(result)
# #     print("Risk Assessment:")
# #     print(risk)



# # if __name__ == "__main__":
# #     asyncio.run(test())



import google.generativeai as genai

genai.configure(api_key="YOUR_GOOGLE_API_KEY")  # Replace with your actual API key

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Explain Retrieval Augmented Generation"
)

print(response.text)

