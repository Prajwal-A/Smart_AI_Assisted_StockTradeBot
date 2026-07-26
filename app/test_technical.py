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

# d = {1 : 'a', 2 : 'b'}
# d[1.0] = 'c'
# print(d)

# lst = [1, 2, 3, 4, 5]
# for i in lst[:]:
#     if i % 2 == 0:
#         lst.remove(i)
# print(lst)

# def gen():
#     yield 1
#     yield 2
# g = gen()
# print(list(g), list(g))

# a = [1, 2, 3]
# b = a
# a = a + [4, 5]
# print(b)

# for i in range(5):
#     if i == 1:
#         break
# else:
#     print("Loop completed without break")
# print("Done")

# lst = [1, 2, 3, 4]
# print(lst[1:3][1:2][0])

# lst = [1, 2, 3, 4]
# for i in lst:
#     lst.remove(i)
# print(lst)

# nums = [1, 2, 3, 4]
# res = [nums.pop(0) for _ in range(len(nums))]
# print(res)


import google.generativeai as genai

genai.configure(api_key="AIzaSyCeAMHEtdGfCdWoZf-HBvxzQhG1IfleJyI")

model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "Explain Retrieval Augmented Generation"
)

print(response.text)

