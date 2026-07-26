START SERVER from trading_bot directory:  
python -m uvicorn app.main:app --reload 


brew services restart mongodb-community
mongosh

test> use trading_assistant
switched to db trading_assistant
trading_assistant> show collections
portfolios
users
trading_assistant> db.portfolios.find().pretty()
