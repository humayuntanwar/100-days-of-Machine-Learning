#reading from our twitter db which is being live updated
import sqlite3
import pandas as pd

conn = sqlite3.connect('twitter.db')
c = conn.cursor()
#passing sql query
df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%trump%' ORDER BY unix DESC LIMIT 1000", conn)
#sort by unix
df.sort_values('unix', inplace=True)
# using rolling mean to smooth sentiment
df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
df.dropna(inplace=True)
print(df.tail())
