import datetime
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

start = datetime.datetime(2018, 1, 1)
end = datetime.datetime.now()

df = web.DataReader('GOOGL', 'iex', start, end)

print(df.head())

df['close'].plot()

plt.show()
