
'''
Applying Comparison Operators to DataFrame 
removing errornous data

'''
import quandl 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
# sample data 6212.42 is errornous
bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}

#make a dataframe
df = pd.DataFrame(bridge_height)

#first thing we can do it std deviation
df['STD'] = df.rolling(2).std()

#using comparison operator, figure out x
#describe dataframe
df_std = df.describe()
print(df_std)
#describe meter col ,std row
df_std = df.describe()['meters']['std']
print(df_std)

#redefine dataframe
df = df [ (df['STD'] < df_std ) ]
print(df)
#plot show
#df.plot()
#output meters
df['meters'].plot()

plt.show()

