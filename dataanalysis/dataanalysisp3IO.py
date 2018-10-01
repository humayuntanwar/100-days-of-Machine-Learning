## handling differnet type of data types
##using IO framework
## uisng quandl datasets
##prentend we are billionaires buying houses in austin

#import quandl
#get data
#df = quandl.get("ZILLOW/Z77004_ZRISFRR")
#save as csv
#df.to_csv('data.csv')
# http://pandas.pydata.org/pandas-docs/stable/io.html

import pandas as pd
#read csv
#df = pd.read_csv('data.csv')
#print first 5
print(df.head())
#set date as index
df.set_index('Date',inplace=True)
df.to_csv('newdata.csv')

#redifine df
df = pd.read_csv('newdata.csv')
print(df.head())
#date as index(column 0 as index)
df = pd.read_csv('newdata.csv',index_col=0)
print(df.head())

#rename columns
df.columns = ['Austin HPI']
print(df.head())

#save as csv
df.to_csv('newdata3.csv')

#what if you dont want header in csv
f.to_csv('newdata4.csv',header =False)

#read headerless csv (give columns name)
df = pd.read_csv('newdata4.csv',names=['Date', 'Austin HPI'],index_col=0)
print(df.head())


# convert to file types other
df.to_html('example.html')# to hml

df = pd.read_csv('newdata4.csv',names=['Date','AUSTIN_HPI'])
print(df.head())

#rename columns
df.rename(columns={'AUSTIN_HPI':'77006_HPI'},inplace=True)
print(df.head())