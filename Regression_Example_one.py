#Learing Regression
#Import Pandas
import pandas as pd
#Import Quandl for data
import quandl 
#import math
import math

#get google wiki data
df = quandl.get('WIKI/GOOGL')
#each column is a feature
#create a long list of all the columns we need
#open prcice, close price, highest and lowest price, total volume
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#print(df.head())
#finding percent volitility HL_PCT coloum name . high - low /low
df['HL_PCT'] = (df['Adj. High']- df['Adj. Low'])/ df['Adj. Low'] *100.00
#daily mood 
df['PCT_change'] = (df['Adj. Close']- df['Adj. Open'])/ df['Adj. Open'] *100.00

#define new data frame 
df= df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print df
#print(df.head())
#make a forecast column
forecast_col = 'Adj. Close'
#fill na not available
df.fillna(-99999,inplace=True)
#math.ceil round to nearest val convert to int predict 10% of data 
forecast_out = int(math.ceil(0.1*len(df)))
#create labels, adjusted close price in 10 days 
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

