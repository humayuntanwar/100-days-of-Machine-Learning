#Learing Regression
#Import Pandas
import pandas as pd
#Import Quandl for data
import quandl 
#import math
import math
#import numpy allows us to use arrays
import numpy as np
#scaling on features , helps in accuracy, testing and validation, svm suport vector machine
from sklearn import preprocessing, model_selection, svm
#imorting linear regression
from sklearn.linear_model import LinearRegression
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
print(forecast_out)# 343 days
#print(df.head())
#feature will be X, features are everything accept label column
X = np.array(df.drop(['label'],1))
#labels will be Y
Y = np.array(df['label'])
#scale X
X = preprocessing.scale(X)
#redefine X, where values for Y
#X = X[:-forecast_out+1]
#df.dropna(inplace = True)
Y = np.array(df['label'])
#print(len(X), len(Y))
#ready to creating traing and test , test_size is 20% data
X_trian, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2)
#define Linear Regression also switch toSvm classifier and fit
clf = LinearRegression()
clf.fit(X_trian, Y_train)
#test classifier
accuracy  = clf.score(X_test,Y_test)

print(accuracy) # 0.88 with Linear Regression , 0.73 with SVR, with kernel poly 0.63