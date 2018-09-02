#Learing Regression
#Import Pandas
import pandas as pd
#Import Quandl for data
import quandl 
#import math & date time
import math, datetime
#import numpy allows us to use arrays
import numpy as np
#scaling on features , helps in accuracy, testing and validation, svm suport vector machine
from sklearn import preprocessing, model_selection, svm
#imorting linear regression
from sklearn.linear_model import LinearRegression
#import matplotmib
import matplotlib.pyplot as plt
from matplotlib import style # make it decent
import pickle # import pickle works like opening and saving file

style.use('ggplot') # specify style type

#get google wiki data
#df = quandl.get('WIKI/GOOGL')
#df.to_csv('data.csv')

#each column is a feature
#create a long list of all the columns we need
#open prcice, close price, highest and lowest price, total volume
df = pd.read_csv('data.csv',header=0, 
                  index_col='Date',
                  parse_dates=True)
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
print(forecast_out)# 343 days
#print(df.head())

#feature will be X, features are everything accept label column
X = np.array(df.drop(['label'],1))
#scale X
#X = X[:-forecast_out+1]
X = preprocessing.scale(X)
# now we make predictions 
# to the - 
X_lately = X[-forecast_out:] # this is the forecast set
#x  equal to the point - forecast out
X = X[:-forecast_out]

#df.dropna(inplace=True)
#labels will be Y
Y=df['label']
Y.dropna(inplace=True)
Y=np.array(Y)
#Y = np.array(df['label'])
#redefine X, where values for Y
#df.dropna(inplace = True)
#Y = np.array(df['label'])

#print(len(X), len(Y))
#ready to creating traing and test , test_size is 20% data
X_trian, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size=0.2)
#define Linear Regression also switch toSvm classifier and fit
#clf = LinearRegression() 
#clf.fit(X_trian, Y_train)#pickle is used to save classifier saves times no need to retrian
#with open('LinearRegression.pickle','wb') as f:
 #   pickle.dump(clf,f) #dump classifier
pickle_in = open('LinearRegression.pickle','rb')
clf = pickle.load(pickle_in)
#test classifier
accuracy  = clf.score(X_test,Y_test)

print(accuracy) # 0.88 with Linear Regression , 0.73 with SVR, with kernel poly 0.63

#day 4
#predict based on x data, can pass single value or array fo data
forecast_set = clf.predict(X_lately) #ACTUAL PREDICTION
print(forecast_set,accuracy, forecast_out) # now we have predicted price for next 343 days 

df['Forecast'] = np.nan # specify entire col is full of nan data
last_date = df.iloc[-1].name #find last date
last_unix = last_date.timestamp()
one_day = 86400 # num of seconds in one day
next_unix = last_unix + one_day
#polulate data frame
for i in forecast_set:
     next_date = datetime.datetime.fromtimestamp(next_unix)
     next_unix+= one_day
     # one line for loop iterating through forecast set taking each forecast data and
     #  setting those as values basically making the future features nan numbers , 
     # the last line just takes of all the first cloumn not a number and the final coloum what ever i is ,
     #  which is forecast in this case
     #df loc references next date , makes it index
     df.loc[next_date] = [np.nan for _ in range (len(df.columns)-1)]+[i]

# there is a gap
#The reason is because there is no overlap between the historical data and the forecast data in terms of the date index. 
# You need a one day overlap for the line to be continuous. 
last_row_before_forecast = df.loc[last_date]
df.loc[last_date] = np.hstack((last_row_before_forecast.values[:-1], last_row_before_forecast[forecast_col]))
df['Adj. Close'].plot() # plot adj close
df['Forecast'].plot() # plot forecast
plt.legend(loc=4) # put in 4th location
plt.xlabel ('Date') # this is date
plt.ylabel('Price') # this is price

plt.show() # show plot
