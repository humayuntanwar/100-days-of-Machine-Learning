##Rolling Apply and Mapping Functions
import quandl 
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm, preprocessing, model_selection

style.use('fivethirtyeight')
#get quandl data 
#key.yxt contains quandl auth key
api_key = open('key.txt','r').read()


# function mapping. 
def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

#custom way to apply a moving-window function.
#We're going to just do a simple moving average example:
def moving_average(values):
    ma = mean(values)
    return ma

#reading all data
housing_data = pd.read_pickle('HPI.pickle')

#applying percent change to all data
housing_data = housing_data.pct_change()
#print(housing_data.head())

#handling errornous data
#nan and infinity using numpy 
housing_data.replace([np.inf,-np.inf], np.nan ,inplace=True)
#shift col down one value
housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)

'''
If the future HPI is higher than the current,
this means prices went up, and we are going to return a 1.
This is going to be our label. If the future HPI is not greater
than current, 
then we return a simple 0. To map this function,
'''
housing_data['label'] = list(map(create_labels,housing_data['United States'], housing_data['US_HPI_future']))
#print(housing_data.head())
#Now, you can use rolling_apply:
housing_data['ma_apply_example'] = housing_data['M30'].rolling(window=10, center=False).apply(moving_average)
housing_data.dropna(inplace=True)

#print(housing_data.tail())

#print(housing_data.head())

'''
housing_data = pd.read_pickle('HPI.pickle')
housing_data = housing_data.pct_change()
housing_data.replace([np.inf, -np.inf],np.nan, inplace=True)
housing_data['US_HPI_future'] = housing_data['United States'].shift(-1)
housing_data.dropna(inplace=True)
#print(housing_data[['US_HPI_future','United States']].head())
housing_data['label'] = list(map(create_labels,housing_data['United States'], housing_data['US_HPI_future']))
#print(housing_data.head())
housing_data['ma_apply_example'] = housing_data['M30'].rolling(10).apply(moving_average)
#print(housing_data.tail())
housing_data.dropna(inplace=True)
'''


#incorporating scikit ML SVM algo on our data
#we can create our features and our labels for training/testing:
#The uppercase X is used to denote a feature set. The y is the label

X = np.array(housing_data.drop(['label','US_HPI_future'], axis=1))
#print(X)
X = preprocessing.scale(X)
y = np.array(housing_data['label'])

#Now our labels are defined, and we're ready to split up our data into training and testing sets. We can do this ourselves, but we'll use the model_selection import from earlier:
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Now, we can establish the classifier that we intend to use:
clf = svm.SVC(kernel='linear')

#train our classifier
clf.fit(X_train, y_train)

#Finally, we could actually go ahead and make predictions from here, but let's test the classifier's accuracy on known data:
print(clf.score(X_test, y_test))

#ACCUURACY = 0.792452830189





