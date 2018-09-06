import numpy as np 
from sklearn import preprocessing, model_selection, neighbors,svm
import pandas as pd
accuracies =  []

for i in range(25):
    df = pd.read_csv('datasets/breast-cancer-wisconsin.data.txt') #reading data set
    df.replace('?', -99999, inplace=True) # repalce ? with -99999 outlier 
    df.drop(['id'],1,inplace=True) # dropping id column

    X= np.array(df.drop(['class'],1) )#features are everything except class
    Y= np.array(df['class']) #labels is class

    X_trian,X_test,Y_trian,Y_test = model_selection.train_test_split(X,Y,test_size=0.2) #shuffle transform data into triant and test data

    #defining classifiers
   # clf =neighbors.KNeighborsClassifier()
    clf = svm.SVC() # using built in SVC from sklearn
    # trian data
    clf.fit(X_trian,Y_trian) 
    #testing classifier
    accuracy = clf.score(X_test,Y_test) 
    # print(accuracy) #currently 97.85% which type of cancer

    # example_measures = np.array([4,2,1,1,1,2,3,2,1]) #testing ton this data

    # example_measures= example_measures.reshape(1,-1) #get rid of deprecation value, reshaping array

    # prediction = clf.predict(example_measures) # predictions
    # print(prediction) #output 2

    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies)) # according to numpy k nearest 97.2%
