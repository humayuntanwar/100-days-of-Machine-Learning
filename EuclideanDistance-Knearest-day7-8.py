from math import sqrt
plot1 =[4,5] # points 1
plot2 =[1,2] #points 2
euclidean_distance = sqrt( ( plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 ) # euclidean distance formula
print(euclidean_distance) #4.242

#creating our own kNearest neighbors algorithms.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

#creating two classes and their features
dataset =  {'k':[ [1,2], [2,3], [3,1]],"r":[[6,5], [7,7], [8,6]]}

new_features = [5,7]

#one line for loop for plotting
# [ [ plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],s=100)
# plt.show()

## pass the data, what we need to predict , k nearest val
def k_nearest_neighbors(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to a valnue less than total voting group')

    distances = [] # distances equals list
    for group in data:
        for features in data[group]:
            euclidean_distance_np = np.linalg.norm(np.array(features)- np.array(predict)) #ED from numpy
            distances.append([euclidean_distance_np,group])# fist item ED, second group
    
    votes = [i[1] for i in sorted(distances)[:k]] # populates votes as list
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0] # most common grups and how many in it
    
    confidence = Counter(votes).most_common(1)[0][1] / k # confidence comes from classifier
    #print(vote_result, confidence)

    return vote_result, confidence

result = k_nearest_neighbors(dataset, new_features,k=3)
print(result) # result is R exactly according to our defined data set


# [ [ plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],color = result, s=100)
# plt.show()

#DAY 8
#applying our own k nearest on breast cancer dataset

import pandas as pd
import random

accuracies =  []

for i in range(25):
    df = pd.read_csv('datasets/breast-cancer-wisconsin.data.txt')  # read dataset
    df.replace('?',-99999,inplace=True) # replace empty with -99999
    df.drop(['id'],1,inplace=True)#dropping id column , because it reduces accurarcy

    #print(df.head())

    full_data = df.astype(float).values.tolist() #make sure all values are in float
    print(full_data[:5]) # unshuffled

    #shuffling the data because its a list of list
    random.shuffle(full_data)
    #print(20*'#')
    #print(full_data[:5]) #after shuffle

    # our version of trained test spilt
    test_size = 0.2 # 20% of data
    train_set = {2:[],4:[]} # 2 , 4 represent types of cancer according to dataset
    test_set = {2:[],4:[]} # these are dictionaries
    train_data = full_data[:-int(test_size*len(full_data))] #first 20% of the data
    test_data = full_data[-int(test_size*len(full_data)):] # last 20% of the data

    #populate the dictionaries

    # train set  i negative 1 , because last column is class, append upto the last one
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) 

    # test populate
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    # for each group in test set we are testing these, for data in testing about to feed through
    #dictionary from train set,  k=5 sklearn default
    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set,data,k=5)
            if group ==vote:
                correct +=1
            else:
                print(confidence) # confidence score of votes we got incorrect
            total +=1

    print('Accurarcy:', correct/total) # this is our own algorithm based accuract 95%
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies)) # our own 96.7%







