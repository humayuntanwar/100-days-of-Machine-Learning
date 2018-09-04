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
    return vote_result

result = k_nearest_neighbors(dataset, new_features,k=3)
print(result) # result is R exactly according to our defined data set


[ [ plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1],color = result, s=100)
plt.show()