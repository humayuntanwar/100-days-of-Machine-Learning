import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

X = np.array([[1, 2],
              [5, 8],
              [1.5, 1.8],
              [8, 8],
              [1, 0.6],
              [9, 11],[1, 3],
              [8, 9],
              [8,2],
              [10,2],
              [9,3],])

colors = 10*["g","r","c","y"]


#first assign every signle feature set is a cluster center
# take all of the data points or featureset within that cluster center radius
# within the bandwidth
#take the mean of all the featuresets, that is your new cluster center
# repeat step 2 until you have convergence

class Mean_shift:
    # for now bandwidth = 4
    def __init__(self, radius=4):
        self.radius = radius


    def fit (self,data):
        centroids = {} # empty dict

        #set initial centroids
        for i in range(len(data)):
            centroids[i]= data[i] # centroid i= id, value is datai

        # can have max iter and tolerance
        #infinite loop
        while True:
            new_centroids = [] #empty list
            #cycle through known centroids
            for i in centroids:
                in_bandwidth = []#emty list
                centroid = centroids[i] #new centroid
                # now iterate through the data and decide fetature set is in bandwidth
                for featureset in data:
                    # euclidean distance < radius meands within radius
                    if np.linalg.norm(featureset-centroid) < self.radius:
                        #just append it
                        in_bandwidth.append(featureset)
                #recalculate means , gives means venctor
                new_centroid = np.average(in_bandwidth,axis=0)
                # add to newcentroids list
                new_centroids.append(tuple(new_centroid)) # coverting array to tuple

            #get unique elements from new centroids list
            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids) # coping

            centroids = {} # empty dict
            #populate it 
            for i in range(len(uniques)):
                #cenverting back to array
                centroids[i] = np.array(uniques[i])

            optimized = True
            
            for i in centroids:
                #compare to see if equals
                if not np.array_equal(centroids[i],prev_centroids[i]):
                    optimized=False
                
                if not optimized:
                    break
            if optimized:
                break

        #reset centroids
        self.centroids =centroids
    
    def predict(self,data):
        pass

clf= Mean_shift()
clf.fit(X)
centroids = clf.centroids

plt.scatter(X[:,0],X[:,1],s=150)

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color='k',marker="*",s=150)

plt.show()










