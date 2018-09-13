import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

#import blobs
from sklearn.datasets.samples_generator import make_blobs
#import random
import random

centers = random.randrange(2,5)

X,Y =make_blobs(n_samples=50,centers=centers,n_features=2)

colors = 10*["g","r","c","y"]

#first assign every signle feature set is a cluster center
# take all of the data points or featureset within that cluster center radius
# within the bandwidth
#take the mean of all the featuresets, that is your new cluster center
# repeat step 2 until you have convergence

class Mean_shift:
    # radius = none, have alot of steps in radius,
    def __init__(self, radius=None,radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step


    def fit (self,data):

        if self.radius==None:
            #find center of all data
            all_data_centroid = np.average(data,axis=0)
            #magnitude from origin
            all_data_norm = np.linalg.norm(all_data_centroid)
            #use magnitude to find decent avg
            self.radius= all_data_norm/self.radius_norm_step

        centroids = {} # empty dict

        #define weights
        weights =[i for i in range(self.radius_norm_step)][::-1]

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
                    #full distance
                    distance = np.linalg.norm(featureset-centroid)
                    #when feature set is comparing diatance to itself (first itr)
                    if distance ==0:
                        distance = 0.0000000001
                    
                    #entire dist/ radius
                    weight_index = int(distance/self.radius)
                    # if dist greater then max distance , we say its max
                    if weight_index> self.radius_norm_step-1:
                        weight_index = self.radius_norm_step-1
                    
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth+=to_add



                #recalculate means , gives means venctor
                new_centroid = np.average(in_bandwidth,axis=0)
                # add to newcentroids list
                new_centroids.append(tuple(new_centroid)) # coverting array to tuple

            #get unique elements from new centroids list
            uniques = sorted(list(set(new_centroids)))

            to_pop = []
            for i in uniques:
                # we're not inspecting centroids in radius of i since i will be popped
                if i in to_pop:
                    break
                for ii in uniques:
                    if i ==ii:
                        pass
                        # skipping already-added centroids
                    elif np.linalg.norm(np.array(i)-np.array(ii))<= self.radius and ii not in to_pop:
                        to_pop.append(ii)

            for i in to_pop:
                uniques.remove(i)
            



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

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i]=[]
        for featureset in data:
            distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #min index will be classification
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    
    def predict(self,data):
        distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
            #min index will be classification
        classification = distances.index(min(distances))
        return classification

clf= Mean_shift()
clf.fit(X)
centroids = clf.centroids


for classification in clf.classifications:
    color =colors[classification] # set color as idex
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="*",color= color,s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0],centroids[c][1],color='k',marker="*",s=150)

plt.show()

##Now we wnat to make radius dynamic 








