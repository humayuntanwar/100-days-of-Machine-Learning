#IMPLEMENTATION OF K MEANS ALGORITHM FROM SCRATHCH

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
              [0, 3],
              [5, 4],
              [6, 6],])


#plt.scatter(X[:,0], X[:,1], s= 100)
plt.show()
colors = ["g","r","c","y"]


# defining our class

#tol = tolenreance, how much centroid gonna move
# max_iteration how many times we wana run
class K_Means:
    def __init__(self, k=2,tol=0.001,max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter= max_iter

    #
    def fit(self,data):
        self.centroids = {} #empty dict

        #populate centroid first two will be first two from x data = x
        for i in range(self.k):
            self.centroids[i] = data[i]

        #optimization process
        for i in range(self.max_iter):
            self.classifications = {} #empty dict

            for i in range(self.k):
                self.classifications[i] = [] #wmpty list
            
            #populating list
            for featureset in X:
                #first calculate distacnes, norm of featureset , creating a list , polulating with k num of value, zeroth index in the list will be ditnace to zeroth centroid
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances)) #
                self.classifications[classification].append(featureset)

            
            prev_centroids = dict(self.centroids) # compare two setroid

            for classification in self.classifications:
                pass
                #finding mean of all the feautres, and redefining all the centroids
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.00)>self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.00))
                    optimized = False

            if optimized:
                break






    
    # method for prediction

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances)) #
        return classification


clf = K_Means()
clf.fit(X)

#plotting centroids
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
    marker="o",color="k",s=150,linewidths=5)

#plotting classifications
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1], marker="x", color=color, s=150, linewidths=5)  
        

# unknowns = np.array([[1, 3],
#               [8, 9],
#               [0, 3],
#               [5, 4],
#               [6, 6],
#                 ])

# for unknown in unknowns:
#     classification = clf.predict(unknown)
#     plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)


plt.show()

##TESTING ON TITANIC DATA

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd 

'''
TITANIC DATA DESCRIPTION

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
df = pd.read_excel('datasets/titanic.xls') # load excel
#print(df.head()) # test print head

#converting  non numeric data to numeric
# lets try for sex first

df.drop(['body', 'name'],1,inplace=True) #dropping body id and name as they are not iportant
df.convert_objects(convert_numeric = True) # convert all coloums to numeric 
df.fillna(0,inplace=True)
print(df.head())

#handle non numericdata
def handle_non_numeric_data(df):
    coloumns = df.columns.values # all coloums

    for column in coloumns:
        text_digit_vals = {} # setting to empty dict for now

        def convert_to_int(val):
            return text_digit_vals[val] # for index of that val
        
        #check if datatype is neither int64 nor float 64
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents =df[column].values.tolist() # convert to list
            unique_elements = set(column_contents) # grabbing unique no reptative value
            x = 0 # interating
            # populating the dict with unique elements
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column])) # mapping ro the int

    return df

df = handle_non_numeric_data(df)
#print(df.head())

# lets drop ticket column
df.drop(['boat'],1,inplace=True) # tickets number matters to cant drop
#using k means on the data to see the chances of survival of people using the data already provided
X = np.array(df.drop(['survived'],1).astype(float)) # droping survived coloumn
#lets scale X now
X = preprocessing.scale(X)

Y = np.array(df['survived']) # our column
clf = KMeans() # kmeans
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me= np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct +=1

print(correct/len(X))  

#currently we are between 49 to 51 which is inconclusive
# after scaling X we jump to 75 to 28
#Clusters are assigned totally arbitararily 
#after dropping boat 70 to 30



