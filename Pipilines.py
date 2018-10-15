import random
'''
got 0.33 using random because there are 3 types of flowers

'''
from scipy.spatial import distance

#euclidean distance
def eau(a,b):
    return distance.euclidean(a,b)

'''
our own classifier accuracy = 0.96
'''

class ScrappyKNN():
    #fit function
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X_test):
        #returns an 2d array
        predictions = []
        for row in X_test:
            #using random
            #label = random.choice(self.y_train)
            #using closest
            label = self.closest(row)
            predictions.append(label)
        return predictions
        #using auclidean distance 
    def closest(self, row):
        best_dist = eau(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = eau(row,self.X_train[i])
            if dist <best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

#import dataset
from sklearn import datasets
iris = datasets.load_iris()
#labels
X = iris.data
y = iris.target
#train and test split
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.5)

# bring in ml algorithm
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#our own classifier
my_classifier = ScrappyKNN()
#fit predict
my_classifier.fit(X_train,y_train)
predictions = my_classifier.predict(X_test)
print(predictions)
#check
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))