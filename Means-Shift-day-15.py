import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import MeanShift
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

# copying for meansshift
original_df = pd.DataFrame.copy(df)

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
clf = MeanShift() # meansshift

clf.fit(X)

# gives labels
labels = clf.labels_
cluster_centers = clf.cluster_centers_ #gives clusters center
 
original_df['cluster_group'] = np.nan #adding new column to original dataset
# iterate thorugh the labels and poluate the value ofthis coloums

for i in range(len(X)):
    #iloc reference indexs of df, 
    original_df['cluster_group'].iloc[i] = labels[i]
n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group']==float(i))]
    survival_cluster = temp_df[ (temp_df['survived']==1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
#Survival rates {0: 0.3836378077839555, 1: 0.0, 2: 0.8421052631578947, 3: 0.1}

print(original_df[ (original_df['cluster_group']==1) ]) # group 1

#Let's look into group 0, which seemed a bit more diverse. This time, we will use the .describe()
print(original_df[ (original_df['cluster_group']==0) ].describe())

#Let's check the final group, 2, which we are expected to all be 3rd class:
print(original_df[ (original_df['cluster_group']==2) ].describe())

#what is the survival rate of the 1st class passengers in cluster 0, compared to the overall survival rate of cluster 0?

cluster_0 = (original_df[ (original_df['cluster_group']==0) ])
cluster_0_fc = (cluster_0[ (cluster_0['pclass']==1) ])
print(cluster_0_fc.describe())
