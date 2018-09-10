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
print(df.head())




