import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot') # type of graph

#dataframe is like a python dictionary

#a dataframe
web_stats = {'day':[1,2,3,4,5],
            'visitors':[43,43,56,67,30],
            'bounce_rate':[56,34,25,45,40]}

#Ecovert to dataframe 
df = pd.DataFrame(web_stats)

#to see
print(df)
#index if not specified will be generated automatically

#print first 5 rows
print(df.head())
#print last 5
print(df.tail())
#pass any val lie 2 to restrict num of vals

#set an index to how we want to visualize related
#returns a new modified dataframe
print(df.set_index('day')) # now day column will be treated as index

#after bunch of calcs
print(df.head())

#make a new df out of it
df2 = df.set_index('day')
print(df2.head())

#another way to do it
#will modify the dataframe then there and that
df.set_index('day',inplace=True)
print(df.head())

#to reference a specific coloums
print(df['visitors'])

#another way to reference specific columns
print(df.visitors)

#refrence multiple columns
print(df[['bounce_rate','visitors']])

#convert into  a list
print(df.visitors.tolist())

#no arrays in python
#use numpy
print(np.array(df[['bounce_rate','visitors']]))

#can you this
df2 = pd.DataFrame(np.array(df[['bounce_rate','visitors']]))
print(df2)