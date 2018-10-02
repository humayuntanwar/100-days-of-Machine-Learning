#building our dataset for real state analysis
import quandl 
import pandas as pd
#get quandl data 
api_key = open('key.txt','r').read()

#house price index for alaska
df=quandl.get('FMAC/HPI_AK', authtoken=api_key)
print(df.head())

#get names of all 50 states using read html
fifty_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
#will be alist of dataframes
print(fifty_states)
#we want values from index 1 (this is a dtaframe
print(fifty_states[0])
# now weant want coloumn 1
print(fifty_states[0][1])
# we want want element 1 onward
# built query by state for quandl like "fmac/hpi_ar"
for abbv in fifty_states[0][1][1:]:
    print("FMAC/HPI_"+str(abbv))