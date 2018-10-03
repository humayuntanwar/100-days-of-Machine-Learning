## Pickling
import quandl 
import pandas as pd
import pickle
#get quandl data 
#key.yxt contains quandl auth key
api_key = open('key.txt','r').read()
#house price index for alaska
#df=quandl.get('FMAC/HPI_AK', authtoken=api_key)
#print(df.head())

#get names of all 50 states using read html

#function to get state list from htnl page
def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][1][1:]
# initial state
def grab_initial_state_data():
    states = state_list()
    # add new df
    main_df = pd.DataFrame() #empty
    # built query by state for quandl like "fmac/hpi_ar"
    for abbv in states:
        query = "FMAC/HPI_"+str(abbv)
        #print(query)
        #getting all data from quandl
        df = quandl.get(query,authtoken=api_key)
        #convert to perchat change
        #df = df.pct_change()
        
        df.columns = [str(abbv)]
        #using pct change formaula on the values
        # new -old / old *100
        df[abbv] = (df[abbv]-df[abbv][0]) / df[abbv][0] * 100.0        #give col names the abbv to replace value
        # if empty 
        if main_df.empty:
            main_df = df
        # if already has values
        else:
            main_df = main_df.join(df)

    print(main_df.head())
    # can save as csv and read csv
    # better option pickle
    #serializes and saves byte stream
    pickle_out = open('fifty_states3.pickle','wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()

#grab_initial_state_data()

#read pickle readbytes
# pickle_in = open('fifty_states.pickle','rb')
# HPI_data = pickle.load(pickle_in)
# print(HPI_data)

# #with pandas version, supposedly fast
# HPI_data.to_pickle('pickle.pickle')
# HPI_data2 = pd.read_pickle('pickle.pickle')
# print(HPI_data2)

##created house prices INDEX of all 50 states of USA dataset 


## bring in new data 
def HPI_Benchmark():
    df = df = quandl.get('FMAC/HPI_USA',authtoken=api_key)
    #remane coloum to united states
    df.columns = ['United States']
    #use the same formula as above to eget pct change
    df['United States'] = (df['United States']-df['United States'][0]) / df['United States'][0] * 100.0 
    return df


##percent change and correlation tables 
#read pickle
#grab_initial_state_data()
HPI_data = pd.read_pickle('fifty_states3.pickle')
benchmark = HPI_Benchmark()
#modifying columns
#HPI_data['TX2'] = HPI_data['TX'] *2
#print(HPI_data[['TX','TX2']]) 

#import matplot lib for graphs
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
fig = plt.figure()
#sub plot with 1x1 grid
ax1 = plt.subplot2grid((1,1), (0,0))
# plot entire data fram
HPI_data.plot(ax = ax1)
benchmark.plot(ax=ax1,color='k',linewidth=10)
plt.legend().remove()
plt.show()

#measure in percent change

#corelation 
#see correlation of all states all on left to right mapped
HPI_State_Correlation = HPI_data.corr()
print(HPI_State_Correlation)

#gives a descriptions , since is a dataframe
print(HPI_State_Correlation.describe())


