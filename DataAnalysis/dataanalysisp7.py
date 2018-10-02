## Pickling
import quandl 
import pandas as pd
import pickle
#get quandl data 
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
        #give col names the abbv to replace value
        df.columns = [str(abbv)]
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
    pickle_out = open('fifty_states.pickle','wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()

#grab_initial_state_data()

#read pickle readbytes
pickle_in = open('fifty_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
print(HPI_data)

#with pandas version, supposedly fast
HPI_data.to_pickle('pickle.pickle')
HPI_data2 = pd.read_pickle('pickle.pickle')
print(HPI_data2)

##created house prices INDEX of all 50 states of USA dataset 