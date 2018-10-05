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
HPI_data = pd.read_pickle('fifty_states.pickle')
#benchmark = HPI_Benchmark()
#modifying columns
#HPI_data['TX2'] = HPI_data['TX'] *2
#print(HPI_data[['TX','TX2']]) 

#import matplot lib for graphs
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
fig = plt.figure()
#sub plot with 1x1 grid
ax1 = plt.subplot2grid((2,1), (0,0))
# plot entire data fram
#HPI_data.plot(ax = ax1)
#benchmark.plot(ax=ax1,color='k',linewidth=10)
#plt.legend().remove()
#plt.show()

#measure in percent change

#corelation 
#see correlation of all states all on left to right mapped
#HPI_State_Correlation = HPI_data.corr()
#print(HPI_State_Correlation)

#gives a descriptions , since is a dataframe
#print(HPI_State_Correlation.describe())

### DAY 37
## RESAMPLING
'''
## change the sample weight of the data we are looking at
# define a new column, put the value of resampling as H=hourly, M=monthly and so on, A= annual
# resampling by mean, how =ohlc open high low close
TX1yr = HPI_data['TX'].resample('A',how='mean') #resampled by year
#lets print head
print(TX1yr.head())


#ploting taxes state , add label
HPI_data['TX'].plot(ax = ax1,  label='MOnthly TX HPI')
#plot
TX1yr.plot(ax=ax1, label='Yearly TX HPI')
plt.legend().remove()
plt.show()
'''
### HANDLING MISSING DATA
'''
we have 4 options
1 ignore
2 delete it
3 fill missing data(previous, future copy it)
4 replace it with static data or other

'''
# delete it 
#HPI_data['TX1yr'] = HPI_data['TX'].resample('A',how='mean')
#print(HPI_data[['TX','TX1yr']].head())

# drop any existance of NaN

#dropna = how=all drop all, any, passes a thrashhold
#HPI_data.dropna(how='all',inplace=True)

#HPI_data.dropna(inplace=True)

# fill na , fill forward
#HPI_data.fillna(method='ffill',inplace=True)

# fill na , fill backwards
#HPI_data.fillna(method='bfill',inplace=True)

# will with static data
#HPI_data.fillna(value=-99999,inplace=True)

#fillna limit
#HPI_data.fillna(value=-99999,limit=10,inplace=True)
#check how many nan remaining
#print(HPI_data.isnull().values.sum())
'''#print(HPI_data[['TX','TX1yr']].head())


HPI_data[['TX','TX1yr']].plot(ax=ax1)
plt.legend(loc=4)
plt.show()
'''
###ROLLING STATISTICS
#take a window of time and do many functions like , sum , mean, max
#rooling apply > make your function apply on your rolling data

# lets calc rolling mean, howmuch time ,12 months 

ax2 = plt.subplot2grid((2,1), (1,0),sharex=ax1)

#12 month for fitting avg
HPI_data['TX12MA'] = HPI_data['TX'].rolling(12).mean()

#lets do # std deviation 12 months
HPI_data['TX12MASTD'] = HPI_data['TX'].rolling(12).std()

#print(HPI_data[['TX','TX12MA']].head())
#if we look at the plot there is significent gap in starting to handle this we will use drop na
#HPI_data.dropna(inplace=True)
#HPI_data[['TX','TX12MA']].plot(ax=ax1)
#HPI_data['TX12MASTD'].plot(ax=ax2)


#correlations
# TX_AK_12corr = HPI_data['TX'].rolling(12).corr(HPI_data['AK'])
# HPI_data['TX'].plot(ax=ax1,label='TX HPI')
# HPI_data['AK'].plot(ax=ax1,label='TX AK')
# ax1.legend(loc=4)
# TX_AK_12corr.plot(ax=ax2, label='TX_AK_12corr')

# #plot
# plt.legend(loc=4)
# plt.show()



### DAY 38
## JOINING 30 YEARS OF DATA 
def mortgage_30y():
    df = df = quandl.get('FMAC/MORTG',trim_start='1975-01-01',authtoken=api_key)
    #use the same formula as above to eget pct change
    df['Value'] = (df['Value']-df['Value'][0]) / df['Value'][0] * 100.0 
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df



#print(state_HPI_M30.corr()['M30'].describe())


## ADDING OTHER ECONOMIC INDICATORS
'''
There are two major economic indicators that come to mind out the gate: 
S&P 500 index (stock market) and GDP (Gross Domestic Product). 
I suspect the S&P 500 to be more correlated than the GDP,
but the GDP is usually a better overall economic indicator, so I may be wrong.

'''
def sp500_data():
    df = quandl.get("MULTPL/SP500_REAL_PRICE_MONTH", trim_start="1975-01-01", authtoken=api_key)
    df.columns = ['Adjusted Close']

    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = quandl.get("EIA/STEO_XRUNR_M", trim_start="1975-01-01", authtoken=api_key)
    df.columns = ['Unemployment Rate']
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df

m30 = mortgage_30y()
HPI_bench = HPI_Benchmark()
#print(df.head())
state_HPI_M30 = HPI_data.join(m30)
m30.columns=['M30']
sp500 = sp500_data()
gdp = gdp_data()
unemployment = us_unemployment()
HPI = HPI_data.join([HPI_bench,m30,sp500,gdp,unemployment])
HPI.dropna(inplace=True)
print(HPI.corr())
HPI.to_pickle('HPI.pickle')