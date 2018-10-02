## Merging and Joining Data frames
# one honors an index , other dont
import pandas as pd
# sample data sets
df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                    index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                    index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                    index = [2001, 2002, 2003, 2004])

#lets merge
# when we merge we want to say where we want to merge
# on column name
# we got duplicate colums
print(pd.merge(df1,df2, on='HPI'))
# we can merge on more than one column
print(pd.merge(df1,df2, on=['HPI','Int_rate']))

# we will use data to bring things together

# so join
#joining df 1 and df3 
# restting index of both
df1.set_index('HPI',inplace=True)
df3.set_index('HPI',inplace=True)
#lets join
joined = df1.join(df3)
#print(joined) #stil lhave redundent data

#let's consider joining and merging on slightly differing indexes

df6= pd.DataFrame({
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55],
                    'Year':[2001, 2002, 2003, 2004]
                    })

df7 = pd.DataFrame({
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53],
                    'Year':[2001, 2003, 2004, 2005]})


#lets try merge
merged = pd.merge(df6,df7, on='Year')
#set year as index
merged.set_index('Year',inplace=True)
print(merged) #missing 2005, 2002

# four choices to merge, right , left , inner outer , df6 is left, df7 is right (using how=) 
#outer is union of keys
#inner is default where key intersact
#try out in how
merged = pd.merge(df6,df7, on='Year',how='left')
merged.set_index('Year',inplace=True)
print(merged)

#we can check out joining, which will join on the index, so we can do something like this:
df6.set_index('Year', inplace=True)
df7.set_index('Year', inplace=True)
joined = df1.join(df3, how="outer")
print(joined)