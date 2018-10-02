#Concatination and Appending
import pandas as pd
#build data frames 
# 1 nad 3 have same index but diffrent columns
#2 .3 have diffrent index and columns
#1,2 have indentical columns
df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                    index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                    index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                    index = [2001, 2002, 2003, 2004])

#concentanation
#concat = pd.concat([df1,df2])
#print(concat)
#adding columns each coloum hpi for each state
#all three
#too much redundency and nans
#concat2 = pd.concat([df1,df2,df3])
#print(concat2)
#now using append adds at end
# not efficient
df4 = df1.append(df2) 
#print(df4)
# we scould add a series
#vals and col name
s = pd.Series([80,2,50],index=['HPI','Int_rate','US_GDP_Thousands'])
df5 = df1.append(s, ignore_index=True)
print(df5)
