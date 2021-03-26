#!/usr/bin/env python
# coding: utf-8

# # DATS 6103: Project 1
# # Xueqi Zhou

# In[1]:


# Analyzing the military spending of 10 nations during 2010-2017
#US 
# China
# Russia
# Germany
# UK
# France
# Italy
# Iran
# Israel
# Saudi Arabia


# In[2]:


import pandas as pd


# # 1. Data collection and pre-processing

# # 1.1 Data Sources

# In[3]:


# all the data comes from world bank
mil_spend = pd.read_csv("/Users/zhouxueqi/Desktop/Fall2020/fall2020/6103/project1/API_MS.MIL.XPND.GD.ZS_DS2_en_csv_v2_1345080/military_spending.csv", skiprows=4)
gdp = pd.read_csv("/Users/zhouxueqi/Desktop/Fall2020/fall2020/6103/project1/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_1345540/GDP.csv", skiprows=4)
pop = pd.read_csv("/Users/zhouxueqi/Desktop/Fall2020/fall2020/6103/project1/API_SP.POP.TOTL_DS2_en_csv_v2_1345178/population.csv", skiprows=4)


# In[4]:


print(mil_spend.head())
print(gdp.head())
print(pop.head())


# # 1.2 Select Partial Data

# In[5]:


# 10 countries I want to focus
# 8 years (2010-2017)
country = ['United States','China','Russian Federation','Germany','United Kingdom','France','Italy','Iran, Islamic Rep.','Israel','Saudi Arabia']
year = list(map(str,range(2010,2018)))
name_year = ['Country Name']+year
print(name_year)


# In[6]:


#subset the data accroding to country
mil_co = mil_spend[mil_spend['Country Name'].isin(country)]
gdp_co = gdp[gdp['Country Name'].isin(country)]
pop_co = pop[pop['Country Name'].isin(country)]


# In[7]:


print(mil_co)
print(gdp_co)
print(pop_co)


# In[8]:


#subset the data accroding to year
mil_8 = mil_co[name_year]
gdp_8 = gdp_co[name_year]
pop_8 = pop_co[name_year]


# In[9]:


print(mil_8)
print(gdp_8)
print(pop_8)


# # 1.3 Checking Missing Values

# In[10]:


# to check the NA value in the dataset
mil_8.isna().sum()


# In[11]:


gdp_8.isna().sum()


# In[12]:


pop_8.isna().sum()


# # 1.4 Merge Data Sets

# In[13]:


# rename the columns in order to merge two data sets
mil_8=mil_8.rename(columns={'Country Name':'name',
                      '2010':'2010_mil',
                     '2011':'2011_mil',
                     '2012':'2012_mil',
                     '2013':'2013_mil',
                     '2014':'2014_mil',
                     '2015':'2015_mil',
                     '2016':'2016_mil',
                     '2017':'2017_mil'})

gdp_8=gdp_8.rename(columns={'Country Name':'name',
                      '2010':'2010_gdp',
                     '2011':'2011_gdp',
                     '2012':'2012_gdp',
                     '2013':'2013_gdp',
                     '2014':'2014_gdp',
                     '2015':'2015_gdp',
                     '2016':'2016_gdp',
                     '2017':'2017_gdp'})

pop_8=pop_8.rename(columns={'Country Name':'name',
                      '2010':'2010_pop',
                     '2011':'2011_pop',
                     '2012':'2012_pop',
                     '2013':'2013_pop',
                     '2014':'2014_pop',
                     '2015':'2015_pop',
                     '2016':'2016_pop',
                     '2017':'2017_pop'})


# In[15]:


#merge three data sets
mil_gdp = pd.merge(mil_8,gdp_8)
mil_gdp = pd.merge(mil_gdp, pop_8)
print(mil_gdp)


# # 1.5 Calculate total military spending

# In[16]:


# calculate military spending according to the percentage
for i in range(0,8):
    new_co = mil_gdp.columns[1+i][:5]+'mil_value'
    mil_gdp[new_co] = mil_gdp.apply(lambda row: row[i+1]*row[i+9]/100, axis = 1)
print(mil_gdp)


# # 2. Analysis

# # 2.1 Compare the military data to that country’s GDP

# In[17]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# In[18]:


for i in range(0,10): # 10 countries
    x = np.array(mil_gdp.iloc[i,25:33]).reshape(-1, 1)
    y = np.array(mil_gdp.iloc[i,9:17]).reshape(-1, 1)
    # linear regression model
    regr = linear_model.LinearRegression()
    regr.fit(x.reshape(-1, 1), y) # .reshape(-1, 1)，since dim(X)=1
    
    # draw scatter and linear regression line
    plt.scatter(x,y)
    plt.plot(x, regr.predict(x.reshape(-1,1)), color='red', linewidth=1)

    # annotation on graph
    plt.xlabel('Military total spending')
    plt.ylabel('GDP')
    title = 'Military Spending to GDP for '+ mil_gdp.iloc[i,0]
    plt.title(title)
    plt.show()


# # 2.2 Compare the overall military spending of the all 10 countries in absolute and percentages

# In[19]:


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
for i in range(0,8): # 8 years
    labels = list(mil_gdp.iloc[:,0])
    spending = list(mil_gdp.iloc[:,25+i])

    fig = plt.figure(figsize =(10, 10)) 
    plt.tick_params(axis='x', labelsize=8)  
    plt.pie(spending,labels=labels, autopct='%1.1f%%')
    title = 'Overall Military Spending of the All 10 Countries in ' + mil_gdp.columns[i+1][:4]
    plt.title(title)
    plt.show()


# # 2.3 Compare the per person military spending to the per person GDP in absolute and percentages

# In[21]:


# create per person mil spending col and per person gdp col
for i in range(0,8):
    new_co1 = mil_gdp.columns[1+i][:5]+'per_mil'
    #total military/population
    mil_gdp[new_co1] = mil_gdp.apply(lambda row: row[i+25]/row[i+17], axis = 1)
for i in range(0,8):    
    #total gdp/population
    new_co2 = mil_gdp.columns[1+i][:5]+'per_gdp'
    mil_gdp[new_co2] = mil_gdp.apply(lambda row: row[i+9]/row[i+17], axis = 1)
for i in range(0,8):    
    #total military/gdp
    new_co3 = mil_gdp.columns[1+i][:5]+'per_mil_vs_per_gdp'
    mil_gdp[new_co3] = mil_gdp.apply(lambda row: row[i+33]/row[i+41]*100, axis = 1)
    print(mil_gdp)


# In[24]:


for i in range(0,8):
    labels = list(mil_gdp.iloc[:,0])
    spending = list(mil_gdp.iloc[:,49+i])
    
    fig = plt.figure(figsize=(18,4))     
    plt.bar(labels, spending, align='center', alpha=0.5, width=0.35)
    plt.xlabel("Country")
    plt.ylabel("Military/GDP")
    title = 'Per Person Military Spending to the Per Person GDP in ' + mil_gdp.columns[i+49][:4] + ' in Absolute and Percentage'
    plt.title(title)
    plt.show()


# # 2.4 Single out the fastest growing countries in military spending in absolute and percentage

# In[25]:


# firstly, I want to draw the military spending for each country during 2020-2017
for i in range(0,10): 
    Year = list(range(2010,2018))
    Military_Spending = list(mil_gdp.iloc[i,25:33])

    plt.plot(Year, Military_Spending, color='red', marker='o')
    name = list(mil_gdp.iloc[:,0])
    plt.title('Military_Spending Vs Year for '+ str(name[i]))
    plt.xlabel('Year')
    plt.ylabel('Military_Spending ')
    plt.show()


# In[26]:


# calculate growth rate = (present-past)/past

new_col = "growth_rate_10-17"
mil_gdp[new_col] = mil_gdp.apply(lambda row: abs((row[32]-row[25])/row[25]), axis = 1)
labels = list(mil_gdp.iloc[:,0])
growth_rate = list(mil_gdp.iloc[:,57])

# draw a bar graph to show the growth rate for each country
fig = plt.figure(figsize=(18,4))     
plt.bar(labels, growth_rate, align='center', alpha=0.5, width=0.35)
plt.xlabel("Country")
plt.ylabel("Growth rate")
plt.title("Growth Rate for Every Country between 2010 to 2017")

plt.show()
print(mil_gdp[new_col])


# In[ ]:




