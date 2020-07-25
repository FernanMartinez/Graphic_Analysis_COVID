#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as numpy
import pandas as pd


# In[3]:


DEAD_BY_COV = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
CASES_COV = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'


# In[16]:



# def data_preparation(url):
    url = CASES_COV
    data_df = pd.read_csv(url)
    colums_exclud = data_df.columns[[0, 2, 3]]
    data_df.drop(colums_exclud,
                 axis=1,
                 inplace=True)
    data_df = data_df.set_index("Country/Region")
    data_df = data_df.groupby(level=0).sum()

# data_preparation(url=CASES_COV)
data_df


# In[5]:


cases_df


# In[ ]:


cases_df.columns[[0, 2, 3]]


# In[ ]:


cases_df


# In[ ]:


cases_df.groupby(level=0).sum()


# In[ ]:


type(pd.Series(cases_df.index))


# In[ ]:


death_df.values[:, 0]
death_df.set_index(death_df.values[:, 0], inplace=True)
death_df


# In[ ]:


cases_df.values


# In[ ]:


cases_df = pd.read_csv(CASES_COV)


# In[ ]:


cases_df.loc[:, "Country/Region"].values


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




