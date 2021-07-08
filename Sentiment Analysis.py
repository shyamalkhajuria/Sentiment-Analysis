#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import pandas as pd
df = pd.read_csv(r'C:\Users\shyam\reviews.csv')
print(df)


# In[4]:


import matplotlib.pyplot as plt
import numpy as np


# In[6]:


df.shape


# In[7]:


df.shape


# In[8]:


df_new = df.dropna()


# In[9]:


df_new.head()


# In[10]:


from textblob import TextBlob


# In[11]:


get_ipython().system('pip3 install textblob')


# In[12]:


from textblob import TextBlob


# In[13]:


test = TextBlob("The movie was awesome!")
print(test.sentiment)


# In[14]:


test_review = TextBlob("Clean and Hygienic Rooms ... excellent support by F&B Team.... Especially the service and care provided by Shebin from F& B Team")
print(test_review.sentiment)


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


df_new.head()


# In[16]:


pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df_new['polarity'] = df_new['Review'].apply(pol)
df_new['subjectivity'] = df_new['Review'].apply(sub)
df_new


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


df_new.plot(kind='scatter',x='polarity',y='subjectivity') # scatter plot


# In[19]:


df_new[0:582]


# In[ ]:




