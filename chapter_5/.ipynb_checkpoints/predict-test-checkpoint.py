#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().run_line_magic('autosave', '0')


# In[28]:


import requests


# In[29]:


url = 'http://localhost:9696/predict'


# In[30]:


customer = {
    "gender":"female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


# In[36]:


response = requests.post(url, json = customer).json()
response


# In[37]:


if response['churn'] == True:
    print('sending promo email to %s' % 'xyz-123')


# In[ ]:




