#!/usr/bin/env python
# coding: utf-8

# # Chapter 5

# ## 5.1 Intro / Session Overview
# - What we will cover this week

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[2]:


csv_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(csv_url)


# In[3]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[4]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[5]:


y_test = df_test.churn.values


# In[6]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
     'gender',
     'seniorcitizen',
     'partner',
     'dependents',
     'phoneservice',
     'multiplelines',
     'internetservice',
     'onlinesecurity',
     'onlinebackup',
     'deviceprotection',
     'techsupport',
     'streamingtv',
     'streamingmovies',
     'contract',
     'paperlessbilling',
     'paymentmethod',
              ]


# In[7]:


def train(df_train, y_train, C = 1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C = C, max_iter = 1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[8]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


# In[9]:


C = 1.0
n_splits = 5


# In[10]:


kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C = C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[11]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# Save the model

# In[12]:


from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('chapter_overview5.png')
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()


# In[13]:


from PIL import Image
import matplotlib.pyplot as plt
img1 = Image.open('chapter_overview5_part2.png')
plt.figure(figsize=(10, 8))
plt.imshow(img1)
plt.axis('off')
plt.show()


# ## 5.2 Saving and loading the model
# 
# - Saving the model to pickle
# - Loading the model from pickle
# - Turning our notebook into a Python script

# In[14]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[15]:


csv_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(csv_url)


# In[16]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[17]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[18]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
     'gender',
     'seniorcitizen',
     'partner',
     'dependents',
     'phoneservice',
     'multiplelines',
     'internetservice',
     'onlinesecurity',
     'onlinebackup',
     'deviceprotection',
     'techsupport',
     'streamingtv',
     'streamingmovies',
     'contract',
     'paperlessbilling',
     'paymentmethod',
              ]


# In[19]:


def train(df_train, y_train, C = 1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C = C, max_iter = 1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[20]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred


# In[21]:


C = 1.0
n_splits = 5


# In[22]:


kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
    
scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C = C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[23]:


scores


# In[24]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
auc


# ### Save the model

# In[25]:


import pickle


# In[26]:


output_file = f'model_C={C}.bin'
output_file


# In[27]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)


# ### Load the model

# In[35]:


import pickle


# In[36]:


model_file = 'model_C=1.0.bin'


# In[37]:


with open(model_file, 'rb') as f_in:
    dv,model = pickle.load(f_in)


# In[38]:


dv,model


# In[39]:


customer = {
    'gender':'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[41]:


X = dv.transform([customer])


# In[43]:


model.predict_proba(X)[0, 1]


# In[ ]:




