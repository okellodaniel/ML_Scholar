#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


df = pd.read_csv('data_week_3.csv')

df.columns = df.columns.str.lower().str.replace(' ','_')

categorical_columns =list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[43]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[46]:


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


# In[48]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C,solver='liblinear',max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[50]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[52]:


C = 1.0
n_splits = 5


# In[54]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[56]:


scores


# In[58]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# ## Save Model

# In[23]:


import pickle


# In[24]:


output_file = f'model_C={C}.bin'
output_file


# In[25]:


with open(model_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)


# ## load model
# - for better results, restart kernel at this section

# In[12]:


import pickle


# In[14]:


model_file_name = 'model_C=1.0.bin'


# In[16]:


with open(model_file_name,'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[18]:


dv


# In[20]:


model


# In[78]:


import random


# In[80]:


customer = df_full_train.iloc[random.randint(0,100)].to_dict()
customer


# In[84]:


X_customer = dv.transform([customer])


# In[88]:


model.predict_proba(X_customer)[0,1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




