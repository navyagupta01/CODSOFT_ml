#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from collections import Counter
import datetime
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,roc_curve,classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# In[2]:


df_train=pd.read_csv("C:/games/codsoft/Credit Card fraud/fraudTrain.csv")
df_test=pd.read_csv("C:/games/codsoft/Credit Card fraud/fraudTest.csv")


# In[3]:


df_train.info()


# In[4]:


df_train.describe()


# In[5]:


df_train.columns


# In[6]:


df_train.notnull().sum()


# In[7]:


df_train["trans_date_trans_time"] = pd.to_datetime(df_train["trans_date_trans_time"])
df_train["dob"] = pd.to_datetime(df_train["dob"])

df_test["trans_date_trans_time"] = pd.to_datetime(df_test["trans_date_trans_time"])
df_test["dob"] = pd.to_datetime(df_test["dob"])


# In[8]:


labels=["Actual","Fraud"]

fraud_count = df_train["is_fraud"].value_counts().tolist()
values = [fraud_count[0], fraud_count[1]]

fig = px.pie(values=df_train['is_fraud'].value_counts(), names=labels, color_discrete_sequence=["green","red"]
             ,title="Fraud vs Genuine transactions")
fig.show()


# In[9]:


cols = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
        'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
encoder = OrdinalEncoder()
df_train[cols] = encoder.fit_transform(df_train[cols])

cols = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last',
        'gender', 'street', 'city', 'state', 'job', 'dob', 'trans_num']
encoder = OrdinalEncoder()
df_test[cols] = encoder.fit_transform(df_test[cols])


# In[10]:


x=df_train.drop(['is_fraud'],axis=1)
y=df_train['is_fraud']


# In[11]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[12]:


LR_model=LogisticRegression()
LR_model.fit(X_train,y_train)

NB_model=GaussianNB()
NB_model.fit(X_train,y_train)


# In[13]:


LR_pred = LR_model.predict(X_test)

accuracy = accuracy_score(y_test, LR_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, LR_pred))

NB_pred = NB_model.predict(X_test)

accuracy = accuracy_score(y_test, NB_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, NB_pred))


# In[14]:


df_test


# In[15]:


X_test=df_test.drop(['is_fraud'],axis=1)
y_test=df_test['is_fraud']

X_train,X_test,y_train,y_test=train_test_split(X_test,y_test,test_size=0.2,random_state=42)

LR_pred = LR_model.predict(X_test)

accuracy = accuracy_score(y_test, LR_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, LR_pred))

NB_pred = NB_model.predict(X_test)

accuracy = accuracy_score(y_test, NB_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, NB_pred))

