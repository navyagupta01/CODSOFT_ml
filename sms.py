#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk 
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# In[2]:


df = pd.read_csv("C:/games/codsoft/SMS detection/spam.csv", encoding='latin1')


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df = df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])


# In[14]:


sns.countplot(x = df["v1"], data = df)


# In[7]:


df['v1'] = df["v1"].map({'spam':1,'ham':0})


# In[8]:


stemmer = LancasterStemmer()
stop_words = set(stopwords.words("english"))

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z+]', ' ', text) 
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i)>2])
    text = re.sub("\s[\s]+", " ", text).strip()
    return text

df["Shorten_msg"] = df["v2"].apply(cleaning_data)


# In[10]:


df


# In[11]:


vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["Shorten_msg"])

y = df["v1"]


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.3, random_state=0)

LR_model=LogisticRegression()
LR_model.fit(X_train,y_train)

NB_model= MultinomialNB()
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

