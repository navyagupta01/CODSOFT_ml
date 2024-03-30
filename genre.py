#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import re  
import string
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,roc_curve,classification_report


# In[3]:


train_df=pd.read_csv("C:/games/codsoft/Movie genre/Genre Classification Dataset/train_data.txt", sep=":::", names=["Name", "Genre","Synopsis"],engine="python")


# In[4]:


train_df


# In[5]:


train_df.info()
train_df.describe()


# In[6]:


train_df.isnull().sum()


# In[7]:


test_df=pd.read_csv("C:/games/codsoft/Movie genre/Genre Classification Dataset/test_data.txt", sep=":::", names=["Name","Synopsis"],engine="python")
test_df


# In[8]:


test_solution_df=pd.read_csv("C:/games/codsoft/Movie genre/Genre Classification Dataset/test_data_solution.txt", sep=":::", names=["Name","Genre","Synopsis"],engine="python")
test_solution_df


# In[9]:


test_solution_df.drop(test_solution_df.columns[[0,2]],axis=1,inplace=True)
test_solution_df


# In[10]:


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

train_df["Shorten Synopsis"] = train_df["Synopsis"].apply(cleaning_data)
test_df["Shorten Synopsis"] = test_df["Synopsis"].apply(cleaning_data)
train_df
test_df


# In[11]:


plt.figure(figsize=(12, 15))
sns.countplot(data=train_df, y="Genre", order=train_df["Genre"].value_counts().index,palette = sns.color_palette("dark"))
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=14)
plt.show()


# In[12]:


train_df["Actual_Length"]=train_df['Synopsis'].apply(len)
train_df["Shorten_Length"]=train_df['Shorten Synopsis'].apply(len)
plt.figure()
plt.hist(train_df["Actual_Length"], bins=20, color="yellow", alpha=0.5, label="Actual Length")
plt.hist(train_df["Shorten_Length"], bins=20, color="Green", alpha=0.5, label="Shorten Length")
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


vectorize = TfidfVectorizer()

X_train_vector = vectorize.fit_transform(train_df["Shorten Synopsis"])


# In[14]:


X = X_train_vector
y = train_df['Genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


logreg = LogisticRegression(multi_class='multinomial', solver='sag')
logreg.fit(X_train, y_train)


# In[16]:


y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# In[17]:


nb = MultinomialNB()
nb.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = nb.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# In[18]:


test_df_vector = vectorize.transform(test_df["Shorten Synopsis"])


# In[19]:


y_pred = logreg.predict(test_df_vector)

accuracy = accuracy_score(test_solution_df, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(test_solution_df, y_pred))


# In[20]:


y_pred = nb.predict(test_df_vector)

accuracy = accuracy_score(test_solution_df, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(test_solution_df, y_pred))

