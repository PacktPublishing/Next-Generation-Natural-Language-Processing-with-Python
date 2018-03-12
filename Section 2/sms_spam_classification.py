
# coding: utf-8

# ## Analysis of spam SMS messages (data from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/))

# In[2]:


import pandas as pd
import sklearn

get_ipython().magic('matplotlib inline')


# In[8]:


df=pd.read_csv('SMSSpamCollection',sep='\t',header=None,names=['class','text'])


# In[9]:


df.head()


# ## Split into test data and training data

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.25)


# ## Some Pre-processing

# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer


# In[11]:


count_vect = CountVectorizer()


# In[34]:


X_train_counts = count_vect.fit_transform(X_train)


# In[35]:


count_vect.vocabulary_.items()[0:3]


# In[36]:


len(count_vect.vocabulary_)


# In[56]:


lab_bin=LabelBinarizer()
y_train_bin=lab_bin.fit_transform(y_train)
y_test_bin=lab_bin.fit_transform(y_test)


# ## Train

# In[58]:


from sklearn.naive_bayes import MultinomialNB


# In[59]:


clf = MultinomialNB().fit(X_train_counts, y_train_bin)


# In[60]:


len(clf.coef_[0])


# In[40]:


import collections


# In[61]:


importanceCount=collections.Counter()


# In[62]:


for word,imp in zip(count_vect.vocabulary_.keys(),clf.coef_[0]):
    importanceCount[word]=imp


# In[81]:


importanceCount.most_common()[-10:]


# ## Now test

# In[64]:


X_test_counts = count_vect.transform(X_test)


# In[65]:


pred=clf.predict(X_test_counts)


# In[66]:


from sklearn.metrics import average_precision_score


# In[68]:


average_precision_score(y_test_bin,pred)


# ## Sanity check

# In[71]:


clf.predict(count_vect.transform(['win big on this offer']))


# In[72]:


clf.predict(count_vect.transform(['hi how are you? shall we meet up soon?']))

