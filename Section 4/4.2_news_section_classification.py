
# coding: utf-8

# ## Financial and technology articles taken from [webhose.io](https://webhose.io/datasets)

# In[1]:


import pandas as pd
import json
import glob
get_ipython().magic('matplotlib inline')


# ## Take a look at one JSON file

# In[2]:


with open('financial_news/data/09/news_0000001.json','r') as inFile:
    d=json.loads(inFile.read())


# In[3]:


print(d.keys())


# In[4]:


print(d['text'])


# ## Define a function to open a file and get the text

# In[5]:


def getText(f):
    with open(f,'r') as inFile:
        d=json.loads(inFile.read())
    return d['text']


# In[6]:


get_ipython().magic("time financeTexts=list(map(getText,glob.glob('financial_news/data/[0-9][0-9]/news_*json')))")


# In[7]:


len(financeTexts)


# In[8]:


get_ipython().magic("time techTexts=list(map(getText,glob.glob('tech_news/data/[0-9][0-9]/news*json')))")


# In[9]:


len(techTexts)


# ## Combine tech and financial news into one dataframe

# In[10]:


df=pd.DataFrame(data={'text':financeTexts,'category':'finance'})


# In[11]:


df=df.append(pd.DataFrame(data={'text':techTexts,'category':'tech'}))


# In[12]:


df.head()


# In[13]:


df.shape


# ## Build up a pipeline

# In[14]:


from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn import preprocessing


# ## Binarise the category labels

# In[15]:


lb = preprocessing.LabelBinarizer()


# In[16]:


lb.fit(df['category'])
df['category_bin']=lb.transform(df['category'])


# ## Test Naive Bayes Classifier for our baseline

# In[17]:


steps=[('vectorise',CountVectorizer()),       ('transform',TfidfTransformer()),       ('clf',MultinomialNB())]
# Our pipeline has three steps


# In[18]:


pipe=Pipeline(steps)


# In[19]:


X_train, X_test, y_train, y_test=train_test_split(df['text'],df['category_bin'],test_size=0.25)


# In[20]:


pipe.fit(X_train,y_train)


# In[21]:


pred=pipe.predict(X_test)


# In[22]:


print('Accuracy = {:.3f}'.format(f1_score(y_test,pred)))


# ## Write out model

# In[23]:


import pickle
pickle.dump(pipe,open('model.out','wb'))


# ## Video 4.3

# ## Grid Search

# In[24]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# In[25]:


param_grid = dict(vectorise__min_df=[1,5,10])

#vectorise__stop_words=[None,'english'],
#vectorise__binary=[True,False],
#clf__class_weight=[None,'balanced'],
#transform__norm=['l1','l2']


# In[26]:


grid_search = GridSearchCV(pipe, param_grid=param_grid,                           scoring=make_scorer(f1_score),n_jobs=2)
# Can set n_jobs to -1 for all processors


# In[27]:


res=grid_search.fit(df['text'],df['category_bin'])


# In[28]:


res.best_params_


# In[29]:


print('Best score = {:.3f}'.format(res.best_score_))

