
# coding: utf-8

# ## Video 5.3

# In[1]:


import gensim,pickle
from gensim.parsing.preprocessing import preprocess_string
import pandas as pd
import json
import collections
import matplotlib.pyplot as plt
from utils import plotTopicProjections
get_ipython().magic('matplotlib inline')


# ### Define path to Reddit data

# In[19]:


pathToData='./RC_2010-01'


# ### Test gensim in-built text pre-processing

# In[20]:


preprocess_string('What is the point of learning NLP, do you think?')


# ## Define Text Generator

# In[21]:


class textGen():
    '''
    Object to iterate over text out of memory
    Generator: Yields values one at a time
    @n is number of lines to read, -1 means all lines
    '''
    def __init__(self,n=-1):
        print('Initialising textgenerator...')
        self.n=n

    def __iter__(self):    
        with open(pathToData,'r',errors='ignore') as inFile:
            for nLine,line in enumerate(inFile):
                
                if self.n>-1 and nLine>self.n:
                    break
                if len(line)>0:
                
                    if not len(line)==0:
                        yield preprocess_string(line)


# ### Create dictionary by looping over tokens in all documents to build vocabulary

# In[8]:


get_ipython().magic('time dictionary = gensim.corpora.Dictionary(textGen())')


# In[9]:


with open('dict.pkl','wb') as outFile:
    pickle.dump(dictionary,outFile)


# In[10]:


dictionary.filter_extremes()
# Drop terms that appear less than 5 times and more than 50% of documents
# Then filter to top 100k terms


# In[11]:


len(dictionary.keys())


# ## Define Corpus Object

# In[22]:


class redditCorpus():
    '''
    Class wrapper for reading Reddit dump
    Generator: Yields indexed documents one at a time
    @n is number of lines to read, -1 means all lines
    '''
    def __init__(self,n=-1):
        print('Initialising corpus...')
        self.n=n
        
    def __iter__(self):
        with open(pathToData,'r',errors='ignore') as inFile:
            for nLine,line in enumerate(inFile):
                
                
                
                if self.n>-1 and nLine>self.n:
                    break
                if len(line)>0:
                    
                    
                    d=json.loads(line)
                    tokens=preprocess_string(d['body'])
                    yield dictionary.doc2bow(tokens)


# ## LDA

# ### Compare regular and multicore run times

# In[29]:


get_ipython().magic('time resLda = gensim.models.ldamodel.LdaModel(redditCorpus(100000),num_topics=4,                                            id2word=dictionary)')
# Time test of single core implementation
# 2m.48


# In[36]:


get_ipython().magic('time resLda = gensim.models.ldamulticore.LdaMulticore(redditCorpus(100000),num_topics=4,                                            id2word=dictionary)')
# Time test of multi-core implementation
# 1m.53


# In[23]:


get_ipython().magic('time resLda = gensim.models.ldamulticore.LdaMulticore(redditCorpus(),num_topics=4,                                        id2word=dictionary)')


# In[24]:


with open('lda.pkl','wb') as outFile:
    pickle.dump(resLda,outFile)


# In[25]:


for n,t in resLda.show_topics():
    print(t)
    print('-----')


# In[26]:


plotTopicProjections(resLda,dictionary,scale=True)


# ### Each document has a projection on each topic

# In[27]:


resLda.get_document_topics(dictionary.doc2bow(            preprocess_string('I love this great video! Take a look')))


# ### Look at the maximum projection for each document to see degree of match
# Baseline is 0.25 for n=4 topics

# In[28]:


def plotProjections(model,n):
    '''
    Covenience function to visualise the dominant topic
    matches over first @n documents
    -1 means all documents
    '''
    maxProjections=map(lambda c: max([p[1] for p in model.get_document_topics(c)]),                    redditCorpus(n))
    plt.hist(list(maxProjections),bins=15)


# In[29]:


plotProjections(resLda,100000)


# ### TFIDF

# ### Take a second pass over corpus to calculate TFIDF scaling

# In[30]:


get_ipython().magic('time res_tfidf = gensim.models.ldamulticore.LdaMulticore(tfidf_corpus[redditCorpus()],num_topics=4,                                            id2word=dictionary)')


# In[383]:


for n,t in res_tfidf.show_topics():
    print(t)
    print('-----')


# In[385]:


plotTopicProjections(res_tfidf,scale=True)


# ## LSI

# In[386]:


get_ipython().magic('time res = gensim.models.lsimodel.LsiModel(redditCorpus(100000),num_topics=4,                                            id2word=dictionary)')


# In[388]:


plotTopicProjections(res)


# In[31]:


get_ipython().magic('time tfidf_corpus = gensim.models.TfidfModel(redditCorpus())')


# ### TFIDF Transformation

# In[389]:


get_ipython().magic('time resLsi = gensim.models.lsimodel.LsiModel(tfidf_corpus[redditCorpus()],num_topics=4,                                            id2word=dictionary)')


# In[390]:


for n,topic in res.show_topics():
    print(n)
    print(topic)


# In[391]:


plotTopicProjections(resLsi)


# In[393]:


projections=map(lambda c: max([p[1] for p in resLsi.get_document_topics(c)]),                    redditCorpus(10000))


# In[403]:


from sklearn.metrics import silhouette_score
import numpy as np


# In[490]:


c=redditCorpus(100000)


# In[491]:


projections=map(lambda c: np.argmax([p[1] for p in resLda.get_document_topics(c)]),c)


# In[1]:


termDocumentMatrix = gensim.matutils.corpus2dense(tfidf[c],num_terms=len(dictionary))
documentLabels = projections

