
# coding: utf-8

# ## Video 6.1

# In[8]:


import gensim
import nltk
import numpy as np
from sklearn.decomposition import PCA
from nltk.corpus import  brown,gutenberg
import nltk.corpus
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# ## Start with Brown corpus from NLTK

# In[2]:


nltk.download('brown')
# Grab Brown corpus from NLTK if not already downloaded


# In[9]:


len(nltk.corpus.brown.sents())


# In[10]:


sentences = brown.sents()


# In[11]:


sentences[0]


# ## Create a word2vec vector space model

# In[12]:


get_ipython().magic('time model = gensim.models.word2vec.Word2Vec(sentences,     size=100, window=5, min_count=5, workers=4, hs=1,negative=0)')


# In[13]:


model.most_similar('mother')


# In[14]:


model.doesnt_match(['king','queen','prince','Edward'])


# ## Use Pretrained Google News Vectors

# ### Grab from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

# In[64]:


get_ipython().magic("time model = gensim.models.KeyedVectors.load_word2vec_format(                        './GoogleNews-vectors-negative300.bin', binary=True)")


# ## Visualise

# In[65]:


X = model[model.wv.vocab]


# In[66]:


pca = PCA(n_components=2)
result = pca.fit_transform(X)


# In[91]:


words = list(model.wv.vocab)

colours=['black','red','blue','grey','green']

for n,wordList in enumerate([['paris','london','berlin'],['near','close','similar'],        ['terrible','awful','poor'],['carrot','apple','turnip'],['car','bicycle']]):
    for nn,word in enumerate(wordList):
        i=words.index(word)
        plt.annotate(word, xy=(result[i, 0], result[i, 1]),color=colours[n])
    
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()

