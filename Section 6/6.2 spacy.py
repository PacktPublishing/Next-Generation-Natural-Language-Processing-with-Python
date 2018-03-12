
# coding: utf-8

# In[8]:


import spacy
from spacy import displacy


# In[4]:


get_ipython().system('python -m spacy download en')


# In[9]:


nlp = spacy.load('en')


# ## Break down a document

# In[10]:


doc = nlp(u'This is a sentence.')


# In[11]:


for token in doc:
    print('{:s} : {:s}'.format(token.text,token.pos_))


# ## Visualise the parse tree

# In[12]:


displacy.serve(doc)


# ## Extract entities

# In[14]:


for ent in nlp('I love the Beatles').ents:
    print(ent)


# ## Strip adjectives from nouns

# In[15]:


for ingredient in ['fried egg','boiled egg','pickled egg']:
    for token in nlp(ingredient):
        print('{:s} : {:s}'.format(token.text,token.pos_))
    print('-------')

