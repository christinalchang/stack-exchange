#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import re
import nltk
import nltk.corpus
import numpy as np
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
plt.rcParams["figure.figsize"] = [10, 8]


# In[65]:


def tokenize_text(doc):
    """Combine the strings in the "response" column of dataframe df into one long string. Then, tokenize the
    string and make all words lowercase."""

    # Tokenize and make lowercase.
    words = nltk.word_tokenize(doc)
    words = [w.lower() for w in words]
    
    return words


def wordnet_pos(tag):
    """Map a Brown POS tag to a WordNet POS tag."""
    
    table = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
    
    # Default to a noun.
    return table.get(tag[0], wordnet.NOUN)


def lemmatize_text(words):
    """Lemmatize words to get the base words. The input 'words' is a list of of words."""
    
    lemmatizer = nltk.WordNetLemmatizer()
    word_tags = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(w, wordnet_pos(t)) for (w, t) in word_tags]
    
    return words


def remove_stopwords(words):
    """Remove stopwords from a string."""
    
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w not in stopwords]
    
    return words

def clean_text(doc):    
    """Tokenize, lemmatize, and remove stopwords for the text of all articles."""
    
    words = re.sub("< ?/?[a-z]+ ?>|\n", "", doc)
    words = tokenize_text(words)
    words = lemmatize_text(words)
    words = remove_stopwords(words)
    doc = [w for w in words if w.isalnum()]
    doc = ' '.join(doc)
    
    return doc

def clean_df(df):
    """Combine the title and content of each post into one string and clean each string."""
    text = df['title'] + " " + df['content']
    df_clean = pd.DataFrame([clean_text(i) for i in text])
    df_clean.columns = ["text"]
    #df_clean["tags"] = df["tags"]
    df_clean = pd.concat([df_clean, pd.DataFrame(df["tags"])],axis = 1, sort = False)
    return df_clean


# In[51]:


## Functions from examine_tags.ipynb
def get_top_tags(df):
    tag_list = [tags.split(' ') for tags in df['tags']]
    flat_list = [item for sublist in tag_list for item in sublist]
    fq = nltk.FreqDist(w for w in flat_list)
    df_fq = pd.DataFrame.from_dict(fq, orient="index").reset_index()
    df_95 = df_fq[df_fq.iloc[:,1] >= np.percentile(df_fq.iloc[:,1],95)].reset_index(drop=True)
    df_95.columns = ["term", "fq"]
    return df_95

def subset_top_df(df, top_tags_df):
    """
    df: DataFrame with all posts
    top_tags_df: Data frame of top tags
    """
    tags_list = [tags.split(' ') for tags in df['tags']]
    indeces = [i for i in range(len(tags_list))
               if list(set(top_tags_df['term']) & 
                       set(tags_list[i])) != []]
    return df.loc[indeces]


# In[82]:


def trim_tags_clean(df):
    top = get_top_tags(df)
    top_subset = subset_top_df(df, top).reset_index()
    return clean_df(top_subset)


# In[ ]:


# Stack exchange topic names
names = ["biology","cooking","crypto","diy","robotics","travel"]

def get_paths(name):
    """Get path names for each file."""
    path = "data/"+name+".csv"
    return path

# Get path names
paths = [get_paths(i) for i in names]

# All data frames in a list.
dfs = [pd.read_csv(i) for i in paths]

# Get a list of the cleaned data frames.
trim_clean_dfs = [trim_tags_clean(i) for i in dfs]

# Save cleaned dfs as csv
for i in range(len(names)):
    trim_clean_dfs[i].to_pickle(names[i]+"_trim_clean.csv")


# In[ ]:




