#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

df=pd.read_csv("metadata_processed_files_hg38.tsv",sep='\t')
df.head()


# In[4]:


df=df.loc[:,["File accession","File format","File assembly","Biosample term name","Experiment target","Biological replicate(s)","md5sum","File download URL"]]


# In[5]:


df=df[(df["File format"]=="bed narrowPeak")&(df["Biological replicate(s)"]=="1, 2")&(df["File assembly"]=="GRCh38")&(df["Biosample term name"].isin(["HepG2","K562"]))]
                                                                                                                


# In[6]:


df.to_csv("metadata_processed_files_hg38_version_1.tsv")
    


# In[7]:


df.shape

