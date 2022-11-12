#!/usr/bin/env python
# coding: utf-8

# # Violin plots of least five mean - K562

# In[34]:


least_five_mean_names=['NSUN2_K562_IDR.csv',
'GNL3_K562_IDR.csv',
'UTP3_K562_IDR.csv',
'PUS1_K562_IDR.csv',
'SBDS_K562_IDR.csv']


# In[35]:


# creating a dataframe of least five mean
import os
import pyranges as pr
import pandas as pd
df_least_five_mean_names=pd.DataFrame()
os.chdir(r'D:\University of Freiburg\Semester-4\project 2\K562')
for i,files in enumerate(os.listdir(r"D:\University of Freiburg\Semester-4\project 2\K562")):
    if files in least_five_mean_names:
        path=pr.get_example_path(r"D:\University of Freiburg\Semester-4\project 2\K562/"+str(files))
        df=pd.read_csv(path)
        df_least_five_mean_names=pd.concat([df_least_five_mean_names,df])


# In[36]:


def number_remover(x):
    x=x.split("_")
    return "_".join(x[:3])
df_least_five_mean_names["Name"]=df_least_five_mean_names["Name"].apply(number_remover)


# In[37]:


print(df_least_five_mean_names["Name"].unique())


# In[38]:


df_least_five_mean_names["Site_length"]=df_least_five_mean_names["End"]-df_least_five_mean_names["Start"]


# In[76]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.violinplot(y=df_least_five_mean_names["Site_length"],x=df_least_five_mean_names["Name"])
plt.title("Violin plots of least five mean-K562")
plt.xlabel("Site")
plt.xticks(rotation=90)
plt.show()


# # Violin plots of top five mean - K562

# In[1]:


highest_five_mean=['ENCFF318RSO_HNRNPK_K562_IDR.csv',
'ENCFF897MSA_EIF4E_K562_IDR.csv',
'HNRNPK_K562_IDR.csv',
'ENCFF603WDI_MBNL1_K562_IDR.csv',
'FUS_K562_IDR.csv']


# In[25]:


import os
import pyranges as pr
import pandas as pd

df_highest_five_mean=pd.DataFrame()
os.chdir(r'D:\University of Freiburg\Semester-4\project 2\K562 - Copy_only_bed')
for i,files in enumerate(os.listdir(r"D:\University of Freiburg\Semester-4\project 2\K562 - Copy_only_bed")):
    if files in highest_five_mean:
        path=pr.get_example_path(r"D:\University of Freiburg\Semester-4\project 2\K562 - Copy_only_bed/"+str(files))
        df=pd.read_csv(path)
        df_highest_five_mean=pd.concat([df_highest_five_mean,df])


# In[27]:


def number_remover(x):
    x=x.split("_")
    return "_".join(x[:-1])
df_highest_five_mean["Name"]=df_highest_five_mean["Name"].apply(number_remover)


# In[28]:


print(df_highest_five_mean["Name"].unique())


# In[29]:


df_highest_five_mean["Site_length"]=df_highest_five_mean["End"]-df_highest_five_mean["Start"]


# In[30]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.violinplot(y=df_highest_five_mean["Site_length"],x=df_highest_five_mean["Name"])
plt.title("Violin plots of highest five mean-K562")
plt.xlabel("Site")
plt.xticks(rotation=90)
plt.show()


#  # Violin plots of top five mean - HepG2

# In[2]:


highest_five_mean_names=['EXOSC5_HepG2_IDR.csv',
'HNRNPK_HepG2_IDR.csv',
'NCBP2_HepG2_IDR.csv',
'PTBP1_HepG2_IDR.csv',
'TAF15_HepG2_IDR.csv']


# In[48]:


import os
import pyranges as pr
import pandas as pd
df_highest_five_mean_names=pd.DataFrame()
os.chdir(r'D:\University of Freiburg\Semester-4\project 2\HepG2(1)')
for i,files in enumerate(os.listdir(r"D:\University of Freiburg\Semester-4\project 2\HepG2(1)")):
    if files in highest_five_mean_names:
        path=pr.get_example_path(r"D:\University of Freiburg\Semester-4\project 2\HepG2(1)/"+str(files))
        df=pd.read_csv(path)
        df_highest_five_mean_names=pd.concat([df_highest_five_mean_names,df])


# In[50]:


def number_remover(x):
    x=x.split("_")
    return "_".join(x[:3])
df_highest_five_mean_names["Name"]=df_highest_five_mean_names["Name"].apply(number_remover)


# In[51]:


print(df_highest_five_mean_names["Name"].unique())


# In[52]:


df_highest_five_mean_names["Site_length"]=df_highest_five_mean_names["End"]-df_highest_five_mean_names["Start"]


# In[53]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.violinplot(y=df_highest_five_mean_names["Site_length"],x=df_highest_five_mean_names["Name"])
plt.title("Violin plots of highest five mean-HepG2")
plt.xlabel("Site")
plt.xticks(rotation=90)
plt.show()


# #  Violin plots of least five mean - HepG2

# In[3]:


least_five_mean=['NOLC1_HepG2_IDR.csv',
'DHX30_HepG2_IDR.csv',
'TBRG4_HepG2_IDR.csv',
'SMNDC1_HepG2_IDR.csv',
'DKC1_HepG2_IDR.csv']


# In[68]:


import os
import pyranges as pr
import pandas as pd

df_least_five_mean=pd.DataFrame()
os.chdir(r'D:\University of Freiburg\Semester-4\project 2\HepG2(1)')
for i,files in enumerate(os.listdir(r'D:\University of Freiburg\Semester-4\project 2\HepG2(1)')):
    if files in least_five_mean:
        path=pr.get_example_path(r'D:\University of Freiburg\Semester-4\project 2\HepG2(1)/'+str(files))
        df=pd.read_csv(path)
        df_least_five_mean=pd.concat([df_least_five_mean,df])


# In[70]:


def number_remover(x):
    x=x.split("_")
    return "_".join(x[:3])
df_least_five_mean["Name"]=df_least_five_mean["Name"].apply(number_remover)


# In[73]:


print(df_least_five_mean["Name"].unique())


# In[74]:


df_least_five_mean["Site_length"]=df_least_five_mean["End"]-df_least_five_mean["Start"]


# In[75]:


import seaborn as sns
from matplotlib import pyplot as plt

sns.violinplot(y=df_least_five_mean["Site_length"],x=df_least_five_mean["Name"])
plt.title("Violin plots of least five mean-HepG2")
plt.xlabel("Site")
plt.xticks(rotation=90)
plt.show()

