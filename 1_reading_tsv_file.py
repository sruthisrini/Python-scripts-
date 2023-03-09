import pandas as pd
import numpy as np

df=pd.read_csv("metadata_processed_files_hg38.tsv",sep='\t')
df.head()

df=df.loc[:,["File accession","File format","File assembly","Biosample term name","Experiment target","Biological replicate(s)","md5sum","File download URL"]]

df=df[(df["File format"]=="bed narrowPeak")&(df["Biological replicate(s)"]=="1, 2")&(df["File assembly"]=="GRCh38")&(df["Biosample term name"].isin(["HepG2","K562"]))]
                                                                                                                
df.to_csv("metadata_processed_files_hg38_version_1.tsv")

df.shape
