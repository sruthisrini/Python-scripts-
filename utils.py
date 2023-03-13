"""
Reading the metadata tsv file and filtering out the required columns
"""
import pandas as pd
import numpy as np

df=pd.read_csv("metadata_processed_files_hg38.tsv",sep='\t')
df=df.loc[:,["File accession","File format","File assembly","Biosample term name","Experiment target",
            "Biological replicate(s)","md5sum","File download URL"]]

df=df[(df["File format"]=="bed narrowPeak")&(df["Biological replicate(s)"]=="1, 2")&(df["File assembly"]=="GRCh38")&
        (df["Biosample term name"].isin(["HepG2","K562"]))]
                                                                                                                
##############################################################################################################################

"""
After finding overlap among the bed files using 
'$ bedtools intersect -a $(bed_file_1.bed) -b $(bed_file_2.bed) $(bed_file_3.bed) $(bed_file_4.bed)', 
store it in a text file
"""
import pandas as pd
data=pd.read_csv('file_path',sep='\t',header=None)

##############################################################################################################################

""" 
Creating a dataframe with 6 columns such as Chromosome id, Start, End, Strand, Name and Score.
"""
#Rename all the columns and drop the unnecessary columns.
data.rename(columns={0: 'Chromosome', 1: 'Start',2:"End",3:"Name",4:"Score",
            5:"Strand",6: 'Chromosome1', 7: 'Start1',8:"End1",9:"Name1",10:"Score1",
            11:"Strand1"},inplace=True)

data.drop(['Score1','Strand1'], axis=1)

data_name=data.groupby(["Start","End","Strand"])["Name"].apply(lambda x: ','.join(x)).reset_index()
data_name1=data.groupby(["Start","End","Strand"])["Name1"].apply(lambda x: ','.join(x)).reset_index()

name_req=data_name["Name"]+","+data_name1["Name1"]
data_chr=data.groupby(["Start","End","Strand"])["Chromosome"].apply(
                    lambda x: ','.join(x.unique()).split(",")[0]).reset_index()["Chromosome"]

data_name["Chromosome"]=data_chr.values

data_name["Name"]=name_req.apply(lambda x: ",".join(list(set(x.split(",")))))
data_name = data_name[['Chromosome', 'Start', 'End', 'Name','Score','Strand']]

#consider the sequence whose maximum site length is 200
data_name_final=data_name[(data_name["End"]-data_name["Start"]<=200)]
data_name_final.shape   #(205309, 6)

##############################################################################################################################

"""
Sorting the protein names based on the number of times they bind with the sequence.
"""

from collections import Counter

names=list(data_name_final["Name"].values)
names_count=Counter(names)
names_sorted={k: v for k, v in sorted(names_count.items(), key=lambda item: item[1],reverse=True)}

"""
{'UCHL5_K562_IDR': 47551,
 'CSTF2T_K562_IDR': 40816,
 'ZNF622_K562_IDR': 33260,
 'EWSR1_K562_IDR': 32547,
.
.
.
'ENCFF723JLS_RPS10_K562_IDR': 273,
 'HNRNPU_K562_IDR': 246,
 'KHDRBS1_K562_IDR': 241,
 'HNRNPA1_K562_IDR': 196}
"""

names_inlist=list(names_sorted.keys())[:30]

##############################################################################################################################

"""
Taking the top 60 protein names from the above sorted list and adding it to the dataframe
"""
def top_60(x):
    for i in x["Name"].split(","):
        if i not in names_inlist:
            return False
    return True
        
final_dataset=data_name_final.apply(top_60,axis=1)

shortened_data=data_name_final[final_dataset]
shortened_data["Name"]=shortened_data["Name"].apply(lambda x: ",".join(list(set(x.split(",")))))

##############################################################################################################################

def ten_thousand(x):
    input_seq=set(x["Name"].split(","))
    compare_seq=set(names_inlist)
    initial_length=len(input_seq)
    diff_seq=input_seq-compare_seq
    final_length=len(diff_seq)
    if initial_length-final_length==1:
        return True
    return False
    
ten_thousand_dataset=data_name_final.apply(ten_thousand,axis=1)

shortened_ten_thousand_dataset=data_name_final[ten_thousand_dataset]
first_10000=shortened_ten_thousand_dataset.iloc[:10000,:]


next_40000=shortened_data.iloc[:47101,:]
shortened_data=pd.concat([first_10000,next_40000])
shortened_data.shape #(57101,6)

##############################################################################################################################

def intersection_finder(x):
    input_seq=set(x["Name"].split(","))
    compare_seq=set(names_inlist)
    int_seq=input_seq.intersection(compare_seq)
    return ",".join(list(int_seq))
shortened_data["Name"]=shortened_data.apply(intersection_finder,axis=1)

check_names=[]
for i in shortened_data["Name"].values:
    name=i.split(",")
    for j in name:
        check_names.append(j)

##############################################################################################################################

"""
Some sequences have less site length i.e.,below 10. Increasing the sitelength by decreasing start position by 30 and 
increasing the end position by 30
"""

import pandas as pd
df=pd.read_csv('file_path')

df["Diff"]=df["End"]-df["Start"]

index=[]
for i,j in enumerate(df["Diff"].values):
    if j<=50:
        index.append(i)

start_list=df["Start"].values
end_list=df["End"].values

for i in index:
    start_list[i]=start_list[i]-30
    end_list[i]=end_list[i]+30


df["Start"]=start_list
df["End"]=end_list

df["Diff"]=df["End"]-df["Start"]

df[df["Diff"]>=50]

df.to_csv("Formatted_dataset.csv")

##############################################################################################################################

"""
Save the dataframe in the bed format. 
"""
import pyranges as pr
import pandas as pd

file_bed=pr.PyRanges(df)
file_bed.to_bed("Formatted_dataset.bed")

##############################################################################################################################

"""
Using the corresponding bed file and human genome(hg38) fasta file, fasta sequences are extracted by using 
"""

$ bedtools getfasta -fi $(fasta_file.fa) -bed $(bed_file.bed)

##############################################################################################################################

import pandas as pd
data_seq=pd.read_csv('sequence_generated_file.txt',sep="/n",header=None)   #fasta sequence generated text file

data=pd.read_csv('formatted_dataset.csv')  

chromosome=[]
seq=[]
position=[]

for i in data_seq.values:
    i=i[0]
    if ">" in i:
        chromosome.append(i.split(":")[0])
        position.append(i.split(":")[1])
    else:
        seq.append(i)

print(len(chromosome),len(position),len(seq))
'''
57101 57101 57101
'''
df_mapping=pd.DataFrame({"Chromosome":chromosome,
                        "Position":position,
                         "Sequence":seq})

df_mapping["Start"]=df_mapping["Position"].apply(lambda x:int(x.split("-")[0]))

df_mapping["End"]=df_mapping["Position"].apply(lambda x:int(x.split("-")[1]))

df_mapping.drop(['Chromosome','Position'],axis=1,inplace=True)


df_final=data.merge(df_mapping,on=["Start","End"],how="inner")


df_final.drop(['Chromosome','Start','End','Name','Score','Strand'],axis=1,inplace=True)


df_final = df_final[['Sequence',
                     'UCHL5_K562_IDR',
 'CSTF2T_K562_IDR',
 'ZNF622_K562_IDR',
 'EWSR1_K562_IDR',
 'AQR_K562_IDR',
 'NONO_K562_IDR',
 'GTF2F1_K562_IDR',
 'FUS_K562_IDR',
 'BUD13_K562_IDR',
 'UPF1_K562_IDR',
 'DDX24_K562_IDR',
 'YBX3_K562_IDR',
 'ENCFF111LMZ_FXR2_K562_IDR',
 'DROSHA_K562_IDR',
 'SF3B4_K562_IDR',
 'PRPF8_K562_IDR',
 'FAM120A_K562_IDR',
 'PABPC4_K562_IDR',
 'AGGF1_K562_IDR',
 'EFTUD2_K562_IDR',
 'ENCFF630KXK_FMR1_K562_IDR',
 'EIF4G2_K562_IDR',
 'FXR2_K562_IDR',
 'TARDBP_K562_IDR',
 'ENCFF136CWQ_FXR1_K562_IDR',
 'ENCFF593RED_TARDBP_K562_IDR',
 'DDX3X_K562_IDR',
 'RBFOX2_K562_IDR',
 'AKAP8L_K562_IDR',
 'GRWD1_K562_IDR']]


df_final
"""
	Sequence	UCHL5_K562_IDR	CSTF2T_K562_IDR	ZNF622_K562_IDR	EWSR1_K562_IDR	AQR_K562_IDR	NONO_K562_IDR	GTF2F1_K562_IDR	FUS_K562_IDR	BUD13_K562_IDR	...	ENCFF630KXK_FMR1_K562_IDR	EIF4G2_K562_IDR	FXR2_K562_IDR	TARDBP_K562_IDR	ENCFF136CWQ_FXR1_K562_IDR	ENCFF593RED_TARDBP_K562_IDR	DDX3X_K562_IDR	RBFOX2_K562_IDR	AKAP8L_K562_IDR	GRWD1_K562_IDR
0	TTCACAAAGCGCCTTCCCCCGTAAATGATATCATCTCAACTTAGTA...	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	TCACAAAGCGCCTTCCCCCGTAAATGATATCATCTCAACTTAGTAT...	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	AAGCGCCTTCCCCCGTAAATGATATCATCTCAACTTAGTATTATAC...	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	AGCGCCTTCCCCCGTAAATGATATCATCTCAACTTAGTATTATACC...	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	CGCCTTCCCCCGTAAATGATATCATCTCAACTTAGTATTATACCCA...	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
57096	GGGAAATTGCTACCTATGGCATGTCCCCTTACCCGGGAATTGACCT...	0	0	0	0	0	0	0	0	0	...	1	0	1	0	0	0	0	0	0	0
57097	TGTAAGCCTTCCTCAGCCTGTTCTCACGAGTATATGTGGGCATTCC...	0	0	0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
57098	CAGAGAAGGTCTATGAACTCATGCGAGCATGTAAGCCTTCCTCAGC...	0	0	0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
57099	CTTCAGACTTTGATAACCGTGAAGAAAGAACAAGATAGAAGGTGAG...	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
57100	AGACTTTGATAACCGTGAAGAAAGAACAAGATAGAAGGTGAGCTGT...	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
57101 rows , 31 columns
"""

##############################################################################################################################

import pandas as pd

df = pd.read_csv("K562_formatted_data.csv")

proportions = []

for col in df.columns:
    vals = df[col].value_counts(normalize=True).to_dict()
    proportions.append((col, vals.get(0, None), vals.get(1, None)))

pdf = pd.DataFrame(proportions, columns=['Label', '0', '1'])
# Proportions of 0 and 1 for each label
pdf

##############################################################################################################################

import torch
from torchmetrics.classification import MultilabelAccuracy

accuracy = MultilabelAccuracy(num_labels=30)

label_cols = [col for col in df.columns if col != 'Sequence']
target = torch.tensor(df[label_cols].values)

predictions = torch.zeros(target.shape)

# What's the accuracy of a model that always predicts 0s?
accuracy(target, predictions)

# What percent of all labels are 0s?
# Note: same as the accuracy above 
(1 - (target.sum() / (target.shape[0] * target.shape[1]))) * 100



