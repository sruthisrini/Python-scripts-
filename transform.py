import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import math
from model import RNNDataset
from torch.nn.utils.rnn import pad_sequence


def read_data_to_list(csv_file, labels,skip_n_seqs=True):
    RNA_data = pd.read_csv(csv_file, header=0)
    
    if labels==1:
        RNA_data=RNA_data.iloc[:,:2] #for binary

    RNA_seq = RNA_data["Sequence"].tolist()[:100]

    RNA_labels = RNA_data.iloc[:100, 1:]
    RNA_labels = RNA_labels.values.tolist()

    return RNA_seq, RNA_labels


def string_vectorizer(seq,empty_vectors=False,embed_numbers=False,embed_one_vec=False,custom_alphabet=False):
    alphabet=['A','C','G','T']
    if custom_alphabet:
        alphabet = custom_alphabet
    if empty_vectors:
        vector = []
        for letter in seq:
            vector.append([])
    else:
        if embed_numbers:
            vector = []
            ab2nr_dic = {}
            for idx,c in enumerate(alphabet):
                ab2nr_dic[c] = idx+1
            for letter in seq:
                idx = ab2nr_dic[letter]
                if embed_one_vec:
                    vector.append(idx)
                else:
                    vector.append([idx])
        else:
            vector = torch.Tensor([[1 if char == letter.upper() else 0 for char in alphabet] for letter in seq])
    return vector


def prepare_data(csv_filename,labels):
    if labels==1:
        rna_seq, rna_labels = read_data_to_list(csv_filename,1)
    else:
        rna_seq, rna_labels = read_data_to_list(csv_filename,30)
    rna_vec = []
    for seq in rna_seq:
        vec = string_vectorizer(seq)
        rna_vec.append(vec)

    return rna_vec, rna_labels

def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys 


if __name__ == "__main__":
    csv_file_path = r"Dataset.csv"
    rna_vecs,rna_labels = prepare_data(csv_file_path)
    data=RNNDataset(rna_vecs,rna_labels)
    dataloader=DataLoader(dataset=data,batch_size=32,shuffle=True,collate_fn=pad_collate,pin_memory=True)

    dataiter=iter(dataloader)
    datas=dataiter.next()
    features,*labels=datas
  
    
