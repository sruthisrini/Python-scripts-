import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
import math
from model import RNNDataset
from torch.nn.utils.rnn import pad_sequence


def read_data_to_list(csv_file, skip_n_seqs=True):
    """
    Read in FASTA sequences, store in dictionary and return dictionary.
    FASTA file can be plain text or gzipped (watch out for .gz ending).

    >>> test_fasta = "test_data/test.fa"
    >>> read_fasta_into_dic(test_fasta)
    {'seq1': 'acguACGUacgu', 'seq2': 'ugcaUGCAugcaACGUacgu'}

    """

    RNA_data = pd.read_csv(csv_file, header=0)
    RNA_seq = RNA_data["Sequence"].tolist()

    RNA_labels = RNA_data.iloc[::, 1:]
    RNA_labels = RNA_labels.values.tolist()

    return RNA_seq, RNA_labels

def string_vectorizer(seq,
                      empty_vectors=False,
                      embed_numbers=False,
                      embed_one_vec=False,
                      custom_alphabet=False):
    """
    Take string sequence, look at each letter and convert to one-hot-encoded
    vector.
    Return array of one-hot encoded vectors.

    empty_vectors:
        If empty_vectors=True, return list of empty vectors.
    custom alphabet:
        Supply custom alphabet list. By default RNA alphabet is used.
    embed_numbers:
        Instead of one-hot, print numbers in order of dictionary (1-based).
        So e.g. ACGU becomes [[1], [2], [3], [4]].

    >>> string_vectorizer("ACGU")
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    >>> string_vectorizer("")
    []
    >>> string_vectorizer("XX")
    [[0, 0, 0, 0], [0, 0, 0, 0]]
    >>> string_vectorizer("ABC", empty_vectors=True)
    [[], [], []]
    >>> string_vectorizer("ACGU", embed_numbers=True)
    [[1], [2], [3], [4]]
    >>> string_vectorizer("ACGU", embed_numbers=True, embed_one_vec=True)
    [1, 2, 3, 4]


    """
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
            vector = torch.Tensor([[1 if char == letter else 0 for char in alphabet] for letter in seq])
    return vector


def prepare_data(csv_filename):
    rna_seq, rna_labels = read_data_to_list(csv_filename)
    rna_vec = []
    for seq in rna_seq:
        vec = string_vectorizer(seq)
        rna_vec.append(vec)

    return rna_vec, rna_labels

def pad_collate(batch):
    (xs, ys) = zip(*batch)
    xs_lens = [len(x) for x in xs]
    # xs = torch.FloatTensor([[x] for x in xs])
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.FloatTensor([[y] for y in ys])
    return xs_pad, ys 


if __name__ == "__main__":
    csv_file_path = r"D:\University_of_Freiburg\Semester-4\project_2\K562\K562_sequenced_data.csv"
    #prepare_data(csv_file_path)
    rna_vecs,rna_labels = prepare_data(csv_file_path)
    data=RNNDataset(rna_vecs,rna_labels)
    dataloader=DataLoader(dataset=data,batch_size=32,shuffle=True,collate_fn=pad_collate,pin_memory=True)

    dataiter=iter(dataloader)
    datas=dataiter.next()
    features,*labels=datas
    print("features:",features)
    print("labels:",labels)

    # epochs=2
    # n_samples=len(rna_vecs)
    # iteration=math.ceil(n_samples/32)
    # print(n_samples,iteration)

    # for epoch in range(epochs):
    #     for i,(input,*label) in enumerate(dataloader):
    #         if(i+1)%5==0:
    #             print(f'epoch:{epoch+1}/{epochs},step:{i+1}/{iteration},inputs:{input.shape}')



