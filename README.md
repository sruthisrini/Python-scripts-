# This project aims to understand the distribution of K562 and HepG2 Site-ID's binding length
 
The bed files can be downloaded from the website given below,  

[https://www.encodeproject.org/search/?type=File&searchTerm=bed](https://www.encodeproject.org/search/?type=File&searchTerm=bed)

<br>

To work with bed files, bed tools need to be installed by using the following command,
```
$ conda install -c bioconda bedtools
```

<br>

An overlap between two bed files can be calculated by using the following command,
```
$ bedtools intersect -a bed_file_1.bed -b bed_file_2.bed
```

<br>

An overlap between multiple bed files can be calculated by using the following command,
```
$ bedtools intersect -a bed_file_1.bed -b bed_file_2.bed bed_file_3.bed bed_file_4.bed
```
<br>

To extract the sequence of the binding sites, the following command is used,
```
$ bedtools getfasta -fi fasta_file.fa -bed bed_file.bed
```
