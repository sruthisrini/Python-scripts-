# Prediction of RNA binding proteins using Pytorch 
 
## Bedtools
* Bedtools is a fast, flexible toolset for genome arithmetic. <br>
* The two most widely used formats for representing genome features are the BED (Browser Extensible Data) and GFF (General Feature Format) formats. <br>
* Bedtools allows one to intersect, merge, count, complement, and shuffle genomic intervals from multiple files in widely-used genomic file formats such as BAM, BED, GFF/GTF, VCF.

The bed files can be downloaded from the website given below,  

[https://www.encodeproject.org/search/?type=File&searchTerm=bed](https://www.encodeproject.org/search/?type=File&searchTerm=bed)

<br>

To work with bed files, bed tools need to be installed by using the following command,
```
$ conda install -c bioconda bedtools
```

## Overlap calculation
Two genome features are said to overlap or intersect if they share at least one base in common. 
An overlap between two bed files can be calculated by using the following command,
```
$ bedtools intersect -a $(bed_file_1.bed) -b $(bed_file_2.bed)
```

<br>

An overlap between multiple bed files can be calculated by using the following command,
```
$ bedtools intersect -a $(bed_file_1.bed) -b $(bed_file_2.bed) $(bed_file_3.bed) $(bed_file_4.bed)
```

## getfasta
bedtools getfasta extracts sequences from a FASTA file for each of the intervals defined in a BED/GFF/VCF file.
To extract the sequence of the binding sites, the following command is used,
```
$ bedtools getfasta -fi $(fasta_file.fa) -bed $(bed_file.bed)
```
##Installation

