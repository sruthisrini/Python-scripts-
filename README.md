# Prediction of RNA binding proteins using Long Short Term Memory (LSTM)
This project has taken [RNAProt](https://github.com/BackofenLab/RNAProt) as a reference which uses LSTM model, an advanced version of RNN to find if the RNA sequence binds to a protein or not. Our project aims to find the number of proteins that binds to a sub-sequence of RNA and compares it with the binary classifier. The input to the model is given as sequences in FASTA format as given below,

```
GCTGGGCGGCCCCAAGACCTGCTCTGCCTGGGCTTCTCATTGGTGGCATTTCTCAAGTTTGTCGTCTGCACCATCCGGAAAACCACCTCTCTCT
```


Since RNAProt is used as a reference, the guidelines to clone and use them is given below, and the following approach will only work for Linux systems.
```
git clone https://github.com/BackofenLab/RNAProt.git
conda --version
conda create -n rnaprotenv python=3.8 -c conda-forge -c bioconda
conda activate rnaprotenv
conda install -c bioconda rnaprot
conda config --show channels
conda config --add channels conda-forge
```
 
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
The following approach will only work for Linux systems.

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
## Libraries required to run the code

```
torch
numpy
pandas
sklearn
collections
pyranges
```
## Run the code from the terminal
```
classifier.py -h

positional arguments:
  train_path    Enter the path to the folder containing dataset
  {1,30}        Enter the number of labels. Label 1 is used for binary classifier and Label 30 is used for multi label classifier
  epoch         Enter the number of epochs for training
  patience_val  Enter a value for patience
  test_path     Enter the path to the folder containing the test dataset
  {1,30}        Enter the number of labels for testing. Label 1 is used for binary classifier and Label 30 is used for multi label classifier

options:
  -h, --help    show this help message and exit
 
-----------------------------------------------------------------------------------------------------------------------------------------------------

To clone the repository:
git clone "https://github.com/sruthisrini/RNA-binding-protein-RBP.git"

Example call:
python classifier.py "train_dataset.csv" 30 100 10 "test_dataset.csv" 30 (for multilabel)
python classifier.py "train_dataset.csv" 1 100 10 "test_dataset.csv" 1 (for binary)
```
## Documentation

* The model.py file contains the LSTM model for both binary and multi-label classifiers. Only the number of classes needs to be changed based on the number of labels we have. 
* The transform.py file contains functions necessary to convert the input fasta sequences to one-hot encoded values.
* The utils.py file contains the step-by-step procedure to generate the final dataset required to train and test the model. In order to begin with that process, the metadata needs to be downloaded from [Encode](https://www.encodeproject.org/search/?type=File&searchTerm=bed) which are in BED format.


## Results
Having the binary classifier from [RNAProt](https://github.com/BackofenLab/RNAProt), we compare it with the multi-label classifier for one protein by evaluating the accuracy. It is clearly shown that the binary classifier gives more accuracy than the multi-label classifier.




