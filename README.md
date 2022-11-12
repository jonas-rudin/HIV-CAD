# HIV Convolutional Autoencoder with Clustering

Viruses such as HIV with a high mutation rate tend to build drug resisting strains. Thereby a patient can simultaneously
be the host of multiple HIV strains. Detecting the different haplotypes of the strains is crucial for the treatment of
the patient. The genomics sequences of the viruses are obtained using high throughput sequencing which provides short
base sequences originating from random locations all over the original sequence. Reconstructing the haplotypes given
these sequences poses a difficult problem since the data is oversampled and two sequences do not necessarily overlap,
making it a non-conventional clustering problem. To solve this problem, we built a framework using a convolutional
autoencoder using clustering on the learned feature space to reconstruct viral haplotypes. The network is tested with
synthetic and experimental HIV sequencing data of which it is able to perfectly reconstruct 7 out of 13 HIV genes.

## Software Requirements

### main program

- python v3.8
- tensorflow v2.4
- tensorflow v2.9 if run on Apple M1
- numpy v1.22
- scikit-learn v0.22

### data preparation

- bwa
- samtools

## How to run it

### synthetic sequencing data

check config file:

set:

- data: created
- pooling: False or True

in the created section set:

- haplotype_length
- batch_size
- read_length
- sequencing_error
- threshold
- n_clusters
- coverage

for example as:

- haplotype_length: 1000
- batch_size: 500
- read_length: 250
- sequencing_error: 0.00473
- threshold: 0
- n_clusters: 3
- coverage: 1000

If you want to simulate Illumina reads from 3 strains with a coverage of 1000

Run the program:

1. python main.py

### experimental HIV-1 sequencing data

Check config file:

set:

- data: experimental
- pooling: False or True

in the experimental section set:

choose:

- cleaned: True or False

if True you only net to choose the gene again, the rest of the data can be left as it is. (check that source is set to
Illumina)

if False you have to set:

- source: Illumina or 454/Roche depending on the data
- reads_path: path to the read file (FASTA format)

Then proceed by uncommenting the gene you want to test

Run the program:

1. python prepare_data.py
2. python main.py
