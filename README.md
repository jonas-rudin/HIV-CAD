# HIV Clustering Convolutional Autoencoder

HIV Haplotype Inference Using a Convolutional Auto-Encoder

## Software Requirements

- python 3.8
- tensorflow 2.9
- yaml
- h5py
- sklearn


- bwa
- samtools
- seqtk


- SPAdes 3.12 (for de novo alignment)

## How to run it

## Improvements

- more in matrix not text
- use CUDA for matrix calculations

## Questions

[//]: # (- de novo macht kei sinn mit MEC teste odr?)

- MEC if aligned the wrong way -> should not exist? check how 454/Roche works -> check if have to get complement
  sequence
- l/4 -> check SNP rate -> align and compare 5 strains (eg HD)
- clean data beforehand? remove short reads? no
- 5881/5881 [==============================] - 828s 141ms/step - loss: 5.3851e-05 => overfitting no problem? kind of
  should be.
- remove no interesting (no SNPs)
- might not work for too many cluster => test with generated data

## TODO

- alignment of reads to HXB2
- adapt one_hot
- adapt pipeline with dimensions (max length)
- how to run on my pc
- add majority voting -> remove haplotype alignment
- add MEC and Hamming distance
- align sequences and get difference
- look at chi's solution
- work with fake data -> create this data
- write to cluster of uni basel
- write to sergej
- ettiketen bestellen
- Roth schreiben