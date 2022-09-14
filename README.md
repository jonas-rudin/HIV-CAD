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

- l/4 -> check SNP rate -> align and compare 5 strains (eg HD)
- clean data beforehand? remove short reads? no
- 5881/5881 [==============================] - 828s 141ms/step - loss: 5.3851e-05 => overfitting no problem? kind of
  should be.
- remove no interesting (no SNPs)
- might not work for too many cluster => test with generated data

## TODO

- adapt one_hot
- adapt pipeline with dimensions (max length)
- how to run on my pc
- align sequences and get difference of
- look at chi's solution possible runtime improvement
- work with fake data -> create this data
- write to cluster of uni basel
- Roth schreiben -> antworten
- 3.70GHz Intel i7-8700K processor, 2 NVIDIA GeForce GTX 1080Ti computer graphics cards and 32GB RAM.

# TODO text

- install texit
- us uni vorlage
- create structure
- measure time


1. Abstract
2. Introduction
3. Related Work
4. Methods
5. Results
    1. time and reconstruction
    2. Discussion
6. Conclusion