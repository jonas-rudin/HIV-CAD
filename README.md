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
- remove no interesting (no SNPs)
- might not work for too many cluster => test with generated data
- small batch size problem?
- what is c++ file doing?
- try chi's approach
- check nn parameter!!!!
- check autoencoder bach_size on input vs on training

## TODO

- adapt CPR for shorter version of 454 for example

- align sequences and get difference of
- look at chi's solution possible runtime improvement
- work with fake data -> create this data
- write to cluster of uni basel
- Roth schreiben -> antworten
- Try de nove in the begining for alignment => see if works as ref
- 3.70GHz Intel i7-8700K processor, 2 NVIDIA GeForce GTX 1080Ti computer graphics cards and 32GB RAM.

# TODO text

- measure time
- limitations -> no new alignment (or maybe do de nove first -> align and see?)
- illumina vs 454 better error rate but less data -> data overweights here? maybe?
- filter data might make it to rough for distant mutations

ssh rudjon00@dmi-wheatstone.dmi.unibas.ch