
# Scripts for processing datasets

This folder contains scripts that process the datasets into train, validation and test partitions in a way required to create LMDB datasets. Read the [README](https://github.com/dmitrijsk/AttentionHTR) on the main page of this repo for a better introduction.

## IAM 

* Download the IAM dataset from [here](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) and place the word-level images into the `iam/words` directory. 
* The `/iam_rwth_partitions` directory contains the RWTH Aachen partitions, taken from [the official GitHub repository](https://github.com/omni-us/research-seq2seq-HTR) of [1].
* Run `python3 process_iam.py` to create `iam_train`, `iam_val` and `iam_test` directories with images and gt.txt.
* Convert each partition into an LMDB dataset using the instructions on [the main page](https://github.com/dmitrijsk/AttentionHTR#lmdb-datasets).

Run `python3 process_iam.py --help` to see available command line arguments.

## Imgur5K

Description to be added.

Run `python3 process_imgur5k.py --help` to see available command line arguments.

# References

[1] Kang, L. et al. (2018). Convolve, attend and spell: An attention-based sequence-to-sequence model for handwritten word recognition. In *German Conference on Pattern Recognition* (pp. 459-472). Springer, Cham. https://doi.org/10.1007/978-3-030-12939-2_32

