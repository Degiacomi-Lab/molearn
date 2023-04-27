# TRAINING EXAMPLES

This folder contains example data and scripts demonstrating the usage of *molearn*.

## Dataset
The folder `data` contains a dataset of the MurD protein. Please see *molearn*'s main publication (see main page of Github repo) for details.

#### Training data

The files `MurD_closed.tar.gz` and `MurD_open.tar.gz` each contain 900 conformations of MurD, generated with MD simulations of its closed and open state. Extracting these files will yield `MurD_closed_selection.pdb` and `MurD_open_selection.pdb`.

#### Validation data

The file `MurD_closed_apo.tar.gz` features 900 conformation of an MD simulation of MurD switching from closed to open state. Unzipping it will yield the file `MurD_closed_apo_selection.pdb`.

The files `5A5E_full.pdb` and `5A5E_full.pdb` are experimentally determined structures of two intermediates.


## Neural network parameters

An example output generated when a neural network is trained is saved in the folder `xbb_foldingnet_checkpoints`. This includes a chekpoint file (containing neural network parameters) and a logfile, tracking the performance of the neural network during training.


## Scripts

* `bb_foldingnet_basic.py`: minimal example demonstrating how to load data, setup foldingnet, and train it.
* `analysis_example.py`: minimal example of analysis of trained neural network. By default, this script will load the content of the `data` and `xbb_foldingnet_checkpoints` folders.
