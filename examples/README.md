# TRAINING EXAMPLES

This folder contains example scripts demonstrating the usage of *molearn*.

## Dataset
The folder `data` contains a dataset of the MurD protein. Please see *molearn*'s main publication (see main page of Github repo) for details.

### Training data

The files `MurD_closed.tar.gz` and `MurD_open.tar.gz` each contain 900 conformations of MurD, generated with MD simulations of its closed and open state. Unzipping these folders will yield the files `MurD_closed_selection.pdb` and `MurD_open_selection.pdb`.

### Validation data

The file `MurD_closed_apo.tar.gz` features 900 conformation of an MD simulation of MurD switching from closed to open state. Unzipping it will yield the file `MurD_closed_apo_selection.pdb`.

The files `5A5E_full.pdb` and `5A5E_full.pdb` are experimentally determined structures of two intermediates.


## Training protocols

* `bb_foldingnet_basic.py`: minimal example demonstrating how to load data, setup foldingnet, and train it

More examples will follow.
