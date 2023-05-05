# Training Examples

This folder contains example data and scripts demonstrating the usage of *molearn*.

## Dataset
The folder `data` contains a dataset of the MurD protein. For details, please see [this publication](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052).

#### Training data

The files `MurD_closed.tar.gz` and `MurD_open.tar.gz` each contain 900 conformations of MurD, generated with MD simulations of its closed and open state. Extracting these files will yield `MurD_closed_selection.pdb` and `MurD_open_selection.pdb`.

#### Validation data

The file `MurD_closed_apo.tar.gz` features 900 conformation of an MD simulation of MurD switching from closed to open state. Unzipping it will yield the file `MurD_closed_apo_selection.pdb`.

The files `5A5E_full.pdb` and `5A5E_full.pdb` are experimentally determined structures of two intermediates.


## Neural network parameters

In `xbb_foldingnet_checkpoints`, an example output generated when the *foldingnet* neural network is trained on the files in `data` is provided. This includes a chekpoint file (containing neural network parameters) and a logfile, tracking the performance of the neural network during training.


## Scripts

#### Training examples

* `bb_foldingnet_basic.py`: minimal example demonstrating how to load data, setup *foldingnet*, and train it. This script operates on training examples in the `data` folder and can be executed as-is (after training multiPDB files are extracted). This script will generate output similar to that provided in `xbb_foldingnet_checkpoints`.
* `bb_example_subclassing_trainer.py`: example of subclassing molearn Trainer and adding features to it. This script features a modification of `bb_foldingnet_basic.py`, and can also be run as-is.
* `bb_many_subclassing_examples.py`: several examples of Trainer subclasses, implementing various features. This script is not intended to be used as is, and is instead thought as an inspiration for creating your own Trainers.

#### Analysis examples

* `analysis_example.py`: minimal example of analysis of trained neural network. This script operates on the content of the `data` and `xbb_foldingnet_checkpoints` folders. Note that more detailed explanations on analysis are available out our [molearn notebooks](https://github.com/Degiacomi-Lab/molearn_notebook)
