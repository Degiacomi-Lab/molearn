# Training Examples

This folder contains example data, neural network parameters, and scripts demonstrating the usage of *molearn*.

## Dataset

The folder `data` contains molecular dynamics (MD) simulation data of the protein MurD. For details, please see [this publication](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052).

#### Training data

The files `MurD_closed.tar.gz` and `MurD_open.tar.gz` contain 900 conformations each of MurD, generated with MD simulations of its closed and open state. Extracting these files will yield `MurD_closed_selection.pdb` and `MurD_open_selection.pdb`.
In order to use them as training data please run the `prepare_example.py` in order to obtain a joined and prepared trajectory.

#### Test data

The file `MurD_closed_apo.tar.gz` contains 900 conformation of an MD simulation of MurD switching from the closed to the open state. Unzipping it will yield the file `MurD_closed_apo_selection.pdb`.

The files `5A5E_full.pdb` and `5A5E_full.pdb` are experimentally determined structures of two intermediates.


## Neural network parameters

In `xbb_foldingnet_checkpoints`, an example output generated when the *foldingnet* neural network is trained on the files in `data` is provided. This includes a chekpoint file (containing neural network parameters) and a logfile, tracking the performance of the neural network during training.


## Scripts

#### Data preparation examples

* `prepare_example.py`: an example on how to combine two trajectories in a single dataset that can be used as training examples for *molearn* models.

#### Training examples

* `bb_foldingnet_basic.py`: minimal example demonstrating how to load data, setup *foldingnet*, and train it. This script operates on training examples in the `data` folder and can be executed as-is (after the multiPDB files are extracted). This script will generate output similar to that provided in the folder `xbb_foldingnet_checkpoints`.
* `bb_example_subclassing_trainer.py`: example of subclassing molearn Trainer and adding features to it. This script features a modification of `bb_foldingnet_basic.py`, and can also be run as-is.
* `bb_many_subclassing_examples.py`: several examples of Trainer subclasses, implementing various features. This script is not intended to be used as-is, and is instead thought as an inspiration for creating your own Trainers.

#### Analysis examples

* `analysis_example.py`: minimal example of analysis of trained neural network. This script operates on the content of the `data` and `xbb_foldingnet_checkpoints` folders. Note that more detailed explanations on analysis are available on our [molearn notebooks](https://github.com/Degiacomi-Lab/molearn_notebook)
