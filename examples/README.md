# Training Examples

This folder contains example data, neural network parameters, and scripts demonstrating the usage and analysis of *Molearn*.

## Dataset

The folder `data` contains molecular dynamics (MD) simulation data of the protein MurD. For details, please see [this publication](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052).

#### Training data

The files `MurD_closed.tar.gz` and `MurD_open.tar.gz` contain 900 conformations each of MurD, generated with MD simulations of its closed and open state. Extracting these files will yield `MurD_closed.pdb` and `MurD_open.pdb`.

#### Test data

The file `MurD_closed_apo.tar.gz` contains 900 conformation of an MD simulation of MurD switching from the closed to the open state. Unzipping it will yield the file `MurD_closed_apo.pdb`. The files `5A5E_full.pdb` and `5A5E_full.pdb` are experimentally determined structures of two intermediates.

## Scripts

#### Training examples

* `cb_foldingnet_basic.py`: minimal example demonstrating how to load data, setup *foldingnet*, and train it. This script operates on training examples in the `data` folder and can be executed as-is (with training data extracted). This script will generate an output folder `foldingnet_chechkpoints` that include multiple checkpoint files and a log file, and a `data_statistics.json`. 
The training process can take ~1 hour to converge on a NVIDIA RTX 3080. Adjust the batch size based on your own GPU.

* `example_subclassing_trainer.py`: several examples of subclassing molearn Trainer and adding/overloading features to it. This script is not intended to be used as-is, and is instead thought as an inspiration for creating your own Trainers.

#### Analysis example

* `analysis_example.ipynb`: example analysis notebook of a trained neural network. This notebook operates on the content of the `data` and `foldingnet_checkpoints` folders. 

#### Data preprocessing

* `prepare_example.py`: demonstrates how to combine two trajectories in a single dataset using `Molearn.data.DataAssembler`. This script is not required for running the above training and analysis examples. This class is useful when you are pooling multiple trajectories, potentially with different topologies such as point mutations, into one dataset for model training. You may need to design your own preprocessing pipeline depending on your specific use case.