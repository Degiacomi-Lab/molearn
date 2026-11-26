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

* `train_foldingnet_withCB.py`: minimal example demonstrating how to load data, setup *foldingnet*, and train it. Atoms are selected as N, CA, CB, C and O. This script operates on training examples in the `data` folder and can be executed as-is (with training data extracted). It generates an output folder `foldingnet_checkpoints` containing checkpoint files and a log, plus a `data_statistics.json`.
The training process can take ~1 hour to converge on a NVIDIA RTX 3080. Adjust the batch size based on your own GPU.

* `train_DMAE_withCB.py`: the same data and atom selection, training `DistanceMatrix_AE` with `DistanceMatrix_AE_Trainer`. The encoder reads an inter-atomic distance matrix and the decoder emits Cartesian coordinates, so the model is invariant to rotation and translation by construction. Output goes to `DMAE_checkpoints`. It runs 3 epochs by default so the example finishes in about a minute; swap in `trainer.run_until_converge()` for a real model. 

* `simrna_distance_matrix_basic_RNA.py`: trains the five-site SimRNA RNA
  representation using one all-site distance matrix. For L nucleotides, the
  matrix is `5L × 5L` over `P, C4', N1/N9, C2, C4/C6` in residue-major order.
  Its unique triangle is packed losslessly before encoding. Distance creation,
  local-k weighting, and full symmetric distance loss match the protein DMAE
  implementation. RNA additionally reports a length- and weight-independent
  `dm_rmse_A` metric. By default, raw frames `[0, 7007)` of every ensemble form
  the training/validation pool and all later frames remain held out for
  testing. The trainer can optionally add the harmonic five-site OpenMM term.
  The original FoldingNet RNA workflow remains available.

* `md_distance_matrix_basic_RNA.py`: trains the same five-site distance-matrix
  model from ordinary MD topology/trajectory pairs. It explicitly reorders
  each nucleotide as `P, C4', N1/N9, C2, C4/C6`, independent of topology atom
  order. By default, apo and CTI raw frames `[0, 22500)` are used for fitting,
  with a random 90/10 training/validation split. Apo frames `[22500, end)` and
  all LUM frames never enter fitting. Their intervals are saved for the
  separate `analysis_MD_RNA.py` workflow.

* `example_subclassing_trainer.py`: several examples of subclassing molearn Trainer and adding/overloading features to it. This script is not intended to be used as-is, and is instead thought as an inspiration for creating your own Trainers.

#### Analysis example

* `analysis_MD_RNA.py`: analyses the separate MD distance-matrix model on the
  test intervals saved during training. It reports reconstruction RMSD,
  compactness, PCA and latent-coordinate diagnostics, generates an apo-to-LUM
  path as connected five-site PDB files, and compares local path geometry with
  the MD ensembles. Use `--detail-analysis` for per-coordinate distributions,
  correlation, variance, and most-separating-dimension plots; `--show` displays
  figures in addition to saving them.

* `analysis_example.ipynb`: example analysis notebook of a trained neural network. This notebook operates on the content of the `data` and `foldingnet_checkpoints` folders.


#### Data preprocessing

* `prepare_example.py`: demonstrates how to combine two trajectories in a single dataset using `Molearn.data.DataAssembler`. This script is not required for running the above training and analysis examples. This class is useful when you are pooling multiple trajectories, potentially with different topologies such as point mutations, into one dataset for model training. You may need to design your own preprocessing pipeline depending on your specific use case.
