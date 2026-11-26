# molearn

[![status](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217/status.svg)](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217)
[![Documentation Status](https://readthedocs.org/projects/molearn/badge/?version=latest)](https://molearn.readthedocs.io/en/latest/?badge=latest)
![TEST](https://github.com/Degiacomi-Lab/molearn/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/145391811.svg)](https://zenodo.org/badge/latestdoi/145391811)



*protein conformational spaces meet machine learning*

molearn is a Python package streamlining the implementation of machine learning models dedicated to the generation of protein conformations from example data obtained via experiment or molecular simulation.

Included in this repository are the following:
* Source code in the `molearn` folder
* Software documentation (API and FAQ) in the `docs` folder, also accessible at [molearn.readthedocs.io](https://molearn.readthedocs.io/).
* Example training and analysis scripts, along with example data, in the `examples` folder

## Table of contents

- [Dependencies](#dependencies)
  - [Required Packages](#required-packages)
  - [Optional Packages](#optional-packages)
- [Installation](#installation)
  - [Anaconda installation from conda-forge](#anaconda-installation-from-conda-forge)
  - [Clone the repo and manually install](#clone-the-repo-and-manually-install)
  - [Using molearn without installation](#using-molearn-without-installation)
- [Usage](#usage)
- [RNA workflows](#rna-workflows)
  - [Input files and training](#input-files-and-training)
  - [Distance-matrix models and Optuna](#distance-matrix-models-and-optuna)
  - [Reconstruction and transition analysis](#reconstruction-and-transition-analysis)
  - [Optional refinement with SimRNA](#optional-refinement-with-simrna)
- [References](#references)
- [Contributing](#contributing)

## Dependencies

The current version of molearn only supports Linux, and has verified to support Python >=3.9.

#### Required Packages

* numpy
* PyTorch (1.7+)
* [Biobox](https://github.com/Degiacomi-Lab/biobox)
* [MDAnalysis](https://www.mdanalysis.org/)

#### Optional Packages

To prepare a raw trajectory for training:
* [mdtraj](https://mdtraj.org/1.9.4/index.html)

To run energy evaluations with OpenMM:
* [OpenMM](https://openmm.org/documentation)
* [openmmtorchplugin](https://github.com/SCMusson/openmmtorchplugin)

To evaluate Sinkhorn distances during training:
* [geomloss](https://www.kernel-operations.io/geomloss/)

To calculate DOPE and Ramachandran scores during analysis:
* [Modeller](https://salilab.org/modeller/) (requires academic license)
* [cctbx](https://cctbx.github.io/)

To run the GUI:
* [plotly](https://plotly.com/python/)
* [NGLView](http://nglviewer.org/nglview/latest/)

## Installation ##

#### Anaconda installation from conda-forge ####

The most recent release can be obtained through Anaconda:

`conda install molearn -c conda-forge` or the much faster `mamba install -c conda-forge molearn`

We advise the installation is carried out in a new environment.

#### Clone the repo and manually install ####

Manual installation requires the following three steps:
* Clone the repository `git clone https://github.com/Degiacomi-Lab/molearn.git`
* Install all required packages (see section *Dependencies > Required Packages*, above). The easiest way is by calling `mamba install -c conda-forge --only-deps molearn`, where the option `--only-deps` will install the molearn required dependencies but not molearn itself. Optionally, packages enabling additional molearn functionalities can also be installed. This has to be done manually (see links in *Dependencies > Optional Packages*).
* Use pip to install molearn from within the molearn directory `python -m pip install .`

#### Using molearn without installation ####

Molearn can used without installation by making the sure the requirements above are met, and adding the `src` directory to your path at the beginning of every script. For instance, to install all requirements in a new environment `molearn_env`:
```
conda env create --file environment.yml -n molearn_env
```
Then, within this environment, run scripts starting with:

```
import sys
sys.path.insert(0, 'path/to/molearn/src')
import molearn
```

> **Note**
> in case of any installation issue, please consult our [FAQ](https://molearn.readthedocs.io/en/latest/FAQ.html)

## Usage ##

* See example scripts and notebooks in the `examples` folder.
<!-- * Jupyter notebook tutorials describing the usage of a trained neural network are available [here](https://github.com/Degiacomi-Lab/molearn_notebook). -->
* software API and a FAQ page are available at [molearn.readthedocs.io](https://molearn.readthedocs.io/).

## RNA workflows

Molearn uses FoldingNet to learn a two-dimensional latent space from RNA
conformations on this branch. Each nucleotide is represented by five sites:
`P, C4′, N9, C2, C6` for purines and `P, C4′, N1, C2, C4` for pyrimidines.
Ensembles can come from different conditions, such as apo and holo, and must
have matching selected RNA sites. The examples align the ensembles to a
shared reference and save that reference and the coordinate normalization
statistics for analysis.

### Input files and training

- **Molecular dynamics (MD).** Supply a trajectory and its matching PDB topology
  for each ensemble. Set the `trajectory` and `topology` fields in
  [`examples/rna_ensembles.json`](examples/rna_ensembles.json). The loader allows
  separate topologies for systems with different ligands.
- **SimRNA.** Supply a matching `.trc`, `.hdr`, and coarse-grained `.pdb`
  for each ensemble. Set the `trc`, `hdr`, and `pdb` fields in
  [`examples/simrna_ensembles.json`](examples/simrna_ensembles.json). The
  reader uses the PDB to identify the physical sites and excludes trailing
  dummy sites from training.

The manifest paths are examples. Point them to your own inputs using paths
relative to the repository root or absolute paths. Use `--ensemble-config`
to supply an alternate manifest. Run the following commands from the
repository root in your Molearn environment, using the names in your manifest.

For MD training with coordinate reconstruction loss:

```bash
python examples/cb_foldingnet_basic_RNA.py --ensembles apo CTI --frame-step 1
```

For SimRNA FoldingNet training with a random 90/10 split of the combined frames:

```bash
python examples/simrna_foldingnet_basic_RNA.py \
  --ensembles apo holo --frame-step 1 --batch-size 16
```

Both examples use `--frame-step 1` to read all frames and train with validation
early stopping by default. Supply `--max-epochs` to run a fixed number of
epochs, or `--dry-run` to check loading and reconstruction before training.
The SimRNA trainer can add a differentiable local statistical potential for
backbone distances, angles, and coupled η/θ pseudo-torsions. Set
`--simrna-sp-weight` to a positive value to enable it. This training loss uses
the tables installed with Molearn.

### Distance-matrix models and Optuna

The distance-matrix autoencoder encodes the unique pairwise distances between
the five sites per nucleotide and decodes Cartesian coordinates. Its default
latent dimension for SimRNA is 32. The SimRNA training example uses the first
7,007 raw frames per ensemble for training and validation by default, and
reserves later frames for testing. Set `--development-frames` to choose the
boundary for your trajectories.

```bash
python examples/simrna_distance_matrix_basic_RNA.py --ensembles apo holo
python examples/analysis_SimRNA_RNA.py \
  --model distance-map --ensembles apo holo --trajectory-part test --frame-step 1
```

For ordinary MD trajectories, use `examples/md_distance_matrix_basic_RNA.py`
and `examples/analysis_MD_RNA.py`. See the [examples guide](examples/README.md)
for the default MD training and test intervals.

`run_optuna_docker.sh` launches the SimRNA distance-matrix search with a shared
PostgreSQL database and one worker per listed GPU. Use `DATA_DIRECTORY` to point
to your local `apo/` and `holo/` inputs. Set `N_TRIALS` per worker, `PATIENCE`
for validation early stopping, and `MAX_EPOCHS` for the per-trial safety ceiling.

### Reconstruction and transition analysis

Use the [RNA analysis notebook](examples/analysis_example_RNA.ipynb) for MD
results. For the SimRNA run above, the analysis script finds the checkpoint
and statistics under `examples/`:

```bash
python examples/analysis_SimRNA_RNA.py \
  --ensembles apo holo --start-state apo --end-state holo \
  --trajectory-part all --frame-step 1 --transition-states 10
```

The FoldingNet command reports reconstruction over all selected frames,
including frames used for fitting. For an independent evaluation, use a
separate trajectory excluded from training and validation.
Use `--checkpoint` and `--statistics` to specify files explicitly when
analysing a model stored elsewhere.

The script reports reconstruction RMSD, compares original and reconstructed
compactness, plots the latent space with PCA, and writes connected
coarse-grained PDBs. In the SimRNA analysis script, `--transition-states 10`
writes ten structures including the two decoded endpoints. The endpoints
come from representative frames of the selected ensembles, and the
intermediate structures come from latent-space interpolation. Use
`--detail-analysis` for additional diagnostic plots and `--show` for
interactive display.

See the [Slurm guide](slurm/README.md) for running the RNA examples with
Docker-built images and Apptainer on a cluster.

### Optional refinement with SimRNA

To refine generated RNA conformations locally, download the standalone SimRNA
package for your platform from the [SimRNA downloads page](https://genesilico.pl/software/simrna)
or the standalone package link on the [SimRNAweb About page](https://iimcb.genesilico.pl/SimRNAweb/about).

Extract the package and copy its executables and complete `data/` directory
into `SimRNA/` at the root of the Molearn repository. Keep the
`SimRNA` executable and the `SimRNA_trafl2pdbs` converter directly
beside `data/`.

```text
molearn/
└── SimRNA/
    ├── SimRNA
    ├── SimRNA_trafl2pdbs
    └── data/
```

Append `--simrna-refine --simrna-iterations 1000` to the SimRNA analysis
command to run restrained refinement on each generated structure. Molearn
selects the lowest unrestrained-energy frame from each refinement, combines
the selected frames into a TRAFL path, and uses `SimRNA_trafl2pdbs` for
all-atom reconstruction. For a custom installation, specify
`--simrna-executable` and `--simrna-data-directory` and keep the converter
beside the executable.

## References ##

If you use `molearn` in your work, please cite: [S.C. Musson and M.T. Degiacomi (2023). Molearn: a Python package streamlining the design of generative models of biomolecular dynamics. Journal of Open Source Software, 8(89), 5523](https://doi.org/10.21105/joss.05523)

Theory and benchmarks of a neural network training against protein conformational spaces are presented here:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

## Contributing ##

For information on how to report bugs, request new features, or contribute to the code, please see [CONTRIBUTING.md](CONTRIBUTING.md).
For any other question please contact matteo.degiacomi@ed.ac.uk.
