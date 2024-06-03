# molearn

[![status](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217/status.svg)](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217)
[![Documentation Status](https://readthedocs.org/projects/molearn/badge/?version=latest)](https://molearn.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/145391811.svg)](https://zenodo.org/badge/latestdoi/145391811)



*protein conformational spaces meet machine learning*

molearn is a Python package streamlining the implementation of machine learning models dedicated to the generation of protein conformations from example data obtained via experiment or molecular simulation.

Included in this repository are the following:
* Source code in the `molearn` folder
* Software documentation (API and FAQ) in the `docs` folder, also accessible at [molearn.readthedocs.io](https://molearn.readthedocs.io/).
* Example training and analysis scripts, along with example data, in the `examples` folder

## Dependencies

The current version of molearn only supports Linux, and has verified to support Python >=3.9.

#### Required Packages

* numpy
* PyTorch (1.7+)
* [Biobox](https://github.com/Degiacomi-Lab/biobox)

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
* [MDAnalysis](https://www.mdanalysis.org/)
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

* See example scripts in the `examples` folder.
* Jupyter notebook tutorials describing the usage of a trained neural network are available [here](https://github.com/Degiacomi-Lab/molearn_notebook).
* software API and a FAQ page are available at [molearn.readthedocs.io](https://molearn.readthedocs.io/).

## References ##

If you use `molearn` in your work, please cite: [S.C. Musson and M.T. Degiacomi (2023). Molearn: a Python package streamlining the design of generative models of biomolecular dynamics. Journal of Open Source Software, 8(89), 5523](https://doi.org/10.21105/joss.05523)

Theory and benchmarks of a neural network training against protein conformational spaces are presented here:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

## Contributing ##

For information on how to report bugs, request new features, or contribute to the code, please see [CONTRIBUTING.md](CONTRIBUTING.md).
For any other question please contact matteo.t.degiacomi@durham.ac.uk.
