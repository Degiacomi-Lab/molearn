# molearn

[![status](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217/status.svg)](https://joss.theoj.org/papers/781a409020f1c37417067aef6fbc3217)
[![Documentation Status](https://readthedocs.org/projects/molearn/badge/?version=latest)](https://molearn.readthedocs.io/en/latest/?badge=latest)

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

The most recent release can be obtained through Anaconda:

`conda install molearn -c conda-forge` or the much faster `mamba install -c conda-forge molearn`

Manual installation requires the following three steps:
* Clone the repository 
* Install the necessary requirements with `mamba install -c conda-forge --only-deps molearn`. The option `--only-deps` will install the molearn dependencies but not molearn itself.
* Use pip to install molearn from within the molearn directory `python -m pip install .`

Molearn can used without installation by making the sure the requirements above are met, and adding the `src` directory to your path at the beginning of every script, e.g.:
```
import sys
sys.path.insert(0, 'path/to/molearn/src')
import molearn
```

## Usage ##

* See example scripts in the `examples` folder.
* Jupyter notebook tutorials describing the usage of a trained neural network are available [here](https://github.com/Degiacomi-Lab/molearn_notebook).
* software API and a FAQ page are available at [molearn.readthedocs.io](https://molearn.readthedocs.io/).

## Reference ##

If you use molearn in your work, please cite:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

## Contact ##

For any question please contact samuel.musson@durham.ac.uk
