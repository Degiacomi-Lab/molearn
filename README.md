# molearn

*protein conformational spaces meet machine learning*

molearn is a Python package streamlining the implementation of machine learning models dedicated to the generation of protein conformations from example data obtained via experiment or molecular simulation.

Included in this repository are the following:
* Source code in the `molearn` folder
* Software documentation in the `doc` folder.
* Example training and analysis scripts, along with example data, in the `examples` folder

## Dependencies

#### Required

Molearn requires Python 3.x and the following packages (and their associated packages):
* numpy
* PyTorch (1.7+)
* [Biobox](https://github.com/Degiacomi-Lab/biobox)

#### Optional

To run energy evaluations with OpenMM:
* [OpenMM](https://openmm.org/documentation)
* [openmmtorchplugin](https://github.com/SCMusson/openmmtorchplugin)

To evaluate Sinkhorn distances during training:
* [geomloss](https://www.kernel-operations.io/geomloss/)

To calculate DOPE and Ramachandran scores during analysis:
* [Modeller](https://salilab.org/modeller/)
* [cctbx](https://cctbx.github.io/)

## Installation ##

molearn requires no installation. Simply clone the repository and make sure the requirements above are met.
The most recent release can also be obtained through Anaconda:

`conda install molearn -c conda-forge`

## Usage ##

* See example scripts in the `examples` folder, and documentation in the `doc` folder.
* Jupyter notebook tutorials describing the usage of a trained neural network are available [here](https://github.com/Degiacomi-Lab/molearn_notebook).

## Reference ##

If you use molearn in your work, please cite:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

## Contact ##

For any question please contact samuel.musson@durham.ac.uk
