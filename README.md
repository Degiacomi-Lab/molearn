# molearn

*molecular conformational spaces meet machine learning*

This software trains a generative neural network on an ensemble of molecular conformations (typically obtained by molecular dynamics).
The trained model can be used to generate new, plausible conformations repesentative of poorly sampled transition states.

Included in this repository are the following:
* Source code is in the `molearn` folder
* Documentation for this is in the `doc` folder.
* Example launching scripts in the `examples` folder

## Requirements ##

Molearn requires Python 3.x and the following packages (and their associated packages):
* numpy
* PyTorch (1.7+)
* [OpenMM](https://openmm.org/documentation)
* [Biobox](https://github.com/Degiacomi-Lab/biobox)

Optional packages are:
* [Modeller](https://salilab.org/modeller/), for calculation of DOPE score
* [geomloss](https://www.kernel-operations.io/geomloss/), for calculation of sinkhorn distances during training
* [cctbx](https://cctbx.github.io/), for calculation of Ramachandran scores during analysis

## Installation ##

molearn requires no installation. Simply clone the repository and make sure the requirements above are met.
molearn can also be obtained through Anaconda: `conda install molearn -c conda-forge`

## Usage ##
* See example scripts in the `examples` and documentation in the `doc` folder.
* Jupyter notebook tutorials describing the usage of a trained neural network are available [here](https://github.com/Degiacomi-Lab/molearn_notebook).

## Reference ##

If you use molearn in your work, please cite:
[V.K. Ramaswamy, S.C. Musson, C.G. Willcocks, M.T. Degiacomi (2021). Learning protein conformational space with convolutions and latent interpolations, Physical Review X 11](
https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052)

## Contact ##

If you have any issues or questions please contact samuel.musson@durham.ac.uk
