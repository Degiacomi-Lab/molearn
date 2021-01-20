# molearn

*molecular conformational spaces meet machine learning*

This software trains a generative neural network on an ensemble of molecular conformations (typically obtained by molecular dynamics).
The trained model can be used to generate new, plausible conformations repesentative of poorly sampled transition states.

Included in this repository are the following:
* A stripped down version of biobox, a software suite for loading, manipulating, and saving protein pdb files. The full version of biobox is currently unpublished. 
* Torch functions for calculating the energy of a protein conformation
  * Documentation for this is in the **doc** folder.
  * Source code is in the **molearn** folder
* example_script.py detailing:
  * An example of how to prepare and use the above code
  * A convolutional autoencoder
  * Loading and saving *.pdb* files with biobox
* An example *.pdb* file in the **test** folder
* The [Amber parameters](https://ambermd.org/AmberModels.php) released in the [AmberTools20 package](https://ambermd.org/AmberTools.php) published under a GNU General Public Licence. These are loaded in by the torch potential to calculate the energy of a protein conformation

## Requirements ##

Molearn requires python 3.x and the following packages (and their associated packages):
* numpy
* biobox
  * pandas
* PyTorch (1.7+)

## Installation ##

molearn requires no installation. Simply clone the repository and make sure the requirements above are met

## Usage ##
See the example script "*example_script.py*" and the documentation in the **doc** folder


## Contact ##

If you have any issues or questions please contact use the contact at samuel.musson@durham.ac.uk
