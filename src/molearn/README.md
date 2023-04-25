# Molearn source

This folder contains *molearn* source code.

- `analysis`: tools to analyse a trained neural network (see demonstration in [Jupyter notebook tutorials](https://github.com/Degiacomi-Lab/molearn_notebook))
- `data`: protein data handlers
- `loss_functions`: classes enabling loss function evaluation. These include methods to assess models energy via calls to OpenMM
- `models`: neural network models
- `scoring`: classes providing tools to quantify, without gradients, the quality of generated models (DOPE score, Ramachandran plots)
- `trainers`: training protocols
- `utils.py`: useful functions used throughout the code

The folder `loss_functions` contais [Amber parameters](https://ambermd.org/AmberModels.php), released in the [AmberTools20 package](https://ambermd.org/AmberTools.php) under a GNU General Public Licence. These are loaded in by the Torch potential to calculate the energy of a protein conformation.

