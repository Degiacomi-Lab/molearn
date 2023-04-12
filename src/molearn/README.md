# Molearn source

This folder contains *molearn* source code. ***NOTE: this README file is in preparation***.

## Folders

- `analysis`: tools to analyse a trained neural network (see demonstration in [Jupyter notebook tutorials](https://github.com/Degiacomi-Lab/molearn_notebook))
- `models`: neural network models
- `openmm_loss`: classes enabling the evaluation of protein energy via calls to OpenMM
- `parameters`: AMBER parameter files (necessary for evaluation of models energy)
- `scoring`: classes providing tools to quantify the quality of generated models (DOPE score, Ramachandran plots)

## Files

- `loss_functions.py`: evaluation of model energy without OpenMM
- `molearn_trainer.py`: main class to set up and train a neural network with molearn loss function
- `pdb_data.py`: protein data loader
- `protein_handler.py`: functions to load protein and force field data
- `utils.py`: useful functions used throughout the code
