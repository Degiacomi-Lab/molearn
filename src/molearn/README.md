# Molearn source

This folder contains *molearn* source code.

## Folders

- `analysis`: tools to analyse a trained neural network (see demonstration in [Jupyter notebook tutorials](https://github.com/Degiacomi-Lab/molearn_notebook))
- `models`: neural network models
- `openmm_loss`: classes enabling the evaluation of protein energy via calls to OpenMM
- `parameters`: AMBER parameter files (necessary for evaluation of models energy)
- `scoring`: classes providing tools to quantify the quality of generated models (DOPE score, Ramachandran plots)

## Files

- `molearn_trainer.py`: main class to set up and train a neural network with molearn loss function
- `sinkhorn_trainer.py`: main class to set up and train a neural network with sinkhorn loss function (not imported by default)
- `utils.py`: useful functions used throughout the code
- `pdb_data.py`: protein data handler
- `loss_functions.py`: legacy functions, evaluation of model energy without OpenMM
- `protein_handler.py`: legacy functions, protein data handler without OpenMM
