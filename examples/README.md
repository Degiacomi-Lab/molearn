# TRAINING EXAMPLES

This folder contains example scripts demonstrating the usage of *molearn*.

## Dataset
The file `Murd_test.pdb` is an abridged version of the MD trajectory of the MurD protein described in *molearn*'s main publication (see main page of Github repo).
It can be used to verify that *molearn*'s code is running as intended.

## Training protocols

### Training using GPU
- `cuda_example_molearn_trainer.py`: train by evaluating MSE only, running on GPU
- `cuda_example_molearn_physics_trainer.py`: train by combining MSE and energies evaluated internally on GPU
- `cuda_example_molearn_openmm_trainer.py`: train by combining MSE and energies evaluated with OpenMM running on GPU

### Training using CPU
- `example_molearn_trainer.py`: train by evaluating MSE only
- `example_molearn_physics_trainer.py`: train by combining MSE and energies evaluated internally
- `example_molearn_openmm_trainer.py`: train by combining MSE and energies evaluated with OpenMM

