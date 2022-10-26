import sys
import os
import shutil
import numpy as np
import time
import torch
import biobox as bb
import csv
from IPython import embed
sys.path.insert(0,'/home2/wppj21/Workshop/molearn/src')
import molearn


if __name__=='__main__':
    trainer = molearn.Molearn_Trainer(device=torch.device('cuda'))
    data = molearn.PDBData()
    data.import_pdb('/projects/cgw/proteins/molearn/MurD_closed_open.pdb')
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    trainer.set_data(data, batch_size=128, validation_split=0.1)
    trainer.get_network(autoencoder_kwargs=
            {   'init_z': 32,
                'latent_z': 2,
                'depth': 3,
                'm': 2.0,
                'r': 2,
                'use_spectral_norm': True,
                'use_group_norm': False,
                'num_groups': 8,
                'init_n': 206
                })
    trainer.get_optimiser(dict(lr=1e-4, momentum=0.9))
    runkwargs = dict(log_filename='log_file.dat',
                     checkpoint_folder='checkpoints',
                     checkpoint_frequency=1)
    trainer.run(max_epochs=1000, **runkwargs)
    trainer.update_optimiser_hyperparameters(lr=1e-5)
    trainer.run(max_epochs=2000, **runkwargs)
    trainer.update_optimiser_hyperparameters(lr=1e-6)
    trainer.run(max_epochs=3000, **runkwargs)

