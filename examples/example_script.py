import sys
import os
import shutil
import numpy as np
import time
import torch
import biobox as bb
import csv
from IPython import embed
sys.path.insert(0,'/home/wppj21/Workshop/molearn/src')
import molearn


if __name__=='__main__':
    trainer = molearn.Molearn_Trainer(device=torch.device('cpu'))
    data = molearn.PDBData()
    #data.import_pdb('/home/wppj21/Workshop/proteins/MurD-Degiacomi/MurD_closed_open.pdb')
    data.import_pdb('MurD_test.pdb')
    #data.atomselect(atoms='no_hydrogen')
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    #trainer.set_dataloader(*data.get_dataloader(batch_size=8, validation_split=0.1, dataset_sample_size=80))
    trainer.set_data(data, batch_size=8, validation_split=0.1, dataset_sample_size=80)
    #trainer.get_dataset(filename='/home/wppj21/Workshop/proteins/MurD-Degiacomi/MurD_closed_open.pdb', dataset_sample_size=40)
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
    trainer.get_optimiser(dict(lr=1e-3, momentum=0.9, weight_decay=1e-5))
    runkwargs = dict(log_filename='log_file.dat', checkpoint_folder='checkpoints', max_epochs=100, checkpoint_frequency=1)
    trainer.run(**runkwargs)


