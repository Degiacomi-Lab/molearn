import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import Sinkhorn_Trainer
from molearn.models.foldingnet import AutoEncoder
import torch


import torch
from geomloss import SamplesLoss


#This is an idea of how Sinkhorn_Trainer is implemented in molearn.trainers.sinkhorn_trainer
#We are only using the decoder of the autoencoder.
#i.e. we are training a generator but we still use the autoencoder terms because the classes are subclassed from the original trainer for training autoencoders
'''
#define loss function
self.sinkhorn = SamplesLoss(**kwargs)

#### Sample from a normal distribution
z = torch.randn(batch.shape[0], self.latent_dim, 1).to(self.device)

#### Decode those latent structures
structures = self.autoencoder.decode(z)[:,:,:batch.shape[2]]

#### Calcuate Sinkhorn distance between actual structures and generated structures 
loss = self.sinkhorn(structures.reshape(structures.size(0), -1), batch.reshape(batch.size(0),-1))

#### We alse calculate a physical energy loss and add it
final_loss = results['sinkhorn']+scale*results['physics_loss']
'''

if __name__ == '__main__':

    ##### Load Data #####
    data = PDBData()
    data.import_pdb('data/MurD_closed_selection.pdb')
    data.import_pdb('data/MurD_open_selection.pdb')
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])

    ##### Prepare Trainer #####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Sinkhorn_Trainer(device=device)

    trainer.set_data(data, batch_size=8, validation_split=0.1, manual_seed = 25)
    trainer.prepare_physics(remove_NB = True)
    
    trainer.set_autoencoder(AutoEncoder, out_points = data.dataset.shape[-1])
    trainer.prepare_optimiser()


    ##### Training Loop #####
    #Keep training until loss does not improve for 32 consecutive epochs

    runkwargs = dict(
        log_filename='log_file.dat',
        log_folder='xbb_sinkhorn_checkpoints',
        checkpoint_folder='xbb_sinkhorn_checkpoints',
        )

    best = 1e24
    while True:
        trainer.run(max_epochs = 32+trainer.epoch,**runkwargs)
        if not best>trainer.best:
            break
        best = trainer.best
    print(f'best {trainer.best}, best_filename {trainer.best_name}')
