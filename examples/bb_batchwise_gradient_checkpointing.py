import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import Sinkhorn_Trainer
from molearn.models.foldingnet import AutoEncoder
import torch


import torch
from geomloss import SamplesLoss

#This script is based off of the sinkhorn script.
# We may want a larger batch size for sinkhorn but may not have the GPU memory to do this.
#We can use gradient checkpointing in the batch dimension to give us access to dramatically larger batch sizes in the same memory (We are trading compute for memory so this might be a bit slower)
from molearn.utils import CheckpointBatch

class CustomAutoEncoder(AutoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Gradient checkpointing but in the batch dimension
        #I have been able to use batch sizes of up to 4000 structures (my entire dataset)
        #This function is ideally used with model that don't use batch_norm which foldingnet does.
        #I would recommend replacing with layer norm or group norm in future.
        self.decoder = CheckpointBatch(self.decoder, backward_batch_size = 16, forward_batch_size=16)




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
    
    trainer.set_autoencoder(CustomAutoEncoder, out_points = data.dataset.shape[-1])
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
