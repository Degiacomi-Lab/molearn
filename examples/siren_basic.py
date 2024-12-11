#%%
import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.siren import AutoEncoder
import torch
#%%
if __name__ == '__main__':

    ##### Load Data #####
    data = PDBData()
    #data.import_pdb('data/aggregated_align_1-84.pdb')
    # data.import_pdb('data/MurD_closed_selection.pdb')
    #data.import_pdb('data/MurD_open_selection.pdb')
    data.import_pdb('data/nirk_training.pdb')

    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])

    ##### Prepare Trainer #####
    device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
    trainer = OpenMM_Physics_Trainer(device=device)

    trainer.set_data(data, batch_size=8, validation_split=0.1, manual_seed = 25)
    trainer.prepare_physics(soft_NB = True)
    
    trainer.set_autoencoder(AutoEncoder, mol = data.mol)
    #trainer.prepare_optimiser()
    trainer.optimiser = torch.optim.AdamW(trainer.autoencoder.parameters(), lr=1e-6, weight_decay=0.0001)



    ##### Training Loop #####
    #Keep training until loss does not improve for 32 consecutive epochs

    runkwargs = dict(
        log_filename='log_file.dat',
        log_folder='siren_checkpoints_nirK',
        checkpoint_folder='siren_checkpoints_nirK',
        )

    best = 1e24
    while True:
        trainer.run(max_epochs = 32+trainer.epoch,**runkwargs)
        if not best>trainer.best:
            break
        best = trainer.best
    print(f'best {trainer.best}, best_filename {trainer.best_name}')
# %%
