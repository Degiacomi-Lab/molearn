import torch
from molearn.data import PDBData
from molearn.trainers import Trainer
from molearn.models.foldingnet import AutoEncoder


if __name__=='__main__':
    data = PDBData()
    data.import_pdb('MurD_test.pdb')
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])

    trainer = Trainer(device=torch.device('cpu'))
    trainer.set_data(data, batch_size=8, validation_split=0.5, manual_seed = 25)

    trainer.set_autoencoder(AutoEncoder, out_points = data.dataset.shape[-1])
    trainer.prepare_optimiser()

    runkwargs = dict(log_filename='log_file.dat',
                     checkpoint_folder='checkpoint_folder',
                     log_folder = 'checkpoint_folder',
                    )
    trainer.run(max_epochs = 10, **runkwargs)
    print(f'best {trainer.best}')


