import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.foldingnet import AutoEncoder
import torch
from molearn.scoring import Parallel_DOPE_Score
from time import time
import numpy as np


class CustomTrainer(OpenMM_Physics_Trainer):
#### All commented out sections are not needed, they are there for demonstration purposes    
    ### This is what common_step looks like in Trainer ###
#        def common_step(self, batch):
#        self._internal = {}
#        encoded = self.autoencoder.encode(batch)
#        self._internal['encoded'] = encoded
#        decoded = self.autoencoder.decode(encoded)[:,:,:batch.size(2)]
#        self._internal['decoded'] = decoded
#        return dict(mse_loss = ((batch-decoded)**2).mean())

    ### This is what common_physics_step looks like in OpenMM_Physics_Trainer ###
#    def common_physics_step(self, batch, latent):
#        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
#        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]
#        generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
#        self._internal['generated'] = generated
#        energy = self.physics_loss(generated)
#        energy[energy.isinf()]=1e35
#        energy = torch.clamp(energy, max=1e34)
#        energy = energy.nanmean()
#        return {'physics_loss':energy}#a if not energy.isinf() else torch.tensor(0.0)}

    ### This is what valid_step looks like in OpenMM_Physics_Trainer ###
#    def valid_step(self, batch):
#        results = self.common_step(batch)
#        results.update(self.common_physics_step(batch, self._internal['encoded']))
#        scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
#        final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss']
#        results['loss'] = final_loss
#        return results

    def valid_step(self, batch):
        results  = super().valid_step(batch)
        #rmsd 
        rmsd = (((batch-self._internal['decoded'])*self.std)**2).sum(dim=1).mean().sqrt()
        results['RMSD'] = rmsd # 'valid_' will automatically be prepended onto this in valid_epoch, to distinguish it from train_step

        #calculate some dope
        if self.first_valid_step:
            self.first_valid_step = False
            if not hasattr(self, 'dope_score_class'):
                self.dope_score_class = Parallel_DOPE_Score(self.mol,processes=torch.get_num_threads())
            #Calculated dope of decoded structures 
            self.dope_scores = []
            decoded_batch = (self._internal['decoded'].permute(0,2,1)*self.std).data.cpu().numpy()
            for f in decoded_batch:
                if np.isfinite(f).all():
                    self.dope_scores.append(self.dope_score_class.get_score(f,refine=True))

            #Calcutate dope of interpolated/generated structures
            self.interp_dope_scores = []
            interpolated_batch = (self._internal['generated'].permute(0,2,1)*self.std).data.cpu().numpy()
            for f in interpolated_batch:
                if np.isfinite(f).all():
                    self.interp_dope_scores.append(self.dope_score_class.get_score(f,refine=True))
            # These will calculate in the background, synchronize at the end of the epoch.
        return results

    def valid_epoch(self, *args, **kwargs):
        self.first_valid_step = self.epoch%5==0
        results = super().valid_epoch(*args, **kwargs)

        # Might as well keep track of cuda memomry once an epoch
        memory = torch.cuda.max_memory_allocated()/1000000.0
        results['Memory'] = memory

        if self.epoch%5==0:
            t1 = time()
            #self.dope_scores contains multiprocessing result objects, get the results
            #This will synchronize the code
            dope = np.array([r.get() for r in self.dope_scores])
            idope = np.array([r.get() for r in self.interp_dope_scores])

            #Dope score returns (DOPE_score, refined_DOPE_score), might as well log both
            results['valid_DOPE'] = dope[:,0].mean()
            results['valid_DOPE_refined'] = dope[:,1].mean()
            results['valid_DOPE_interp'] = idope[:,0].mean()
            results['valid_DOPE_interp_refined'] = idope[:,1].mean()
            results['valid_DOPE_time'] = time()-t1 # extra time taken to calculate DOPE
        results['lr'] = self._lr
        return results



if __name__ == '__main__':

    ##### Load Data #####
    data = PDBData()
    data.import_pdb('data/MurD_closed_selection.pdb')
    data.import_pdb('data/MurD_open_selection.pdb')
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])

    ##### Prepare Trainer #####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = CustomTrainer(device=device)

    trainer.set_data(data, batch_size=8, validation_split=0.1, manual_seed = 25)
    trainer.prepare_physics(remove_NB = True)
    
    trainer.set_autoencoder(AutoEncoder, out_points = data.dataset.shape[-1])
    trainer.prepare_optimiser()


    ##### Training Loop #####
    #Keep training until loss does not improve for 32 consecutive epochs

    runkwargs = dict(
        log_filename='log_file.dat',
        log_folder='xbb_foldingnet_checkpoints',
        checkpoint_folder='xbb_foldingnet_checkpoints',
        )

    best = 1e24
    while True:
        trainer.run(max_epochs = 32+trainer.epoch,**runkwargs)
        if not best>trainer.best:
            break
        best = trainer.best
    print(f'best {trainer.best}, best_filename {trainer.best_filename}')
