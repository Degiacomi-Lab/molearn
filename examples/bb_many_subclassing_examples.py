import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.foldingnet import AutoEncoder
import torch
from molearn.scoring import Parallel_DOPE_Score
from time import time
import numpy as np


class ValidRMSDTrainer(OpenMM_Physics_Trainer):
    '''
    Calculate additional valid scores by intercepting valid_step
    '''
    def valid_step(self, batch):
        results  = super().valid_step(batch)
        #rmsd 
        rmsd = (((batch-self._internal['decoded'])*self.std)**2).sum(dim=1).mean().sqrt()
        results['RMSD'] = rmsd # 'valid_' will automatically be prepended onto this in trainer.valid_epoch, to distinguish it from train_step
        return results

class ValidDOPETrainer(OpenMM_Physics_Trainer):
    '''
    calculate  DOPE, this might slow your code down significantly so keep an eye on the 'valid_DOPE_extra_time' 
    '''
    def __init__(self,calculate_dope_every_n = 4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculate_dope_every = calculate_dope_every_n

    def valid_epoch(self, *args, **kwargs):
        '''
        Override valid_epoch, (make sure to call super().valid_epoch(*args, **kwargs) to keep default behavior too). 
        '''
        # DOPE Calculations are slow, so you probably won't want to calculate them every epoch. Instead every 4 to 16 epochs.
        self.calculate_dope = self.epoch%self.calculate_dope_every==0

        # super().valid_epoch will call self.valid_step on every batch in the validation set.
        # dope score will be calculated in self.valid_step.
        results = super().valid_epoch(*args, **kwargs)

        # retrieve the scores, clear the scores list, and log the results.
        t1 = time()
        dope = np.array([r.get() for r in self.interp_dope_scores])
        self.interp_dope_scores = []
        t2 = time()
        results['valid_DOPE'] = dope.mean()
        results['valid_DOPE_extra_time'] = t2 - t1
        return results

    def valid_step(self, batch):
        '''
        Keep default behavior by calling super().valid_step(batch), submit jobs to a multiprocessing pool with Parallel_DOPE_Score, then get results back at the end of the epoch in trainer.valid_epoch               
        '''
        results = super().valid_step(batch)
        if not hasattr(self, 'dope_score_class'):
            #Setting this up will be slow the first time
            self.dope_score_class = Parallel_DOPE_Score(self.mol, processes = torch.get_num_threads())
            self.interp_dope_scores = [] # store multiprocessing results objects here
        if not self.calculate_dope:
            # helpfully, we have saved the decoded, and generated (by interpolation) structures in self._internal dict
            # We could calculate dope on either or both
            #structures = (self._internal['decoded'].permute(0,2,1)*self.std).data.cpu().numpy()
            structures = (self._internal['generated'].permute(0,2,1)*self.std).data.cpu().numpy()
            for f in structures:
                if np.isfinite(f).all():
                    self.interp_dope_scores.append(self.dope_score_class.get_score(f, refine=False))
                ### We don't need the results until the validation epoch is done, so synchronise in self.valid_epoch
        return results


class TrackMemoryTrainer(OpenMM_Physics_Trainer):
    '''
    How one might track memory usage
    '''
    def valid_epoch(self, *args, **kwargs):
        results = super().valid_epoch(*args, **kwargs)
        memory = torch.cuda.max_memory_allocated()/1000000.0
        torch.cuda.reset_max_memory_allocated()
        results['Memory'] = memory
        return results
        
class DisablePhysicsTrainer(OpenMM_Physics_Trainer):
    '''
    Disable Physics in train_step. This essentially resets the train_step to that of Molearn.trainers.Trainer.
    OpenMM energy scores are still calculate in the validation loop
    '''

    def train_step(self, batch):
        # self.common_step encodes and decodes the batch and returns mean squared error in a dictionary with key 'mse_loss'
        # encoded vectors and decoded structures are saved in self._internal dict with keys 'encoded' and 'decoded' respectively
        results = self.common_step(batch)
        # backwards will be called on whatever is saved with the key 'loss'
        results['loss'] = results['mse_loss']

        # If you still want to calculate physics uncomment the following line, results will be logged but not used to train the network
        # results.update(self.common_physics_step(batch, self._internal['encoded']))

        return results

class CalculateDecodedEnergyTrainer(OpenMM_Physics_Trainer):
    '''
    OpenMM_Physics_Trainer calculates the energy off interpolated structures. It might be helpful to track the energy of decoded structures too. 
    '''
    def common_physics_step(self, batch, latent):
        # calculate interpolated structures energy as normal
        results = super().common_physics_step(batch, latent)

        #helpfully we save decoded structures in self.common_step
        decoded = self._internal['decoded']

        #calculate energy, and remove any results than might break training
        energy= self.physics_loss(decoded)
        energy[energy.isinf()] = 1e34 # infinities are bad for healthy training
        energy = energy.nanmean() # Nan values are bad for your mental health

        results['physics_decoded_loss': energy]
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
    trainer = <insert_name_of_trainer_here>(device=device)

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
    print(f'best {trainer.best}, best_filename {trainer.best_name}')
