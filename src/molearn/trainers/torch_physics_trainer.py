import torch
from molearn.loss_functions import TorchProteinEnergy
from .trainer import Trainer


class Torch_Physics_Trainer(Trainer):
    '''
    Torch_Physics_Trainer subclasses Trainer and replaces the valid_step and train_step.
    An extra 'physics_loss' (bonds, angles, and torsions) is calculated using pytorch.
    To use this trainer requires the additional step of calling :func: `prepare_physics <molearn.trainers.Torch_Physics_Trainer>`.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1):
        '''
        Create ``self.physics_loss`` object from :func:`loss_functions.TorchProteinEnergy <molearn.loss_functions.TorchProteinEnergy>`
        Needs ``self.std``, ``self._data`` to have been set with :func:`Trainer.set_data <molearn.trainer.Trainer.set_data>`
        :param float physics_scaling_factor: (default: 0.1) scaling factor saved to ``self.psf`` that is used in :func: `train_step <molearn.trainers.Torch_Physics_Trainer.train_step>` It will control the relative importance of mse_loss and physics_loss in training.
        '''
        self.psf = physics_scaling_factor
        self.physics_loss = TorchProteinEnergy(self._data.dataset[0]*self.std, pdb_atom_names=self._data.get_atominfo(), device=self.device, method='roll')

    def common_physics_step(self, batch, latent):
        '''
        Called from both :func:`train_step <molearn.trainers.Torch_Physics_Trainer.train_step>` and :func:`valid_step <molearn.trainers.Torch_Physics_Trainer.valid_step>`.
        Takes random interpolations between adjacent samples latent vectors. These are decoded (decoded structures saved as ``self._internal['generated'] = generated if needed elsewhere) and the energy terms calculated with ``self.physics_loss``.

        :param torch.Tensor batch: tensor of shape [batch_size, 3, n_atoms]. Give access to the mini-batch of structures. This is used to determine ``n_atoms``
        :param torch.Tensor latent: tensor shape [batch_size, 2, 1]. Pass the encoded vectors of the mini-batch.
        '''
        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]
        generated = self.autoencoder.decode(latent_interpolated)[:, :, :batch.size(2)]
        bond, angle, torsion = self.physics_loss._roll_bond_angle_torsion_loss(generated*self.std)
        n = len(generated)
        bond/=n
        angle/=n
        torsion/=n
        _all = torch.tensor([bond, angle, torsion])
        _all[_all.isinf()]=1e35
        total_physics = _all.nansum()
        # total_physics = torch.nansum(torch.tensor([bond ,angle ,torsion]))

        return {'physics_loss':total_physics, 'bond_energy':bond, 'angle_energy':angle, 'torsion_energy':torsion}

    def train_step(self, batch):
        '''
        This method overrides :func:`Trainer.train_step <molearn.trainers.Trainer.train_step>` and adds an additional 'Physics_loss' term.

        Mse_loss and physics loss are summed (``Mse_loss + scale*physics_loss``)with a scaling factor ``self.psf*mse_loss/Physics_loss``. Mathematically this cancels out the physics_loss and the final loss is (1+self.psf)*mse_loss. However because the scaling factor is calculated within a ``torch.no_grad`` context manager the gradients are not computed.
        This is essentially the same as scaling the physics_loss with any arbitary scaling factor but in this case simply happens to be exactly proportional to the ration of Mse_loss and physics_loss in every step. 

        Called from :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: tensor shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        ''' 
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        with torch.no_grad():
            scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        '''
        This method overrides :func:`Trainer.valid_step <molearn.trainers.Trainer.valid_step>` and adds an additional 'Physics_loss' term.

        Differently to :func:`train_step <molearn.trainers.Torch_Physics_Trainer.train_step>` this method sums the logs of mse_loss and physics_loss ``final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss'])``

        Called from super class :func:`Trainer.valid_epoch<molearn.trainer.Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns:  Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict

        '''
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        # scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = torch.log(results['mse_loss'])+self.psf*torch.log(results['physics_loss'])
        results['loss'] = final_loss
        return results


if __name__=='__main__':
    pass
