import torch
from geomloss import SamplesLoss
from .openmm_physics_trainer import OpenMM_Physics_Trainer

class Sinkhorn_Trainer(OpenMM_Physics_Trainer):
    def __init__(self, *args, latent_dim = None, sinkhorn_kwargs = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = 2 if latent_dim is None else latent_dim
        default = dict(loss = 'sinkhorn', p=2, blur=0.05)
        self.sinkhorn = SamplesLoss(default, **sinkhorn_kwargs)

    def common_step(self, batch):
        self._internal = {}
        z = torch.randn(batch.shape[0], self.latent_dim, 1).to(self.device)
        structures = self.autoencoder.decode(z)[:,:,:batch.shape[2]]
        loss = self.sinkhorn(structures.reshape(structures.size(0), -1), batch.reshape(batch.size(0),-1))
        self._internal['decoded'] = structures
        self._internal['encoded'] = z
        return dict(sinkhorn = loss)



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
            scale = (self.psf*results['sinkhorn'])/(results['physics_loss'] +1e-5)
        final_loss = results['sinkhorn']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        '''
        This method overrides :func:`Trainer.valid_step <molearn.trainers.Trainer.valid_step>` and adds an additional 'Physics_loss' term.

        Differently to :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` this method sums the logs of mse_loss and physics_loss ``final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss'])``

        Called from super class :func:`Trainer.valid_epoch<molearn.trainer.Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns:  Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict

        '''

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        # scale = (self.psf*results['sinkhorn'])/(results['physics_loss'] +1e-5)
        final_loss = torch.log(results['sinkhorn'])+self.psf*torch.log(results['physics_loss'])
        results['loss'] = final_loss
        return results

