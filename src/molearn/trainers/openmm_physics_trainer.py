import torch
from molearn.loss_functions import openmm_energy
from .trainer import Trainer


class OpenMM_Physics_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1, clamp_threshold = 10000, clamp=False, start_physics_at=0, **kwargs):
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min = -clamp_threshold)
        else:
            clamp_kwargs = None
        self.physics_loss = openmm_energy(self.mol, self.std, clamp=clamp_kwargs, platform = 'CUDA' if self.device == torch.device('cuda') else 'Reference', atoms = self._data.atoms, **kwargs)


    def common_physics_step(self, batch, latent):
        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]

        generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
        self._internal['generated'] = generated
        energy = self.physics_loss(generated)
        energy[energy.isinf()]=1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()

        return {'physics_loss':energy}#a if not energy.isinf() else torch.tensor(0.0)}

    def train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        with torch.no_grad():
            scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss'])
        results['loss'] = final_loss
        return results

if __name__=='__main__':
    pass
