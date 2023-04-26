import torch
from molearn.loss_functions import TorchProteinEnergy
from .trainer import Trainer

class Torch_Physics_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(self, physics_scaling_factor=0.1):
        self.psf = physics_scaling_factor
        self.physics_loss = TorchProteinEnergy(self._data.dataset[0]*self.std, pdb_atom_names = self._data.get_atominfo(), device = self.device, method = 'roll')

    def common_physics_step(self, batch, latent, mse_loss):
        alpha = torch.rand(int(len(batch)//2), 1, 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]
        generated = self.autoencoder.decode(latent_interpolated)[:,:,:batch.size(2)]
        bond, angle, torsion =  self.physics_loss._roll_bond_angle_torsion_loss(generated*self.std)
        n = len(generated)
        bond/=n
        angle/=n
        torsion/=n
        _all = torch.tensor([bond, angle, torsion])
        _all[_all.isinf()]=1e35
        total_physics = _all.nansum()
        #total_physics = torch.nansum(torch.tensor([bond ,angle ,torsion]))

        return {'physics_loss':total_physics, 'bond_energy':bond, 'angle_energy':angle, 'torsion_energy':torsion}


    def train_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded'], results['mse_loss']))
        with torch.no_grad():
            scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results

    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded'], results['mse_loss']))
        scale = self.psf*results['mse_loss']/(results['physics_loss']+1e-5)
        final_loss = results['mse_loss']+scale*results['physics_loss']
        results['loss'] = final_loss
        return results


if __name__=='__main__':
    pass