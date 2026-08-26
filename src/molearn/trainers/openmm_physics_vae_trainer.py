import os
import torch
from molearn.loss_functions import openmm_energy

from molearn.trainers import OpenMM_Physics_Trainer

class OpenMM_Physics_Trainer_VAE(OpenMM_Physics_Trainer):
    """
    A modified OpenMM Physics Trainer suitable for training a Variational Autoencoder
    """

    def __init__(self, kld_weight=1e-4, physics_inter_weight=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kld_weight = kld_weight

    def common_step(self, batch):
        """
        Called from both train_step and valid_step.
        Calculates the mean squared error loss for self.autoencoder.
        Encoded and decoded frames are saved in self._internal under keys ``encoded`` and ``decoded`` respectively should you wish to use them elsewhere.

        :param torch.Tensor batch: Tensor of shape [Batch size, Number of Atoms, 3] A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return calculated mse_loss and the kld_loss (KL(q_\phi(z|x) || p(z)))
        :rtype: dict
        """
        self._internal = {}
        encoded, mu, logvar = self.autoencoder.encode(batch)
        self._internal["encoded"] = encoded
        decoded = self.autoencoder.decode(encoded)[:, : batch.size(1), :]
        self._internal["decoded"] = decoded
        geometric_loss = torch.nn.functional.mse_loss(batch , decoded)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0) # Closed form KL between Gaussians
        return dict(mse_loss=geometric_loss, kld_loss=kld_loss) 

    def train_step(self, batch):
        """
        This method overrides :func:`OpenMM_Physics_Trainer.train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` and adds an additional 'kld_loss' term.
        Called from :func:`OpenMM_Physics_Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: tensor shape [Batch size, Number of Atoms, 3]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        """

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal["encoded"]))
        loss = results["mse_loss"] + self.kld_weight * results["kld_loss"] + self.physics_inter_weight * results["inter_physics_loss"]
        results["loss"] = loss
        return results

    def valid_step(self, batch):
        """
        This method overrides :func:`OpenMM_Physics_Trainer.valid_step <molearn.trainers.OpenMM_Physics_Trainer.valid_step>` and adds an additional 'kld_loss' term.

        Differently to :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer_VAE.train_step>` this method sums the logs of mse_loss, kld_loss, and physics_loss ``final_loss = torch.log(results['mse_loss'])+kld_scale*torch.log(results["kld_loss"])+scale*torch.log(results['physics_loss'])``

        Called from super class :func:`OpenMM_Physics_Trainer.valid_epoch<molearn.trainer.OpenMM_Physics_Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, Number of Atoms, 3]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns:  Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict

        """

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal["encoded"]))
        physics_loss = self.physics_inter_weight * torch.log(results["inter_physics_loss"])
        kld_loss = self.kld_weight * torch.log(results["kld_loss"])
        final_loss = torch.log(results["mse_loss"]) + physics_loss + kld_loss
        results["loss"] = final_loss
        return results


if __name__ == "__main__":
    pass
