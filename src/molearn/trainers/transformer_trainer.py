from __future__ import annotations
import os
from .trainer import Trainer

from molearn.models.transformer import (
    generate_coordinates,
    generate_square_subsequent_mask,
)
import torch.nn as nn
from torch.nn import functional as F


class Transformer_Trainer(Trainer):
    """
    Torch_Physics_Trainer subclasses Trainer and replaces the valid_step and train_step.
    An extra 'physics_loss' (bonds, angles, and torsions) is calculated using pytorch.
    To use this trainer requires the additional step of calling :func: `prepare_physics <molearn.trainers.Torch_Physics_Trainer>`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_network_summary(self):
        """
        returns a dictionary containing information about the size of the autoencoder.
        """

        def get_parameters(trainable_only, model):
            return sum(
                p.numel()
                for p in model.parameters()
                if (p.requires_grad and trainable_only)
            )

        return get_parameters(True, self.autoencoder)

    def train_epoch(self, epoch):
        """
        Train one epoch. Called once an epoch from :func:`trainer.run <molearn.trainers.Trainer.run>`
        This method performs the following functions:
        - Sets network to train mode via ``self.autoencoder.train()``
        - for each batch in self.train_dataloader implements typical pytorch training protocol:

          * zero gradients with call ``self.optimiser.zero_grad()``
          * Use training implemented in trainer.train_step ``result = self.train_step(batch)``
          * Determine gradients using keyword ``'loss'`` e.g. ``result['loss'].backward()``
          * Update network gradients. ``self.optimiser.step``

        - All results are aggregated via averaging and returned with ``'train_'`` prepended on the dictionary key

        :param int epoch: The epoch is passed as an argument however epoch number can also be accessed from self.epoch.
        :returns:  Return all results from train_step averaged. These results will be printed and/or logged in :func:`trainer.run() <molearn.trainers.Trainer.run>` via a call to :func:`self.log(results) <molearn.trainers.Trainer.log>`
        :rtype: dict
        """
        self.autoencoder.train()
        # should be create once a training but is here for easier usage with normal trainer
        self.tgt_mask = generate_square_subsequent_mask(
            self._data.dataset.shape[-1] - 2
        )
        N = 0
        results = {}
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device).permute(0, 2, 1)
            self.optimiser.zero_grad()

            train_result = self.train_step(batch)

            train_result["loss"].backward()
            self.optimiser.step()
            if i == 0:
                results = {
                    key: value.item() * len(batch)
                    for key, value in train_result.items()
                }
            else:
                for key in train_result.keys():
                    results[key] += train_result[key].item() * len(batch)
            N += len(batch)
        return {f"train_{key}": results[key] / N for key in results.keys()}

    def train_step(self, batch):
        """
        Called from :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        """
        results = self.common_step(batch)
        results["loss"] = results["mse_loss"] + results["similarity_loss"]
        return results

    def common_step(self, batch):
        """
        Called from both train_step and valid_step.
        Calculates the mean squared error loss for self.autoencoder.
        Encoded and decoded frames are saved in self._internal under keys ``encoded`` and ``decoded`` respectively should you wish to use them elsewhere.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms] A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return calculated mse_loss
        :rtype: dict
        """

        # Shape: (n_atoms, batch_size, feature_size)
        src = batch[:, :-1].transpose(0, 1)
        tgt = batch[:, 1:].transpose(0, 1)
        # Teacher forcing
        output = self.autoencoder(src, tgt[:-1], tgt_mask=self.tgt_mask)
        mse_loss = nn.MSELoss()(output, tgt[1:])

        # additional loss to force it to generate more diverse structures
        cs_tgt = F.cosine_similarity(tgt[..., None, :, :], tgt[..., :, None, :], dim=0)
        cs_output = F.cosine_similarity(
            output[..., None, :, :], output[..., :, None, :], dim=0
        )
        # adding the loss and scaling by 1500 to be about the range same as the mse loss
        similarity_loss = (cs_tgt - cs_output).abs().sum() / 1500

        return {"mse_loss": mse_loss, "similarity_loss": similarity_loss}

    def valid_epoch(self, epoch):
        """
        Called once an epoch from :func:`trainer.run <molearn.trainers.Trainer.run>` within a no_grad context.
        This method performs the following functions:
        - Sets network to eval mode via ``self.autoencoder.eval()``
        - for each batch in ``self.valid_dataloader`` calls :func:`trainer.valid_step <molearn.trainers.Trainer.valid_step>` to retrieve validation loss
        - All results are aggregated via averaging and returned with ``'valid_'`` prepended on the dictionary key

          * The loss with key ``'loss'`` is returned as ``'valid_loss'`` this will be the loss value by which the best checkpoint is determined.

        :param int epoch: The epoch is passed as an argument however epoch number can also be accessed from self.epoch.
        :returns: Return all results from valid_step averaged. These results will be printed and/or logged in :func:`Trainer.run() <molearn.trainers.Trainer.run>` via a call to :func:`self.log(results) <molearn.trainers.Trainer.log>`
        :rtype: dict
        """
        self.autoencoder.eval()
        N = 0
        results = {}
        for i, batch in enumerate(self.valid_dataloader):
            batch = batch[0].to(self.device).permute(0, 2, 1)
            valid_result = self.valid_step(batch)
            if i == 0:
                results = {
                    key: value.item() * len(batch)
                    for key, value in valid_result.items()
                }
            else:
                for key in valid_result.keys():
                    results[key] += valid_result[key].item() * len(batch)
            N += len(batch)
        # generate one structure every epoch
        start_sequence = batch[0][:100, :].permute(0, 1).cpu().numpy()
        generated_coords = (
            generate_coordinates(
                self.autoencoder,
                start_sequence,
                self.device,
                self._data.dataset.shape[-1],
            )
            * self.std
            + self.mean
        )
        # save new generated structure (based on the first 100 atoms of a test example) as xyz file
        with open(
            f"{os.path.dirname(self.log_filename)}/epoch{epoch}.xyz", "w+"
        ) as cfile:
            cfile.write(f"{len(generated_coords)}\n")
            for j in generated_coords:
                cfile.write(f"C\t{j[0]}\t{j[1]}\t{j[2]}\n")
        return {f"valid_{key}": results[key] / N for key in results.keys()}
