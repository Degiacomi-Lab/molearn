import torch
from molearn.loss_functions import openmm_energy
from .trainer import Trainer
import os


soft_xml_script = """\
<ForceField>
 <Script>
import openmm as mm
nb = mm.CustomNonbondedForce('C/((r/0.2)^4+1)')
nb.addGlobalParameter('C', 1.0)
sys.addForce(nb)
for i in range(sys.getNumParticles()):
    nb.addParticle([])
exclusions = set()
for bond in data.bonds:
    exclusions.add((min(bond.atom1, bond.atom2), max(bond.atom1, bond.atom2)))
for angle in data.angles:
    exclusions.add((min(angle[0], angle[2]), max(angle[0], angle[2])))
for a1, a2 in exclusions:
    nb.addExclusion(a1, a2)
 </Script>
</ForceField>
"""


class OpenMM_Physics_Trainer(Trainer):
    """
    OpenMM_Physics_Trainer subclasses Trainer and replaces the valid_step and train_step.
    An extra 'physics_loss' is calculated using OpenMM and the forces are inserted into backwards pass.
    To use this trainer requires the additional step of calling :func:`prepare_physics <molearn.trainers.OpenMM_Physics_Trainer.prepare_physics>`.

    """

    def __init__(self, physics_inter_weight=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_inter_weight = physics_inter_weight

    def prepare_physics(
        self,
        clamp_threshold=1e8,
        clamp=False,
        xml_file=None,
        soft_NB=True,
        **kwargs,
    ):
        """
        Create ``self.physics_loss`` object from :func:`loss_functions.openmm_energy <molearn.loss_functions.openmm_energy>`
        Needs ``self.mol``, ``self.std``, and ``self._data.atoms`` to have been set with :func:`Trainer.set_data<molearn.trainer.Trainer.set_data>`

        :param float physics_scaling_factor: scaling factor saved to ``self.psf`` that is used in :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>`. Defaults to 0.1
        :param float clamp_threshold: if ``clamp=True`` is passed then forces will be clamped between -clamp_threshold and clamp_threshold. Default: 1e-8
        :param bool clamp: Whether to clamp the forces. Defaults to False
        :param \*\*kwargs: All aditional kwargs will be passed to :func:`openmm_energy <molearn.loss_functions.openmm_energy>`

        """
        if xml_file is None and soft_NB:
            print("using soft nonbonded forces by default")
            from molearn.utils import random_string

            tmp_filename = f"soft_nonbonded_{random_string()}.xml"
            with open(tmp_filename, "w") as f:
                f.write(soft_xml_script)
            xml_file = ["amber14-all.xml", tmp_filename]
            kwargs["remove_NB"] = True
        elif xml_file is None:
            xml_file = ["amber14-all.xml"]
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min=-clamp_threshold)
        else:
            clamp_kwargs = None
        self.physics_loss = openmm_energy(
            self.mol,
            self.std,
            clamp=clamp_kwargs,
            platform="CUDA" if self.device == torch.device("cuda") else "Reference",
            atoms=self._data.atoms,
            xml_file=xml_file,
            **kwargs,
        )
        os.remove(tmp_filename)

    def common_physics_step(self, batch, latent):
        """
        Called from both :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` and :func:`valid_step <molearn.trainers.OpenMM_Physics_Trainer.valid_step>`.
        Takes random interpolations between adjacent samples latent vectors. These are decoded (decoded structures saved as ``self._internal['generated'] = generated if needed elsewhere) and the energy terms calculated with ``self.physics_loss``.

        :param torch.Tensor batch: tensor of shape [batch_size, n_atoms, 3]. Give access to the mini-batch of structures. This is used to determine ``n_atoms``
        :param torch.Tensor latent: tensor shape [batch_size, 2, 1]. Pass the encoded vectors of the mini-batch.
        """
        alpha = torch.rand(int(len(batch) // 2), 1, 1).type_as(latent)
        latent_interpolated = (1 - alpha) * latent[:-1:2] + alpha * latent[1::2]

        generated = self.autoencoder.decode(latent_interpolated)[:,: batch.size(1), : ]
        self._internal["generated"] = generated
        energy = self.physics_loss(generated)
        energy[energy.isinf()] = 1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()

        return {
            "inter_physics_loss": energy
        }  # a if not energy.isinf() else torch.tensor(0.0)}

    def train_step(self, batch):
        """
        This method overrides :func:`Trainer.train_step <molearn.trainers.Trainer.train_step>` and adds an additional 'Physics_loss' term.
        Called from :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: tensor shape [Batch size, Number of Atoms, 3]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        """

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal["encoded"]))
        loss = results["mse_loss"] + self.physics_inter_weight * results["inter_physics_loss"]
        results["loss"] = loss
        return results

    def valid_step(self, batch):
        """
        This method overrides :func:`Trainer.valid_step <molearn.trainers.Trainer.valid_step>` and adds an additional 'Physics_loss' term.

        Differently to :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` this method sums the logs of mse_loss and physics_loss ``final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss'])``

        Called from super class :func:`Trainer.valid_epoch<molearn.trainer.Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, Number of Atoms, 3]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns:  Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict

        """

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal["encoded"]))
        # scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = torch.log(results["mse_loss"]) + self.physics_inter_weight * torch.log(results["inter_physics_loss"])
        results["loss"] = final_loss
        return results


if __name__ == "__main__":
    pass
