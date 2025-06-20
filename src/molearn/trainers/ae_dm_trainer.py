import torch
from .trainer import *
import os
from dataclasses import dataclass
from molearn.loss_functions import openmm_energy

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

@dataclass
class Trainer_Config:
    dm_weight: float = 1.0
    local_k: int = 4
    local_weight: float = 0.9
    nonlocal_weight: float = 0.1

    dihed_weight: float = 0.0
    dihed_bb_weight: float = 0.0
    dihed_imp_weight: float = 0.0

    physics_weight: float = 0.0
    physics_inter_weight: float = 0.0


class AE_DM_Trainer(Trainer):
    def __init__(self, dm_dim, device, config):
        super().__init__(device=device)
        self.dm_dim = dm_dim

        self.dm_weight = config.dm_weight
        self.local_k = config.local_k
        self.local_weight = config.local_weight
        self.nonlocal_weight = config.nonlocal_weight
        self.W = self._get_weight_matrix()

        self.dihed_weight = config.dihed_weight
        self.dihed_bb_weight = config.dihed_bb_weight
        self.dihed_imp_weight = config.dihed_imp_weight

        self.physics_weight = config.physics_weight
        self.physics_inter_weight = config.physics_inter_weight

    @staticmethod
    def calc_dihedral(p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / b1.norm(dim=-1, keepdim=True)
        v = b0 - (b0 * b1_norm).sum(-1, keepdim=True) * b1_norm
        w = b2 - (b2 * b1_norm).sum(-1, keepdim=True) * b1_norm
        x = (v * w).sum(-1)
        y = (b1_norm.cross(v, dim=-1) * w).sum(-1)
        return torch.atan2(y, x)

    def coords_to_dihedral(self, coords: torch.Tensor) -> torch.Tensor:
        B = coords.size(0)
        N  = coords[:, self.n_idx, :]   # (B, n_res, 3)
        CA = coords[:, self.ca_idx, :]
        C  = coords[:, self.c_idx, :]
        # build CB only at the valid residues
        valid = (self.cb_idx > 0)            # BoolTensor, shape (n_res,)
        CB_atoms = self.cb_idx[valid]         # LongTensor of length n_valid
        CB = coords[:, CB_atoms, :]           # shape (B, n_valid, 3)

        # φ at residue 0 is undefined
        C_prev = torch.roll(C, shifts=1, dims=1)
        phi = self.calc_dihedral(C_prev[:, 1:], N[:, 1:], CA[:, 1:], C[:, 1:])
        # ψ at last residue undefined
        N_next = torch.roll(N, shifts=-1, dims=1)
        psi = self.calc_dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N_next[:, :-1])
        # improper torsion
        N_v  = N[:,  valid, :]
        CA_v = CA[:, valid, :]
        C_v  = C[:,  valid, :]
        imp = self.calc_dihedral(N_v, CA_v, C_v, CB)  # (B, n_valid)

        dihedrals = {'bb': torch.concatenate([phi, psi], axis=1), 'imp': imp}
        return dihedrals
    
    def prepare_physics(
        self,
        clamp_threshold=1e8,
        clamp=False,
        xml_file=None,
        soft_NB=True,
        **kwargs,
    ):
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
        print()

    def common_physics_step(self, batch, latent):
        '''
        Called from both :func:`train_step <molearn.trainers.Torch_Physics_Trainer.train_step>` and :func:`valid_step <molearn.trainers.Torch_Physics_Trainer.valid_step>`.
        Takes random interpolations between adjacent samples latent vectors. These are decoded (decoded structures saved as ``self._internal['generated'] = generated if needed elsewhere) and the energy terms calculated with ``self.physics_loss``.

        :param torch.Tensor batch: tensor of shape [batch_size, 3, n_atoms]. Give access to the mini-batch of structures. This is used to determine ``n_atoms``
        :param torch.Tensor latent: tensor shape [batch_size, 2, 1]. Pass the encoded vectors of the mini-batch.
        '''
        alpha = torch.rand(int(len(batch)//2), 1).type_as(latent)
        latent_interpolated = (1-alpha)*latent[:-1:2] + alpha*latent[1::2]
        generated = self.autoencoder.decode(latent_interpolated)

        self._internal["generated"] = generated
        energy = self.physics_loss(generated)
        energy[energy.isinf()] = 1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()
        return {'inter_physics_loss':energy}
    
    def common_step(self, batch):
        self._internal = {}
        dm_batch = self.autoencoder.coords_to_dm(batch)     # [B, 1, n, n]  
        diheds_batch = self.coords_to_dihedral(batch)        # [B, 3*n_res]
        z = self.autoencoder.encoder(dm_batch)
        decoded_coord = self.autoencoder.decode(z)          # [B, n, 3]
        dm_decoded = self.autoencoder.coords_to_dm(decoded_coord)    # [B, 1, n, n]  
        diheds_decoded = self.coords_to_dihedral(decoded_coord)       # [B, 3*n_res]
        dm_loss = self._get_dm_loss(dm_decoded, dm_batch)
        dihed_loss = self._get_dihed_loss(diheds_decoded, diheds_batch)
        
        # Compute energy for batch as physics loss
        energy = self.physics_loss(decoded_coord)
        energy[energy.isinf()] = 1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()

        self._internal["encoded"] = z
        self._internal["decoded"] = decoded_coord
        return dict(dm_loss=dm_loss, dihed_loss=dihed_loss, physics_loss=energy)
    
    def train_epoch(self):
        self.autoencoder.train()
        N = 0
        results = {}
        for i, batch in enumerate(self.train_dataloader):
            batch = batch[0].to(self.device)
            self.optimiser.zero_grad()
            with torch.autograd.detect_anomaly():
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
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        with torch.no_grad():
            self.phy_scale = self.physics_inter_weight * (self.dm_weight * results['dm_loss'])/(results['inter_physics_loss'] + 1e5)
        results["loss"] = self._get_loss(**results)
        return results

    def valid_epoch(self):
        self.autoencoder.eval()
        N = 0 
        results = {}
        for i, batch in enumerate(self.valid_dataloader):
            batch = batch[0].to(self.device)
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

        avg_results = {key: results[key] / N for key in results.keys()}
        self.results_epoch = avg_results
        return {f"valid_{key}": results[key] / N for key in results.keys()}
    
    def valid_step(self, batch):
        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal['encoded']))
        final_loss = self._get_final_loss(**results)
        results["loss"] = final_loss
        return results
    
    def _get_loss(self, **kwargs):
        loss = self.dm_weight * kwargs['dm_loss'] + \
                self.dihed_weight * kwargs['dihed_loss'] + \
                self.physics_weight * kwargs['physics_loss'] + \
                self.phy_scale * kwargs['inter_physics_loss']
        return loss
    
    def _get_dm_loss(self, dm1, dm2):
        """
        dm1, dm2 : [B, 1, n, n] distance matrices
        self.local_k, self.local_weight, self.nonlocal_weight : scalars
        """
        diff = dm1 - dm2    # [B,1,n,n]
        wsq = diff.pow(2) * self.W     # [B,1,n,n]

        per_sample = wsq.sum(dim=(1,2,3)).sqrt()         # [B]
        loss       = per_sample.mean()                   # scalar
        return loss
    
    def _get_dihed_loss(self, decoded, batch):
        # scale the difference in radians to [-pi, pi]
        # so the gradient of the chord distance would not move in the wrong direction
        bb_delta = (decoded['bb'] - batch['bb'] + torch.pi) % (2*torch.pi) - torch.pi
        imp_delta = (decoded['imp'] - batch['imp'] + torch.pi) % (2*torch.pi) - torch.pi
        bb_loss = (1 - torch.cos(bb_delta)).sum(dim=1).mean()   # scalar
        imp_loss = (1 - torch.cos(imp_delta)).sum(dim=1).mean() # scalar
        dihed_loss = self.dihed_bb_weight * bb_loss + self.dihed_imp_weight * imp_loss
        return dihed_loss
        
    def _get_weight_matrix(self):
        """
        Build and return a [1,1,n,n] weight‐matrix W where
        W[..., i, j] = self.local_weight    if |i-j| <= self.local_k
                        self.nonlocal_weight otherwise
        """
        n = self.dm_dim
        # 1) compute |i - j| offsets
        idx = torch.arange(n, device=self.device)
        offs = torch.abs(idx[:, None] - idx[None, :])   # [n, n]
        # 2) mask local vs nonlocal
        local_mask = offs <= self.local_k               # [n, n]
        # 3) fill weight matrix
        W = torch.full((n, n),
                    fill_value=self.nonlocal_weight,
                    device=self.device)
        W[local_mask] = self.local_weight
        # 4) add batch & channel dims for broadcasting
        return W.unsqueeze(0).unsqueeze(0)              # [1, 1, n, n]

    def update_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Trainer has no attribute {key}")
        self.W = self._get_weight_matrix()
        self.best = None
        self._converge_counter = 0
        self.has_converged = False