"""Differentiable baseline OpenMM potential for SimRNA five-site RNA."""

from __future__ import annotations

import numpy as np
import torch

import openmm as mm
from openmm import unit


class _OpenMMSimRNAFunction_RNA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, potential_RNA, coordinates_RNA):
        scale_RNA = potential_RNA.coordinate_scale_RNA
        positions_RNA = (
            coordinates_RNA.detach().cpu().numpy() * scale_RNA
        )
        energies_RNA = np.empty(len(positions_RNA), dtype=np.float32)
        gradients_RNA = np.empty_like(positions_RNA, dtype=np.float32)
        for index_RNA, frame_RNA in enumerate(positions_RNA):
            potential_RNA.context_RNA.setPositions(frame_RNA)
            state_RNA = potential_RNA.context_RNA.getState(
                getEnergy=True, getForces=True
            )
            energies_RNA[index_RNA] = state_RNA.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            forces_RNA = state_RNA.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            # dE/d(normalized coordinate) = -force * std_A / 10.
            gradients_RNA[index_RNA] = -np.asarray(forces_RNA) * scale_RNA
        gradient_tensor_RNA = torch.as_tensor(
            gradients_RNA, device=coordinates_RNA.device,
            dtype=coordinates_RNA.dtype,
        )
        ctx.save_for_backward(gradient_tensor_RNA)
        return torch.as_tensor(
            energies_RNA, device=coordinates_RNA.device,
            dtype=coordinates_RNA.dtype,
        )

    @staticmethod
    def backward(ctx, output_gradient_RNA):
        (coordinate_gradient_RNA,) = ctx.saved_tensors
        return None, coordinate_gradient_RNA * output_gradient_RNA[:, None, None]


class OpenMMSimRNABaseline_RNA(torch.nn.Module):
    """Harmonic five-site CG baseline executed by OpenMM.

    This is deliberately not labelled as the SimRNA statistical score.  It
    restrains the SimRNA connectivity graph to mean distances measured from
    the training ensemble and exposes OpenMM forces to PyTorch autograd.
    """

    masses_RNA = {"P": 95.0, "C4'": 132.0, "N1": 40.0, "N9": 40.0,
                  "C2": 40.0, "C4": 50.0, "C6": 50.0}

    def __init__(self, data_RNA, bond_k_RNA=5000.0, platform_RNA="Reference"):
        super().__init__()
        self.coordinate_scale_RNA = abs(float(data_RNA.std)) / 10.0
        atoms_RNA = list(data_RNA._mol.atoms)
        residues_RNA = []
        lookup_RNA = {}
        for index_RNA, atom_RNA in enumerate(atoms_RNA):
            key_RNA = (str(getattr(atom_RNA, "segid", "")),
                       str(getattr(atom_RNA, "chainID", "")),
                       int(atom_RNA.resid), str(atom_RNA.resname))
            if key_RNA not in lookup_RNA:
                lookup_RNA[key_RNA] = {}
                residues_RNA.append((key_RNA, lookup_RNA[key_RNA]))
            lookup_RNA[key_RNA][str(atom_RNA.name)] = index_RNA

        pairs_RNA = []
        previous_RNA = None
        previous_key_RNA = None
        for key_RNA, sites_RNA in residues_RNA:
            n_RNA = sites_RNA.get("N9", sites_RNA.get("N1"))
            outer_RNA = sites_RNA.get("C6", sites_RNA.get("C4"))
            candidates_RNA = [
                (sites_RNA.get("P"), sites_RNA.get("C4'")),
                (sites_RNA.get("C4'"), n_RNA),
                (n_RNA, sites_RNA.get("C2")),
                (sites_RNA.get("C2"), outer_RNA),
                (outer_RNA, n_RNA),
            ]
            if previous_RNA is not None and previous_key_RNA[:2] == key_RNA[:2]:
                candidates_RNA.append((previous_RNA.get("C4'"), sites_RNA.get("P")))
            pairs_RNA.extend((a_RNA, b_RNA) for a_RNA, b_RNA in candidates_RNA
                             if a_RNA is not None and b_RNA is not None)
            previous_RNA, previous_key_RNA = sites_RNA, key_RNA
        if not pairs_RNA:
            raise ValueError("No SimRNA five-site bonds were found")

        system_RNA = mm.System()
        for atom_RNA in atoms_RNA:
            system_RNA.addParticle(
                self.masses_RNA.get(str(atom_RNA.name), 50.0) * unit.dalton
            )
        force_RNA = mm.CustomBondForce("0.5*k*(r-r0)^2")
        force_RNA.addGlobalParameter("k", float(bond_k_RNA))
        force_RNA.addPerBondParameter("r0")
        pair_tensor_RNA = torch.as_tensor(pairs_RNA, dtype=torch.long)
        with torch.no_grad():
            physical_nm_RNA = data_RNA.dataset * self.coordinate_scale_RNA
            equilibrium_RNA = torch.linalg.vector_norm(
                physical_nm_RNA[:, pair_tensor_RNA[:, 0]]
                - physical_nm_RNA[:, pair_tensor_RNA[:, 1]], dim=2
            ).mean(dim=0)
        for pair_RNA, r0_RNA in zip(pairs_RNA, equilibrium_RNA):
            force_RNA.addBond(*pair_RNA, [float(r0_RNA)])
        system_RNA.addForce(force_RNA)

        integrator_RNA = mm.VerletIntegrator(0.001 * unit.picoseconds)
        platform_object_RNA = mm.Platform.getPlatformByName(platform_RNA)
        self.context_RNA = mm.Context(system_RNA, integrator_RNA, platform_object_RNA)
        self.system_RNA = system_RNA
        self.integrator_RNA = integrator_RNA
        self.number_of_bonds_RNA = len(pairs_RNA)

    def forward(self, coordinates_RNA):
        return _OpenMMSimRNAFunction_RNA.apply(self, coordinates_RNA)
