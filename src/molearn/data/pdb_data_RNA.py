"""RNA-specific coordinate loading helpers.

This module intentionally lives alongside :mod:`pdb_data` while RNA support is
being developed. It avoids the protein-backbone assumptions made by
``PDBData.prepare_dataset``.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import MDAnalysis as mda

from .pdb_data import PDBData


class PDBData_RNA(PDBData):
    """Prepare fixed-topology RNA trajectories for Molearn."""

    def import_pdb_RNA(self, filename, topology=None):
        """Load an RNA coordinate file or trajectory/topology pair."""
        self._mols_RNA = None
        return super().import_pdb(filename, topology)

    def import_trajectories_RNA(self, filenames, topologies):
        """Load trajectories that each require their own topology.

        This is useful for apo/holo ensembles whose full systems differ because
        of ligands, while the RNA atom selection is identical.
        """
        if len(filenames) != len(topologies):
            raise ValueError("Each RNA trajectory must have one topology")
        if not filenames:
            raise ValueError("At least one RNA trajectory is required")

        self._mols_RNA = [
            mda.Universe(topology, filename) if topology is not None
            else mda.Universe(filename)
            for filename, topology in zip(filenames, topologies)
        ]
        self._mol = self._mols_RNA[0]

    def atomselect_RNA(self, atoms=("P", "C4'")):
        """Select RNA atoms, defaulting to phosphate and C4-prime sites."""
        if isinstance(atoms, list) or isinstance(atoms, tuple):
            selection = " or ".join([f"name {name}" for name in atoms])
        elif isinstance(atoms, str):
            selection = atoms
        else:
            raise ValueError("Unsupported RNA atom selection")

        molecules = self._mols_RNA or [self._mol]
        for molecule in molecules:
            molecule.atoms = molecule.select_atoms(selection)

        reference = [
            (atom.name, atom.resname, int(atom.resid))
            for atom in molecules[0].atoms
        ]
        for index, molecule in enumerate(molecules[1:], start=1):
            identity = [
                (atom.name, atom.resname, int(atom.resid))
                for atom in molecule.atoms
            ]
            if identity != reference:
                raise ValueError(
                    "Selected RNA atoms differ in identity or order between "
                    f"trajectory 0 and trajectory {index}"
                )
        self._mol = molecules[0]

    def atomselect_five_site_RNA(self):
        """Select five RNA sites in the order required by the RNA DMAE.

        Each residue is represented as ``P, C4', N1/N9, C2, C4/C6``.
        Explicit ordering is required because MDAnalysis selections preserve
        topology order, which can place C2 after C4 or C6 in MD topologies.
        """
        purines_RNA = {"A", "G", "A3"}
        pyrimidines_RNA = {"C", "U"}
        molecules_RNA = self._mols_RNA or [self._mol]

        for molecule_RNA in molecules_RNA:
            all_atoms_RNA = molecule_RNA.atoms
            ordered_indices_RNA = []
            for residue_RNA in molecule_RNA.residues:
                residue_name_RNA = str(residue_RNA.resname).strip()
                if residue_name_RNA in purines_RNA:
                    expected_RNA = ("P", "C4'", "N9", "C2", "C6")
                elif residue_name_RNA in pyrimidines_RNA:
                    expected_RNA = ("P", "C4'", "N1", "C2", "C4")
                else:
                    continue

                names_RNA = np.asarray(residue_RNA.atoms.names)
                for name_RNA in expected_RNA:
                    matches_RNA = residue_RNA.atoms[names_RNA == name_RNA]
                    if len(matches_RNA) != 1:
                        raise ValueError(
                            f"Expected exactly one {name_RNA!r} atom in "
                            f"RNA residue {residue_name_RNA} "
                            f"{residue_RNA.resid}; found {len(matches_RNA)}"
                        )
                    ordered_indices_RNA.append(matches_RNA[0].index)

            if not ordered_indices_RNA:
                raise ValueError("No supported RNA residues were found")
            molecule_RNA.atoms = all_atoms_RNA[ordered_indices_RNA]

        reference_RNA = [
            (atom_RNA.name, atom_RNA.resname, int(atom_RNA.resid))
            for atom_RNA in molecules_RNA[0].atoms
        ]
        for index_RNA, molecule_RNA in enumerate(molecules_RNA[1:], start=1):
            identity_RNA = [
                (atom_RNA.name, atom_RNA.resname, int(atom_RNA.resid))
                for atom_RNA in molecule_RNA.atoms
            ]
            if identity_RNA != reference_RNA:
                raise ValueError(
                    "Selected five-site RNA atoms differ in identity or "
                    f"order between trajectory 0 and trajectory {index_RNA}"
                )
        self._mol = molecules_RNA[0]

    def _compute_indices_RNA(self) -> None:
        """Store indices for every selected RNA atom name."""
        names = np.asarray(self._mol.atoms.names)
        self.indices = {
            name: torch.as_tensor(np.flatnonzero(names == name), dtype=torch.long)
            for name in np.unique(names)
        }

    def align_coordinates_RNA(self, coords, reference=None):
        """Rigidly align every frame to one shared RNA reference with Kabsch.

        Alignment removes only global translation and rotation; the internal
        conformational differences that the autoencoder must learn are retained.
        """
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("RNA coordinates must have shape (frames, atoms, 3)")

        if reference is None:
            reference = coords[0].copy()
        reference = np.asarray(reference, dtype=float)
        if reference.shape != coords.shape[1:]:
            raise ValueError(
                "RNA alignment reference must have shape "
                f"{coords.shape[1:]}, not {reference.shape}"
            )

        reference_center = reference.mean(axis=0)
        centered_reference = reference - reference_center
        centered = coords - coords.mean(axis=1, keepdims=True)
        covariance = np.einsum("bni,nj->bij", centered, centered_reference)
        left, _, right_t = np.linalg.svd(covariance)
        rotations = left @ right_t

        # A reflection is not a valid molecular rigid-body transformation.
        reflected = np.linalg.det(rotations) < 0
        left[reflected, :, -1] *= -1
        rotations = left @ right_t
        return np.einsum("bni,bij->bnj", centered, rotations) + reference_center

    def write_statistics_RNA(self, filename):
        """Save normalization and shared alignment data for RNA inference."""
        if not hasattr(self, "alignment_reference_RNA"):
            raise ValueError("Run prepare_dataset_RNA before saving statistics")
        statistics = {
            "mean": float(self.mean),
            "std": float(self.std),
            "alignment_reference_RNA": self.alignment_reference_RNA.tolist(),
        }
        if hasattr(self, "ensemble_names_RNA"):
            statistics["ensemble_names_RNA"] = list(self.ensemble_names_RNA)
        if hasattr(self, "frames_per_trajectory_RNA"):
            statistics["frames_per_trajectory_RNA"] = list(
                self.frames_per_trajectory_RNA
            )
        for key in (
            "development_frame_start_RNA",
            "development_frame_stop_RNA",
            "test_frame_start_RNA",
            "validation_split_RNA",
            "training_ensemble_names_RNA",
            "test_ensemble_names_RNA",
            "test_frame_intervals_RNA",
        ):
            if hasattr(self, key):
                statistics[key] = getattr(self, key)
        with open(filename, "w") as outfile:
            json.dump(statistics, outfile)

    def prepare_dataset_RNA(
        self,
        std=None,
        mean=None,
        start=None,
        stop=None,
        step=None,
        align_RNA=True,
        alignment_reference_RNA=None,
    ):
        """Create a normalized RNA tensor using one common coordinate frame."""
        if std is not None and mean is not None:
            self.std = std
            self.mean = mean
        elif (std is None) != (mean is None):
            raise ValueError("std and mean must be supplied together")

        self._ensure_mol_loaded()
        frame_slice = slice(start, stop, step)
        molecules = self._mols_RNA or [self._mol]
        coordinate_sets = [
            np.asarray([
                molecule.atoms.positions.astype(float).copy()
                for _ in molecule.trajectory[frame_slice]
            ])
            for molecule in molecules
        ]
        coords = np.concatenate(coordinate_sets, axis=0)
        if coords.size == 0:
            raise ValueError("The requested RNA trajectory slice contains no frames")

        if align_RNA:
            if alignment_reference_RNA is None:
                alignment_reference_RNA = coords[0].copy()
            coords = self.align_coordinates_RNA(coords, alignment_reference_RNA)
        elif alignment_reference_RNA is not None:
            raise ValueError(
                "alignment_reference_RNA cannot be used when align_RNA is False"
            )
        else:
            alignment_reference_RNA = coords[0].copy()
        self.alignment_reference_RNA = np.asarray(
            alignment_reference_RNA, dtype=float
        ).copy()

        self._compute_indices_RNA()
        coords = self._standardise_coordinates(coords)
        self.dataset = torch.from_numpy(coords).float()
        self._atom_names = list(np.unique(self._mol.atoms.names))
        self.frame_slice_RNA = frame_slice
        self.frames_per_trajectory_RNA = [len(values) for values in coordinate_sets]
        print(f"RNA dataset shape: {self.dataset.shape}")
        return self.dataset

    # Inherited helpers can still lazily prepare data through the standard name.
    prepare_dataset = prepare_dataset_RNA
