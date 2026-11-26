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
from pathlib import Path

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

    def _compute_indices_RNA(self) -> None:
        """Store indices for every selected RNA atom name."""
        names = np.asarray(self._mol.atoms.names)
        self.indices = {
            name: torch.as_tensor(np.flatnonzero(names == name), dtype=torch.long)
            for name in np.unique(names)
        }

    def get_dataloader_RNA(
        self,
        batch_size,
        validation_split=0.1,
        pin_memory=True,
        manual_seed=None,
        save_indices=False,
        indices_dir=".",
        full_dataset_validation_RNA=False,
        split_by_ensemble_RNA=False,
    ):
        """Create RNA training and validation loaders.

        By default this delegates to the normal Molearn random split. When
        ``full_dataset_validation_RNA`` is true, both loaders contain every
        frame. This is useful for reconstruction experiments, but its
        validation loss is not an independent generalisation estimate.
        """
        if full_dataset_validation_RNA and split_by_ensemble_RNA:
            raise ValueError(
                "Full-data validation and per-ensemble splitting are mutually exclusive"
            )
        if not full_dataset_validation_RNA and not split_by_ensemble_RNA:
            return super().get_dataloader(
                batch_size=batch_size,
                validation_split=validation_split,
                pin_memory=pin_memory,
                manual_seed=manual_seed,
                save_indices=save_indices,
                indices_dir=indices_dir,
            )

        dataset_RNA = self._ensure_dataset_prepared()
        all_indices_RNA = torch.arange(len(dataset_RNA), dtype=torch.long)
        if split_by_ensemble_RNA:
            if not hasattr(self, "frames_per_trajectory_RNA"):
                raise ValueError(
                    "Per-ensemble splitting requires frames_per_trajectory_RNA"
                )
            generator_RNA = (
                torch.Generator().manual_seed(manual_seed)
                if manual_seed is not None else None
            )
            train_parts_RNA = []
            valid_parts_RNA = []
            offset_RNA = 0
            for frame_count_RNA in self.frames_per_trajectory_RNA:
                if frame_count_RNA < 2:
                    raise ValueError(
                        "Every ensemble needs at least two frames for splitting"
                    )
                valid_size_RNA = max(
                    1, int(round(frame_count_RNA * validation_split))
                )
                if valid_size_RNA >= frame_count_RNA:
                    raise ValueError("Validation split leaves no training frames")
                permutation_RNA = (
                    torch.randperm(frame_count_RNA, generator=generator_RNA)
                    + offset_RNA
                )
                train_parts_RNA.append(permutation_RNA[:-valid_size_RNA])
                valid_parts_RNA.append(permutation_RNA[-valid_size_RNA:])
                offset_RNA += frame_count_RNA
            train_indices_RNA = torch.cat(train_parts_RNA)
            valid_indices_RNA = torch.cat(valid_parts_RNA)
        else:
            train_indices_RNA = all_indices_RNA
            valid_indices_RNA = all_indices_RNA

        tensor_dataset_RNA = torch.utils.data.TensorDataset(dataset_RNA)
        if save_indices:
            indices_directory_RNA = Path(indices_dir)
            indices_directory_RNA.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                indices_directory_RNA / "train_indices.txt",
                train_indices_RNA.numpy(), fmt="%d"
            )
            np.savetxt(
                indices_directory_RNA / "valid_indices.txt",
                valid_indices_RNA.numpy(), fmt="%d"
            )

        self.train_indices = train_indices_RNA.clone()
        self.valid_indices = valid_indices_RNA.clone()
        self._split_permutation = torch.cat(
            (train_indices_RNA, valid_indices_RNA)
        )
        self.split_by_ensemble_RNA = bool(split_by_ensemble_RNA)
        self.train_dataset = dataset_RNA[train_indices_RNA]
        self.valid_dataset = dataset_RNA[valid_indices_RNA]
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                tensor_dataset_RNA, train_indices_RNA.tolist()
            ),
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=manual_seed is None,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                tensor_dataset_RNA, valid_indices_RNA.tolist()
            ),
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )
        return self.train_dataloader, self.valid_dataloader

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
        if hasattr(self, "train_indices"):
            statistics["train_indices_RNA"] = self.train_indices.tolist()
        if hasattr(self, "valid_indices"):
            statistics["valid_indices_RNA"] = self.valid_indices.tolist()
        if hasattr(self, "split_by_ensemble_RNA"):
            statistics["split_by_ensemble_RNA"] = bool(
                self.split_by_ensemble_RNA
            )
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
    get_dataloader = get_dataloader_RNA
