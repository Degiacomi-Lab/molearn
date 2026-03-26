from __future__ import annotations
from copy import deepcopy
import json
import biobox as bb
import os
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import MDAnalysis as mda

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
warnings.filterwarnings("ignore")

radii = {
    "H": 1.20,
    "N": 1.55,
    "NA": 2.27,
    "CU": 1.40,
    "CL": 1.75,
    "C": 1.70,
    "O": 1.52,
    "I": 1.98,
    "P": 1.80,
    "B": 1.85,
    "BR": 1.85,
    "S": 1.80,
    "SE": 1.90,
    "F": 1.47,
    "FE": 1.80,
    "K": 2.75,
    "MN": 1.73,
    "MG": 1.73,
    "ZN": 1.39,
    "HG": 1.8,
    "XE": 1.8,
    "AU": 1.8,
    "LI": 1.8,
    ".": 1.8,
}


class PDBData:
    def __init__(self, filename=None, topology=None, fix_terminal=False, atoms=None, standardise=True):
        """
        Create object enabling the manipulation of multi-PDB files into a dataset suitable for training.

        :param None | str | list[str] filename: if not None, :func:`import_pdb <molearn.data.PDBData.import_pdb>` is called on each filename provided.
        :param None | str topology: if not None, :func:`import_pdb <molearn.data.PDBData.import_pdb>` is called with the topology file.
        :param bool fix_terminal: if True, calls :func:`fix_terminal <molearn.data.PDBData.fix_terminal>` after import, and before atomselect
        :param list[str] atoms: if not None, calls :func:`atomselect <molearn.data.PDBData.atomselect>`
        :param bool standardise: if True, standardise the dataset by removing the mean and dividing by the standard deviation.
        """

        self.filename = filename
        self.topology = topology
        self.standardise = standardise
        if filename is not None:
            self.import_pdb(filename, topology)
            if fix_terminal:
                self.fix_terminal()
            if atoms is not None:
                self.atomselect(atoms=atoms)

    def _ensure_mol_loaded(self):
        if not hasattr(self, "_mol"):
            raise ValueError(
                "No trajectory loaded. Call import_pdb before requesting data."
            )

    def _ensure_dataset_prepared(self):
        if not hasattr(self, "dataset"):
            self.prepare_dataset()
        return self.dataset

    def _prepare_coordinates(self) -> np.ndarray:
        self._ensure_mol_loaded()
        return np.asarray(
            [self._mol.atoms.positions.astype(float) for _ in self._mol.trajectory]
        )

    def _compute_backbone_indices(self):
        n_indices, ca_indices, cb_indices, c_indices, o_indices = [], [], [], [], []
        for i, atom in enumerate(self._mol.atoms):
            if atom.name == 'N':
                if not len(n_indices) == len(ca_indices) == len(c_indices) == len(o_indices):
                    raise ValueError("Inconsistent number of N, CA, C, and O atoms in the trajectory.")
                if len(cb_indices) < len(n_indices):
                    cb_indices.append(-1)        
                n_indices.append(i)
            elif atom.name == 'CA':
                ca_indices.append(i)
            elif atom.name == 'C':
                c_indices.append(i)
            elif atom.name == 'O':
                o_indices.append(i)
            elif atom.name == 'CB':
                cb_indices.append(i)
            else:
                raise ValueError(f"Unknown atom name: {atom.name}. Check atom selection.")
        if not len(n_indices) == len(ca_indices) == len(c_indices) == len(o_indices):
            raise ValueError("Inconsistent number of N, CA, C, and O atoms in the trajectory.")
        if len(cb_indices) < len(ca_indices):
            cb_indices.append(-1)
        self.indices = {
            "N": torch.as_tensor(n_indices,  dtype=torch.long),
            "CA": torch.as_tensor(ca_indices,  dtype=torch.long),
            "C": torch.as_tensor(c_indices,  dtype=torch.long),
            "O": torch.as_tensor(o_indices,  dtype=torch.long),
            "CB": torch.as_tensor(cb_indices,  dtype=torch.long),
        }
        self.cb_valid_idx = self.indices["CB"][self.indices["CB"] >= 0]

    def _standardise_coordinates(self, coords: np.ndarray) -> np.ndarray:
        if self.standardise:
            if not hasattr(self, "std") or not hasattr(self, "mean"):
                self.std = coords.std()
                self.mean = coords.mean()
                print(f"Computed mean: {self.mean}, std: {self.std}")
            if self.std == 0 :
                raise ValueError("Standard deviation of coordinates is zero. Check input data.")
            else:
                print(f"Using pre-computed mean: {self.mean}, std: {self.std}")
        else:
            self.std = 1.0
            self.mean = 0.0
            print("Not standardising the dataset.")
        return (coords - self.mean) / self.std

    def _resolve_split_sizes(
        self,
        total: int,
        validation_split: float,
        valid_size: int | None,
        train_size: int | None,
    ) -> tuple[int, int]:
        if total <= 1:
            raise ValueError("Dataset must contain at least two frames to split.")

        if train_size is None and valid_size is None:
            valid_size = max(1, int(round(total * validation_split)))
            train_size = total - valid_size
        elif train_size is None:
            train_size = total - valid_size
        elif valid_size is None:
            valid_size = total - train_size

        if train_size <= 0 or valid_size <= 0:
            raise ValueError("Train and validation sizes must be positive.")
        if train_size + valid_size > total:
            raise ValueError("Requested split sizes exceed dataset length.")

        return train_size, valid_size

    def _get_split_indices(
        self,
        validation_split=0.1,
        valid_size=None,
        train_size=None,
        manual_seed=None,
        save_indices=False,
        indices_dir='.'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self._ensure_dataset_prepared()
        total = len(dataset)
        train_size, valid_size = self._resolve_split_sizes(
            total, validation_split, valid_size, train_size
        )
        generator = (
            torch.Generator().manual_seed(manual_seed)
            if manual_seed is not None
            else None
        )
        indices = torch.randperm(total, generator=generator)
        train_idx = indices[:train_size]
        valid_idx = indices[train_size : train_size + valid_size]

        if save_indices:
            if indices_dir != '.':
                os.makedirs(indices_dir, exist_ok=True)
            np.savetxt(f"{indices_dir}/train_indices.txt", train_idx.numpy(), fmt="%d")
            np.savetxt(f"{indices_dir}/valid_indices.txt", valid_idx.numpy(), fmt="%d")

        self.train_indices = train_idx
        self.valid_indices = valid_idx
        self._split_permutation = indices
        return train_idx, valid_idx

    def analysis_bundle(self) -> dict:
        """Convenience accessor used by analysis utilities."""

        bundle = self.metadata().copy()
        bundle.update({
            "dataset": self._ensure_dataset_prepared(),
            "mol": self.mol,
        })
        return bundle

    def metadata(self) -> dict:
        """Return a metadata dictionary for trainers and analysis tools."""

        self._ensure_dataset_prepared()
        return {
            "mean": self.mean,
            "std": self.std,
            "atoms": self.atoms,
            "indices": self.indices,
        }

    def import_pdb(self, filename: str | list[str], topology: str | None = None) -> None:
        """
        Load one or multiple trajectory files as MDAnalysis Universe.

        :param str | list[str] filename: the path the trajectory as a str or a list of filepaths to multiple trajectories
        :param str | None topology: the path the topology file for the trajector(y)ies
        """

        if isinstance(filename, list) and topology is None:
            first_universe = mda.Universe(filename[0])
            self._mol = mda.Universe(first_universe._topology, filename)
        elif isinstance(filename, list) and topology is not None:
            first_universe = mda.Universe(topology, filename[0])
            self._mol = mda.Universe(first_universe._topology, filename)
        elif topology is None:
            self._mol = mda.Universe(filename)
        else:
            self._mol = mda.Universe(topology, filename)

    def fix_terminal(self):
        """
        Rename OT1 N-terminal Oxygen to O if terminal oxygens are named OT1 and OT2 otherwise no oxygen will be selected during an atomselect using atoms = ['CA', 'C','N','O','CB']. No template will be found for terminal residue in openmm_loss.
        """
        if (
            len(self._mol.select_atoms("name OT1")) > 0
            and len(self._mol.select_atoms("name OT2")) > 0
        ):
            tmp_names = self._mol.atoms.names
            tmp_names[tmp_names == "OT1"] = "O"
            self._mol.atoms.names = tmp_names

    def atomselect(self, atoms: str | list[str]):
        """
        Select atoms used for training.

        :param str | list[str] atoms: if str then should be used with the MDAnalysis atom selection syntax
                    `https://userguide.mdanalysis.org/1.1.1/selections.html`
                    or a list like `["CA", ..., "O"]`
        """

        selection_string = ""
        if isinstance(atoms, list):
            selection_string = " or ".join([f"name {i}" for i in atoms])
        elif isinstance(atoms, str):
            selection_string = atoms
        else:
            raise ValueError("Unsupported atom selection")
        self._mol.atoms = self._mol.select_atoms(selection_string)

    def prepare_dataset(self, std=None, mean=None) -> torch.Tensor:
        """
        Prepare dataset from the loaded trajectory data to create a standardised/unstandardised tensor.
        """
        if std is not None and mean is not None:
            self.std = std
            self.mean = mean

        self._ensure_mol_loaded()
        coords = self._prepare_coordinates()
        self._compute_backbone_indices()
        coords = self._standardise_coordinates(coords)
        self.dataset = torch.from_numpy(coords).float()
        self._atom_names = list(np.unique(self._mol.atoms.names))
        print(f"Dataset shape: {self.dataset.shape}") # (frames, atoms, 3)
        return self.dataset
            
    def get_atominfo(self):
        """
        Generate list of all atoms in dataset, where every line contains [atom name, residue name, resid]
        """
        if not hasattr(self, "atominfo"):
            assert hasattr(
                self, "_mol"
            ), "You need to call import_pdb before getting atom info"

            self.atominfo = np.asarray(
                [[i.name, i.resname, int(i.resid)] for i in self._mol.atoms],
                dtype=object,
            )
        return self.atominfo

    def frame(self):
        """
        Return `biobox.Molecule` object with loaded data
        """
        M = bb.Molecule()
        _ = self._mol.trajectory[0]
        M.coordinates = self._mol.atoms.positions
        M.coordinates = np.expand_dims(M.coordinates, 0)
        data = []
        for ci, i in enumerate(self._mol.atoms):
            intermediate_data = []
            intermediate_data.append(i.record_type)
            # i.index would also be an option but is different from original PDBData
            # replaces M.data["index"] = np.arange(self._mol.coordinates.shape[1])
            intermediate_data += [ci, i.name, i.resname, i.chainID, i.resid]
            try:
                intermediate_data.append(i.occupancy)
            except (mda.exceptions.NoDataError, IndexError):
                intermediate_data.append(1.0)

            try:
                intermediate_data.append(i.tempfactor)
            except (mda.exceptions.NoDataError, IndexError):
                intermediate_data.append(0.0)

            intermediate_data.append(i.type)

            try:
                intermediate_data.append(i.radius)
            except (mda.exceptions.NoDataError, IndexError):
                try:
                    intermediate_data.append(radii[i.type])
                except KeyError:
                    intermediate_data.append(1.2)

            try:
                intermediate_data.append(i.charge)
            except (mda.exceptions.NoDataError, IndexError):
                intermediate_data.append(0.0)

            data.append(intermediate_data)
        data = pd.DataFrame(
            data,
            columns=[
                "atom",
                "index",
                "name",
                "resname",
                "chain",
                "resid",
                "occupancy",
                "beta",
                "atomtype",
                "radius",
                "charge",
            ],
        )
        M.data = data
        M.current = 0
        _ = self._mol.trajectory[M.current]
        M.points = self._mol.atoms.positions.view()
        M.properties["center"] = M.get_center()
        return deepcopy(M)

    def get_dataloader(
        self,
        batch_size,
        validation_split=0.1,
        pin_memory=True,
        manual_seed=None,
        save_indices=False,
        indices_dir='.'
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        :param int batch_size: size of the training batches
        :param float validation_split: ratio of data to randomly assigned as validation
        :param bool pin_memory: if True, pin memory for the dataloader
        :param int | None manual_seed: 
        :param bool save_indices: if True, save train and valid indices to "train_indices.txt" and "valid_indices.txt"
        :return: `torch.utils.data.DataLoader` for training set
        :return: `torch.utils.data.DataLoader` for validation set
        """
        dataset = self._ensure_dataset_prepared()
        train_idx, valid_idx = self._get_split_indices(
            validation_split=validation_split,
            valid_size=None,
            train_size=None,
            manual_seed=manual_seed,
            save_indices=save_indices,
            indices_dir=indices_dir
        )

        tensor_dataset = torch.utils.data.TensorDataset(dataset)
        train_subset = torch.utils.data.Subset(tensor_dataset, train_idx.tolist())
        valid_subset = torch.utils.data.Subset(tensor_dataset, valid_idx.tolist())

        shuffle_train = manual_seed is None
        self.train_dataset = dataset[train_idx]
        self.valid_dataset = dataset[valid_idx]

        self.train_dataloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle_train,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_subset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=False,
        )
        return self.train_dataloader, self.valid_dataloader

    def get_datasets(
        self, 
        validation_split=0.1, 
        valid_size=None, 
        train_size=None, 
        manual_seed=None, 
        save_indices=False
    ):
        """
        Create a training and validation set from the imported data.
        This is deprecated. Use `get_dataloader` instead.
        
        :param validation_split: ratio of data to randomly assigned as validation
        :param valid_size: if not None, specify number of train structures to be returned
        :param train_size: if not None, speficy number of valid structures to be returned
        :param manual_seed: seed to initialise the random number generator used for splitting the dataset. Useful to replicate a specific split.
        :return: two `torch.Tensor`, for training and validation structures.
        """
        dataset = self._ensure_dataset_prepared()
        train_idx, valid_idx = self._get_split_indices(
            validation_split=validation_split,
            valid_size=valid_size,
            train_size=train_size,
            manual_seed=manual_seed,
            save_indices=save_indices
        )
        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        return train_dataset, valid_dataset

    def write_statistics(self, filename: str):
        """
        Write mean and standard deviation to a JSON file.

        :param str filename: path to the output JSON file
        """
        self._ensure_dataset_prepared()
        stats = {"mean": float(self.mean), "std": float(self.std)}
        with open(filename, "w") as f:
            json.dump(stats, f)
            
    @property
    def atoms(self):
        self._ensure_dataset_prepared()
        if hasattr(self, "_atom_names"):
            return self._atom_names
        return list(np.unique(self.frame().data["name"].values))  # all the atoms

    @property
    def mol(self):
        return self.frame()


if __name__ == "__main__":
    pass
