from __future__ import annotations
from copy import deepcopy
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
    def __init__(self, filename=None, topology=None, fix_terminal=False, atoms=None):
        """
        Create object enabling the manipulation of multi-PDB files into a dataset suitable for training.

        :param filename: None, str or list of strings. If not None, :func:`import_pdb <molearn.data.PDBData.import_pdb>` is called on each filename provided.
        :param fix_terminal: if True, calls :func:`fix_terminal <molearn.data.PDBData.fix_terminal>` after import, and before atomselect
        :param atoms: if not None, calls :func:`atomselect <molearn.data.PDBData.atomselect>`
        """

        self.filename = filename
        self.topology = topology
        if filename is not None:
            self.import_pdb(filename, topology)
            if fix_terminal:
                self.fix_terminal()
            if atoms is not None:
                self.atomselect(atoms=atoms)

    def import_pdb(self, filename: str | list[str], topology: str | None = None):
        """
        Load one or multiple trajectory files

        :param str | list[str] filename: the path the trajectory as a str or a list of filepaths to multiple trajectories
        :param str | None topology: the path the topology file for the trajector(y)ies
        """

        if isinstance(filename, list) and topology is None:
            first_universe = mda.Universe(filename[0])
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
        Select atoms of interest

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
            raise ValueError("Unsuported atom selection")
        self._mol.atoms = self._mol.select_atoms(selection_string)

    def prepare_dataset(self):
        """
        Once all datasets have been loaded, normalise data and convert into `torch.Tensor` (ready for training)
        """
        if not hasattr(self, "dataset"):
            assert hasattr(
                self, "_mol"
            ), "You need to call import_pdb before preparing the dataset"

        self.dataset = np.asarray(
            [self._mol.atoms.positions.astype(float) for _ in self._mol.trajectory]
        )
        if not hasattr(self, "std"):
            self.std = self.dataset.std()
        if not hasattr(self, "mean"):
            self.mean = self.dataset.mean()
        self.dataset -= self.mean
        self.dataset /= self.std
        self.dataset = torch.from_numpy(self.dataset).float()
        self.dataset = self.dataset.permute(0, 2, 1)
        print(f"Dataset shape: {self.dataset.shape}")
        print(f"mean: {str(self.mean)}\n std: {str(self.std)}")

    def contactmap(
        self,
        dist_th: float | None = None,
        mask_val: float = 0.0,
        binary_map: bool = False,
    ):
        """
        create dataset `self.contact_maps` containing the contact maps for each frame and replace `self.dataset` with `self.contact_maps` as new dataset

        :param dist_th float | None: distances bigger than that get set to mask_val
        :param mask_val float: value to which exceeding distances are set
        :param binary_map bool: True to convert the distance map to a binary interacting/not interacting map based on dist_th - `mask_val` should be set to 0.0
        """
        assert hasattr(
            self, "mean"
        ), "You need to call prepare_dataset before creating the contact map dataset"
        if binary_map:
            assert (
                dist_th is not None
            ), "Interacting distance needs to be provided for creation of the binary interaction map"

        d0, _, d2 = self.dataset.shape
        self.contact_maps = torch.empty((d0, d2, d2))
        for ci, i in enumerate(self.dataset.permute(0, 2, 1)):
            # so real sized data is used when distance mask should be applied
            if dist_th is not None:
                i = i * self.std + self.mean

            arr1_coords_rs = i.reshape(i.shape[0], 1, 3)
            arr2_coord_rs = i.reshape(1, i.shape[0], 3)
            # calculating the distance between each point and returning a 2D array with all distances
            dist = ((arr1_coords_rs - arr2_coord_rs) ** 2).sum(axis=2).sqrt()

            if dist_th is not None:
                # everything further apart than dist_th will be set to mask_val
                th_bool = dist > dist_th
                dist[th_bool] = mask_val
                if binary_map:
                    dist[~th_bool] = 1.0
                self.contact_maps[ci] = dist

        if not binary_map:
            self.contact_std = self.contact_maps.std()
            self.contact_mean = self.contact_maps.mean()
            self.contact_maps -= self.contact_mean
            self.contact_maps /= self.contact_std
        # to be able to just use self.get_dataloader for training without any modification
        self.dataset = self.contact_maps

    def get_atominfo(self):
        """
        generate list of all atoms in dataset, where every line contains [atom name, residue name, resid]
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
        return `biobox.Molecule` object with loaded data
        """
        M = bb.Molecule()
        _ = self._mol.trajectory[0]
        M.coordinates = self._mol.atoms.positions
        M.coordinates = np.expand_dims(M.coordinates, 0)
        data = []
        for ci, i in enumerate(self._mol.atoms):
            intermediate_data = []
            intermediate_data.append("ATOM")
            # i.index would also be an option but is different from original PDBData
            # replaces M.data["index"] = np.arange(self._mol.coordinates.shape[1])
            intermediate_data += [ci, i.name, i.resname, i.segid, i.resid]
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
        dataset_sample_size=-1,
        manual_seed=None,
        shuffle=True,
        sampler=None,
    ):
        """
        :param batch_size:
        :param validation_split:
        :param pin_memory:
        :param dataset_sample_size:
        :param manual_seed:
        :param shuffle:
        :param sampler:
        :return: `torch.utils.data.DataLoader` for training set
        :return: `torch.utils.data.DataLoader` for validation set
        """
        if not hasattr(self, "dataset"):
            self.prepare_dataset()
        valid_size = int(len(self.dataset) * validation_split)
        train_size = len(self.dataset) - valid_size
        dataset = torch.utils.data.TensorDataset(self.dataset.float())
        if manual_seed is not None:
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                dataset,
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(manual_seed),
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                pin_memory=pin_memory,
                sampler=torch.utils.data.RandomSampler(
                    self.train_dataset,
                    generator=torch.Generator().manual_seed(manual_seed),
                ),
            )
        else:
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                dataset, [train_size, valid_size]
            )
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=True,
            )
        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=True,
        )
        return self.train_dataloader, self.valid_dataloader

    def split(self, *args, **kwargs):
        """
        Split :func:`PDBData <molearn.data.PDBData>` into two other :func:`PDBData <molearn.data.PDBData>` objects corresponding to train and valid sets.

        :param manual_seed: manual seed used to split dataset
        :param validation_split: ratio of data to randomly assigned as validation
        :param train_size: if not None, specify number of train structures to be returned
        :param valid_size: if not None, speficy number of valid structures to be returned
        :return: :func:`PDBData <molearn.data.PDBData>` object corresponding to train set
        :return: :func:`PDBData <molearn.data.PDBData>` object corresponding to validation set
        """
        # validation_split=0.1, valid_size=None, train_size=None, manual_seed = None):
        train_dataset, valid_dataset = self.get_datasets(*args, **kwargs)
        train = PDBData()
        valid = PDBData()
        for data in [train, valid]:
            for key in ["_mol", "std", "mean", "filename"]:
                setattr(data, key, getattr(self, key))
        train.dataset = train_dataset
        valid.dataset = valid_dataset
        return train, valid

    def only_test(self):
        """
        prepare a datset without spliting it into training and validation dataset
        """
        if not hasattr(self, "dataset"):
            self.prepare_dataset()
        dataset = self.dataset.float()
        self.indices = np.arange(len(dataset))
        test = PDBData()
        for key in ["_mol", "std", "mean", "filename"]:
            setattr(test, key, getattr(self, key))
        test.dataset = dataset
        return test

    def get_datasets(
        self, validation_split=0.1, valid_size=None, train_size=None, manual_seed=None
    ):
        """
        Create a training and validation set from the imported data

        :param validation_split: ratio of data to randomly assigned as validation
        :param valid_size: if not None, specify number of train structures to be returned
        :param train_size: if not None, speficy number of valid structures to be returned
        :param manual_seed: seed to initialise the random number generator used for splitting the dataset. Useful to replicate a specific split.
        :return: two `torch.Tensor`, for training and validation structures.
        """
        if not hasattr(self, "dataset"):
            self.prepare_dataset()
        dataset = self.dataset.float()
        if train_size is None:
            _valid_size = int(len(self.dataset) * validation_split)
            _train_size = len(self.dataset) - _valid_size
        else:
            _train_size = train_size
            if valid_size is None:
                _valid_size = validation_split * _train_size
            else:
                _valid_size = valid_size
        from torch import randperm

        if manual_seed is not None:
            indices = randperm(
                len(self.dataset), generator=torch.Generator().manual_seed(manual_seed)
            )
        else:
            indices = randperm(len(self.dataset))

        self.indices = indices
        train_dataset = dataset[indices[:_train_size]]
        valid_dataset = dataset[indices[_train_size : _train_size + _valid_size]]
        return train_dataset, valid_dataset

    @property
    def atoms(self):
        return list(np.unique(self.frame().data["name"].values))  # all the atoms

    @property
    def mol(self):
        return self.frame()


if __name__ == "__main__":
    pass
