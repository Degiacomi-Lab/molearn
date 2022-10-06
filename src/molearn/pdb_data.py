import numpy as np
import torch
from copy import deepcopy
import biobox as bb
import os

class PDBData:
    def __init__(self):
        pass

    def import_pdb(self, filename):
        if not hasattr(self, '_mol'):
            self._mol = bb.Molecule()
        self._mol.import_pdb(filename)
        if not hasattr(self, 'filename'):
            self.filename = []
        self.filename.append(filename) 

    def atomselect(self, atoms, ignore_atoms=[]):
        if atoms == "*":
            _atoms = list(np.unique(self._mol.data["name"].values))
            for to_remove in ignore_atoms:
                if to_remove in _atoms:
                    _atoms.remove(to_remove)
        elif atoms == "no_hydrogen":
            _atoms = self.atoms #list(np.unique(self._mol.data["name"].values))    #all the atoms
            _plain_atoms = []
            for a in _atoms:
                if a in self._mol.knowledge['atomtype']:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a])
                elif a[:-1] in self._mol.knowledge['atomtype']:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a[:-1]])
                else:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a]) # if above failed just raise the keyerror
            _atoms = [atom for atom, element in zip(_atoms, _plain_atoms) if element != 'H']
        else:
            _atoms = [_a for _a in atoms if _a not in ignore_atoms]

        _, self._idxs = self._mol.atomselect("*", "*", _atoms, get_index=True)
        self._mol = self._mol.get_subset(self._idxs)

    def prepare_dataset(self, ):
        if not hasattr(self, 'dataset'):
            assert hasattr(self, '_mol'), 'You need to call import_pdb before preparing the dataset'
            self.dataset = self._mol.coordinates
        
        if not hasattr(self, 'std'):
            self.std = self.dataset.std()
        if not hasattr(self, 'mean'):
            self.mean = self.dataset.mean()
        self.dataset -= self.mean
        self.dataset /= self.std
        self.dataset = torch.from_numpy(self.dataset)
        self.dataset = self.dataset.permute(0,2,1)
        print(f'Dataset.shape: {self.dataset.shape}')
        print(f'mean: {str(self.mean)}, std: {str(self.std)}')

    def get_atominfo(self):
        if not hasattr(self, 'atominfo'):
            assert hasattr(self, '_mol'), 'You need to call import_pdb before getting atom info'
            self.atominfo = self._mol.get_data(columns=['name', 'resname', 'resid'])
        return self.atominfo

    def frame(self):
        M = bb.Molecule()
        M.coordinates = self._mol.coordinates[[0]]
        M.data = self._mol.data
        M.data['index'] = np.arange(self._mol.coordinates.shape[1])
        M.current = 0
        M.points = M.coordinates.view()[M.current]
        M.properties['center'] = M.get_center()
        return deepcopy(M)

    def get_dataloader(self, batch_size, validation_split=None, pin_memory=True, dataset_sample_size=-1):
        if not hasattr(self, 'dataset'):
            self.prepare_dataset()
        if validation_split is not None:
            valid_size = int(len(self.dataset)*validation_split)
            train_size = len(self.dataset) - valid_size
            dataset = torch.utils.data.TensorDataset(self.dataset.float())
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
            self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, pin_memory=pin_memory)
            return self.train_dataloader, self.valid_dataloader
        else:
            self.train_dataset = torch.utils.data.TensorDataset(dataset.float())
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
            return self.train_datajloader

    @property
    def atoms(self):
        return list(np.unique(self._mol.data["name"].values))    #all the atoms
    
    @property
    def mol(self):
        return frame
        



