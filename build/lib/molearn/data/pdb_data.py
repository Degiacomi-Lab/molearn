import numpy as np
import torch
from copy import deepcopy
import biobox as bb
#import os

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

    def fix_terminal(self):
        ot1 = np.where(self._mol.data['name']=='OT1')[0]
        ot2 = np.where(self._mol.data['name']=='OT2')[0]
        oxt = np.where(self._mol.data['name']=='OXT')[0]
        resids = []
        if len(ot1)!=0 and len(ot2)!=0:
            self._mol.data.loc[ot1,'name']='O'
        if len(ot1)!=0:
            for i in ot1:
                resids.append((self._mol.data['resid'][i], self._mol.data['resname'][i]))
        if len(oxt)!=0:
            for i in oxt:
                resids.append((self._mol.data['resid'][i], self._mol.data['resname'][i]))

        #for resid, resname in resids:
            #resname = self._mol.data['resname'][resid]
            #if len(resname)==3:
            #    self._mol.data.loc[self._mol.data.resid.eq(resid), 'resname']=f'C{resname}'
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
            self.dataset = self._mol.coordinates.copy()
        
        if not hasattr(self, 'std'):
            self.std = self.dataset.std()
        if not hasattr(self, 'mean'):
            self.mean = self.dataset.mean()
        self.dataset -= self.mean
        self.dataset /= self.std
        self.dataset = torch.from_numpy(self.dataset).float()
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

    def get_dataloader(self, batch_size, validation_split=0.1, pin_memory=True, dataset_sample_size=-1, manual_seed=None,shuffle=True, sampler=None):
        if not hasattr(self, 'dataset'):
            self.prepare_dataset()
        valid_size = int(len(self.dataset)*validation_split)
        train_size = len(self.dataset) - valid_size
        dataset = torch.utils.data.TensorDataset(self.dataset.float())
        if manual_seed is not None:
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(manual_seed))
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, pin_memory=pin_memory,
                                                                sampler=torch.utils.data.RandomSampler(self.train_dataset,generator=torch.Generator().manual_seed(manual_seed)))
        else:
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
        self.valid_dataloader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size, pin_memory=pin_memory,shuffle=True)
        return self.train_dataloader, self.valid_dataloader

    def get_datasets(self, validation_split=0.1, valid_size=None, train_size=None, manual_seed = None):
        '''
            returns torch.Tensor for training and validation structures.
        '''
        if not hasattr(self, 'dataset'):
            self.prepare_dataset()
        dataset = self.dataset.float()
        if train_size is None:
            _valid_size = int(len(self.dataset)*validation_split)
            _train_size = len(self.dataset) - _valid_size
        else:
            _train_size = train_size
            if valid_size is None:
                _valid_size = validation_split*_train_size
            else:
                _valid_size = valid_size
        from torch import randperm
        if manual_seed is not None:
            indices = randperm(len(self.dataset), generator = torch.Generator().manual_seed(manual_seed))
        else:
            indices = randperm(len(self.dataset))
        train_dataset = dataset[indices[:_train_size]]
        valid_dataset = dataset[indices[_train_size:_train_size+_valid_size]]
        return train_dataset, valid_dataset
        


    @property
    def atoms(self):
        return list(np.unique(self._mol.data["name"].values))    #all the atoms
    
    @property
    def mol(self):
        return self.frame()
        



