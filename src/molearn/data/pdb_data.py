import numpy as np
import torch
from copy import deepcopy
import biobox as bb


class PDBData:
    
    def __init__(self, filename=None, fix_terminal=False, atoms=None):
        '''
        Create object enabling the manipulation of multi-PDB files into a dataset suitable for training.
        
        :param filename: None, str or list of strings. If not None, :func:`import_pdb <molearn.data.PDBData.import_pdb>` is called on each filename provided.
        :param fix_terminal: if True, calls :func:`fix_terminal <molearn.data.PDBData.fix_terminal>` after import, and before atomselect
        :param atoms: if not None, calls :func:`atomselect <molearn.data.PDBData.atomselect>`
        '''
        if isinstance(filename, str):
            self.import_pdb(filename)
        elif filename is not None:
            for _filename in filename:
                self.import_pdb(_filename)
        
        if fix_terminal:
            self.fix_terminal()
        if atoms is not None:
            self.atomselect(atoms=atoms)

    def import_pdb(self, filename):
        '''
        Load multiPDB file.
        This command can be called multiple times to load many datasets, if these feature the same number of atoms
        
        :param filename: path to multiPDB file.
        '''
        if not hasattr(self, '_mol'):
            self._mol = bb.Molecule()
        self._mol.import_pdb(filename)
        if not hasattr(self, 'filename'):
            self.filename = []
        self.filename.append(filename)

    def fix_terminal(self):
        '''
        Rename OT1 N-terminal Oxygen to O if terminal oxygens are named OT1 and OT2 otherwise no oxygen will be selected during an atomselect using atoms = ['CA', 'C','N','O','CB']. No template will be found for terminal residue in openmm_loss. Alternative solution is to use atoms = ['CA', 'C', 'N', 'O', 'CB', 'OT1']. instead.
        '''
        ot1 = np.where(self._mol.data['name']=='OT1')[0]
        ot2 = np.where(self._mol.data['name']=='OT2')[0]
        if len(ot1)!=0 and len(ot2)!=0:
            self._mol.data.loc[ot1,'name']='O'

    def atomselect(self, atoms, ignore_atoms=[]):
        '''
        From all imported PDBs, extract only atoms of interest.
        :func:`import_pdb <molearn.data.PDBData.import_pdb>` must have been called at least once, either at class instantiation or as a separate call.
        
        :param atoms: list of atom names, or "no_hydrogen".
        '''
        if atoms == "*":
            _atoms = list(np.unique(self._mol.data["name"].values))
            for to_remove in ignore_atoms:
                if to_remove in _atoms:
                    _atoms.remove(to_remove)
        elif atoms == "no_hydrogen":
            _atoms = self.atoms  # list(np.unique(self._mol.data["name"].values))    #all the atoms
            _plain_atoms = []
            for a in _atoms:
                if a in self._mol.knowledge['atomtype']:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a])
                elif a[:-1] in self._mol.knowledge['atomtype']:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a[:-1]])
                    print(f'Could not find {a}. I am assuing you meant {a[:-1]} instead.')
                elif a[:-2] in self._mol.knowledge['atomtype']:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a[:-2]])
                    print(f'Could not find {a}. I am assuming you meant {a[:-2]} instead.')
                else:
                    _plain_atoms.append(self._mol.knowledge['atomtype'][a])  # if above failed just raise the keyerror
            _atoms = [atom for atom, element in zip(_atoms, _plain_atoms) if element != 'H']
        else:
            _atoms = [_a for _a in atoms if _a not in ignore_atoms]

        _, self._idxs = self._mol.atomselect("*", "*", _atoms, get_index=True)
        self._mol = self._mol.get_subset(self._idxs)

    def prepare_dataset(self):
        '''
        Once all datasets have been loaded, normalise data and convert into `torch.Tensor` (ready for training)
        '''
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
        '''
        generate list of all atoms in dataset, where every line contains [atom name, residue name, resid]
        '''
        if not hasattr(self, 'atominfo'):
            assert hasattr(self, '_mol'), 'You need to call import_pdb before getting atom info'
            self.atominfo = self._mol.get_data(columns=['name', 'resname', 'resid'])
        return self.atominfo

    def frame(self):
        '''
        return `biobox.Molecule` object with loaded data
        '''
        M = bb.Molecule()
        M.coordinates = self._mol.coordinates[[0]]
        M.data = self._mol.data
        M.data['index'] = np.arange(self._mol.coordinates.shape[1])
        M.current = 0
        M.points = M.coordinates.view()[M.current]
        M.properties['center'] = M.get_center()
        return deepcopy(M)

    def get_dataloader(self, batch_size, validation_split=0.1, pin_memory=True, dataset_sample_size=-1, manual_seed=None, shuffle=True, sampler=None):
        '''
        :param batch_size:
        :param validation_split:
        :param pin_memory:
        :param dataset_sample_size:
        :param manual_seed:
        :param shuffle:
        :param sampler:
        :return: `torch.utils.data.DataLoader` for training set
        :return: `torch.utils.data.DataLoader` for validation set
        '''
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
    
    def split(self, *args, **kwargs):
        '''
        Split :func:`PDBData <molearn.data.PDBData>` into two other :func:`PDBData <molearn.data.PDBData>` objects corresponding to train and valid sets.
        
        :param manual_seed: manual seed used to split dataset
        :param validation_split: ratio of data to randomly assigned as validation
        :param train_size: if not None, specify number of train structures to be returned
        :param valid_size: if not None, speficy number of valid structures to be returned
        :return: :func:`PDBData <molearn.data.PDBData>` object corresponding to train set
        :return: :func:`PDBData <molearn.data.PDBData>` object corresponding to validation set
        '''
        # validation_split=0.1, valid_size=None, train_size=None, manual_seed = None):
        train_dataset, valid_dataset = self.get_datasets(*args, **kwargs)
        train = PDBData()
        valid = PDBData()
        for data in [train, valid]:
            for key in ['_mol', 'std', 'mean', 'filename']:
                setattr(data, key, getattr(self, key))
        train.dataset = train_dataset
        valid.dataset = valid_dataset
        return train, valid

    def get_datasets(self, validation_split=0.1, valid_size=None, train_size=None, manual_seed=None):
        '''
        Create a training and validation set from the imported data
        
        :param validation_split: ratio of data to randomly assigned as validation
        :param valid_size: if not None, specify number of train structures to be returned
        :param train_size: if not None, speficy number of valid structures to be returned
        :param manual_seed: seed to initialise the random number generator used for splitting the dataset. Useful to replicate a specific split.
        :return: two `torch.Tensor`, for training and validation structures.
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
            indices = randperm(len(self.dataset), generator=torch.Generator().manual_seed(manual_seed))
        else:
            indices = randperm(len(self.dataset))

        self.indices = indices
        train_dataset = dataset[indices[:_train_size]]
        valid_dataset = dataset[indices[_train_size:_train_size+_valid_size]]
        return train_dataset, valid_dataset

    @property
    def atoms(self):
        return list(np.unique(self._mol.data["name"].values))  # all the atoms
    
    @property
    def mol(self):
        return self.frame()
