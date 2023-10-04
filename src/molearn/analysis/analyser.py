# Copyright (c) 2021 Venkata K. Ramaswamy, Samuel C. Musson, Chris G. Willcocks, Matteo T. Degiacomi
#
# Molearn is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# molearn is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molearn ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Authors: Matteo Degiacomi, Samuel Musson

from copy import deepcopy
import numpy as np
import torch.optim
try:
    from modeller import *
    from modeller.scripts import complete_pdb
except Exception as e:
    print('Error importing modeller: ')
    print(e)
    
try:
    from ..scoring import Parallel_DOPE_Score
except ImportError as e:
    print('Import Error captured while trying to import Parallel_DOPE_Score, it is likely that you dont have Modeller installed')
    print(e)
try:
    from ..scoring import Parallel_Ramachandran_Score
except ImportError as e:
    print('Import Error captured while trying to import Parallel_Ramachandran_Score, it is likely that you dont have cctbx/iotbx installed')
    print(e)
from ..data import PDBData

from ..utils import as_numpy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pickle

class MolearnAnalysis:
    '''
    This class provides methods dedicated to the quality analysis of a
    trained model.
    '''
    
    def __init__(self):
        self._datasets = {}
        self._encoded = {}
        self._decoded = {}
        self.surfaces = {}
        self.analysis_results = {}
        self.xyvals = {}
        self.batch_size = 1
        self.processes = 1
        self._network = None
        self.grad = False

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        assert self._network is None, 'you will be overriding network?'
        from hashlib import sha1
        h = sha1()
        h.update(str(value).encode())
        if hasattr(self, 'network_architecture_hash'):
            assert self.network_architecture_hash == h.hexdigest()
        else:
            self.network_architecture_hash = h.hexdigest()
        self._network = value
        self._network.eval()
        self.device = next(self._network.parameters()).device

    def save_pickle(self, filename, ):
        assert isinstance(filename, str)
        f = open(filename,'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def from_pickle(cls, filename):
        assert isinstance(filename, str)
        f = open(filename,'rb')
        y = pickle.load(f)
        assert isinstance(y, cls)
        return y
    


    def set_network(self, network):
        '''
        :param network: a trained neural network defined in :func:`molearn.models <molearn.models>`
        '''
        self.network = network
        #self.network.eval()
        #self.device = next(network.parameters()).device
        #self.network_architecture_hash = hash(str())

    def get_dataset(self, key):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        '''
        return self._datasets[key]

    def set_dataset(self, key, data, atomselect="*"):
        '''
        :param data: :func:`PDBData <molearn.data.PDBData>` object containing atomic coordinates
        :param str key: label to be associated with data
        :param list/str atomselect: list of atom names to load, or '*' to indicate that all atoms are loaded.
        '''
        if isinstance(data, str) and data.endswith('.pdb'):
            d = PDBData()
            d.import_pdb(data)
            d.atomselect(atomselect)
            d.prepare_dataset()
            _data = d
        elif isinstance(data, PDBData):
            _data = data
        else:
            raise NotImplementedError('data should be an PDBData instance')
        for _key, dataset in self._datasets.items():
            assert dataset.shape[2]== _data.dataset.shape[2] and dataset.shape[1]==_data.dataset.shape[1], f'number of d.o.f differes: {key} has shape {_data.shape} while {_key} has shape {dataset.shape}'
        
        if hasattr(self, '_datasets_hashed'):
            if key in self._datasets_hashed:
                import hashlib
                assert self._datasets_hashed[key]==hashlib.sha1(np.ascontiguousarray(_data.dataset.float().numpy(force=True))).hexdigest()

        self._datasets[key] = _data.dataset.float()
        if not hasattr(self, 'meanval'):
            self.meanval = _data.mean
        if not hasattr(self, 'stdval'):
            self.stdval = _data.std
        if not hasattr(self, 'atoms'):
            self.atoms = _data.atoms
        if not hasattr(self, 'mol'):
            self.mol = _data.frame()
        if not hasattr(self, 'shape'):
            self.shape = (_data.dataset.shape[1], _data.dataset.shape[2])

    def get_encoded_grad(self, key):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :return: array containing the encoding in latent space of dataset associated with key
        '''
        if key not in self._encoded:
            assert key in self._datasets, f'key {key} does not exist in internal _datasets or in _latent_coords, add it with MolearnAnalysis.set_latent_coords(key, torch.Tensor) '\
            'or add the corresponding dataset with MolearnAnalysis.set_dataset(name, PDBDataset)'
            dataset = self.get_dataset(key)
            batch_size = self.batch_size
            encoded = None
            for i in tqdm(range(0, dataset.shape[0], batch_size), desc=f'encoding {key}'):
                z = self.network.encode(dataset[i:i+batch_size].to(self.device)).cpu().detach()
                if encoded is None:
                    encoded = torch.empty(dataset.shape[0], z.shape[1], z.shape[2])
                encoded[i:i+batch_size] = z
            self._encoded[key] = encoded
                
        return self._encoded[key]



    def get_encoded(self, key, grad=False):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :return: array containing the encoding in latent space of dataset associated with key
        '''
        if grad or self.grad:
            return self.get_encoded_grad(key)
        if key not in self._encoded:
            assert key in self._datasets, f'key {key} does not exist in internal _datasets or in _latent_coords, add it with MolearnAnalysis.set_latent_coords(key, torch.Tensor) '\
            'or add the corresponding dataset with MolearnAnalysis.set_dataset(name, PDBDataset)'
            with torch.no_grad():
                dataset = self.get_dataset(key)
                batch_size = self.batch_size
                encoded = None
                for i in tqdm(range(0, dataset.shape[0], batch_size), desc=f'encoding {key}'):
                    z = self.network.encode(dataset[i:i+batch_size].to(self.device)).cpu()
                    if encoded is None:
                        encoded = torch.empty(dataset.shape[0], z.shape[1], z.shape[2])
                    encoded[i:i+batch_size] = z
                self._encoded[key] = encoded
                
        return self._encoded[key]

    def set_encoded(self, key, coords):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        '''
        self._encoded[key] = torch.tensor(coords).float()

    def get_decoded(self, key):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        '''
        if key not in self._decoded:
            encoded = self.get_encoded(key)
            with torch.no_grad():
                batch_size = self.batch_size
                decoded = torch.empty(encoded.shape[0], *self.shape).float()
                for i in tqdm(range(0, encoded.shape[0], batch_size), desc=f'Decoding {key}'):
                    decoded[i:i+batch_size] = self.network.decode(encoded[i:i+batch_size].to(self.device))[:, :, :self.shape[1]].cpu()
                self._decoded[key] = decoded
        return self._decoded[key]

    def set_decoded(self, key, structures):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        '''
        self._decoded[key] = structures

    def num_trainable_params(self):
        '''
        :return: number of trainable parameters in the neural network previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_network>`
        '''
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def get_error(self, key, align=True):
        '''
        Calculate the reconstruction error of a dataset encoded and decoded by a trained neural network.
        
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param bool align: if True, the RMSD will be calculated by finding the optimal alignment between structures
        :return: 1D array containing the RMSD between input structures and their encoded-decoded counterparts
        '''

        _key = f'error_{key}_align{align}'
        if _key in self.analysis_results:
            return self.analysis_results[_key]

        dataset = self.get_dataset(key)
        z = self.get_encoded(key)
        decoded = self.get_decoded(key)

        err = []
        m = deepcopy(self.mol)
        for i in range(dataset.shape[0]):
            crd_ref = as_numpy(dataset[i].permute(1,0).unsqueeze(0))*self.stdval + self.meanval
            crd_mdl = as_numpy(decoded[i].permute(1,0).unsqueeze(0))[:, :dataset.shape[2]]*self.stdval + self.meanval  # clip the padding of models  
            # use Molecule Biobox class to calculate RMSD
            if align:
                m.coordinates = deepcopy(crd_ref)
                m.set_current(0)
                m.add_xyz(crd_mdl[0])
                rmsd = m.rmsd(0, 1)
            else:
                rmsd = np.sqrt(np.sum((crd_ref.flatten()-crd_mdl.flatten())**2)/crd_mdl.shape[1])  # Cartesian L2 norm

            err.append(rmsd)
        
        self.analysis_results[_key] = np.array(err)
        return self.analysis_results[_key]

    def get_dope(self, key, refine=True, **kwargs):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param bool refine: if True, refine structures before calculating DOPE score
        :return: dictionary containing DOPE score of dataset, and its decoded counterpart
        '''

        _key = f'dope_{key}_refine{refine}'
        if _key in self.analysis_results:
            return self.analysis_results[_key]

        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)
        
        dope_dataset = self.get_all_dope_score(dataset, refine=refine, **kwargs)
        dope_decoded = self.get_all_dope_score(decoded, refine=refine, **kwargs)

        self.analysis_results[_key] = dict(dataset_dope=dope_dataset, 
                    decoded_dope=dope_decoded)
        return self.analysis_results[_key]

    def get_ramachandran(self, key):
        '''
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        '''
        _key = f'ramachandran_{key}'
        if _key in self.analysis_results:
            return self.analysis_results[_key]
        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)

        ramachandran = {f'dataset_{key}':value for key, value in self.get_all_ramachandran_score(dataset).items()}
        ramachandran.update({f'decoded_{key}':value for key, value in self.get_all_ramachandran_score(decoded).items()})
        self.analysis_results[_key] = ramachandran
        return self.analysis_results[_key]

    def setup_grid(self, samples=64, bounds_from=None, bounds=None, padding=0.1, grid_key='grid'):
        '''
        Define a NxN point grid regularly sampling the latent space.
        
        :param int samples: grid size (build a samples x samples grid)
        :param str/list bounds_from: Name(s) of datasets to use as reference, either as single string, a list of strings, or 'all'
        :param tuple/list bounds: tuple (xmin, xmax, ymin, ymax) or None
        :param float padding: define size of extra spacing around boundary conditions (as ratio of axis dimensions)
        :param str grid_key: The key that will be associated with this grid
        '''
        assert grid_key not in self._encoded, f'{grid_key} already exists in encoded, try using a different value for grid_key'
        assert grid_key not in self._datasets, f'{grid_key} already exists in datasets, try using a different value for grid_key'
        assert grid_key not in self._decoded, f'{grid_key} already exists in decoded, try using a different value for grid_key'
        if bounds is None:
            if bounds_from is None:
                bounds_from = "all"
            
            bounds = self._get_bounds(bounds_from, exclude=[grid_key, f'{grid_key}_decoded'])
        
        bx = (bounds[1]-bounds[0])*padding
        by = (bounds[3]-bounds[2])*padding
        self.xyvals[grid_key] = (np.linspace(bounds[0]-bx, bounds[1]+bx, samples), np.linspace(bounds[2]-by, bounds[3]+by, samples))
        self.n_samples = samples
        meshgrid = np.meshgrid(*self.xyvals[grid_key])
        stack = np.stack(meshgrid, axis=2).reshape(-1, 2, 1)
        self.set_encoded(grid_key, stack)
        
        return grid_key

    def as_numpy(self,x):
        return as_numpy(x)

    def _get_bounds(self, bounds_from, exclude=['grid', 'grid_decoded']):
        '''        
        :param bounds_from: keys of datasets to be considered for identification of boundaries in latent space
        :param exclude: keys of dataset not to consider
        :return: four scalars as edges of x and y axis: xmin, xmax, ymin, ymax
        '''
        if isinstance(exclude, str):
            exclude = [exclude,]
            
        if bounds_from == 'all':
            bounds_from = [key for key in self._datasets.keys() if key not in exclude]
        elif isinstance(bounds_from, str):
            bounds_from = [bounds_from,]
        
        xmin, ymin, xmax, ymax = [], [], [], []
        for key in bounds_from:
            z = self.get_encoded(key)
            xmin.append(z[:, 0].min())
            ymin.append(z[:, 1].min())
            xmax.append(z[:, 0].max())
            ymax.append(z[:, 1].max())
            
        xmin, ymin = min(xmin), min(ymin)
        xmax, ymax = max(xmax), max(ymax)
        return xmin, xmax, ymin, ymax

    def scan_error_from_target(self, target_key, grid_key ='grid', index=None, align=True):
        '''
        Calculate landscape of RMSD vs single target structure. Target should be previously loaded datset containing a single conformation.  
  
        :param str target_key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param str grid_key: key pointing to the grid to scan error from
        :param int index: index of conformation to be selected from dataset containing multiple conformations.
        :param bool align: if True, structures generated from the grid are aligned to target prior RMSD calculation.
        :return: RMSD latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        '''
        s_key = f'RMSD_from_{target_key}_in_{grid_key}' if index is None else f'RMSD_from_{target_key}_index_{index}_in_{grid_key}'
        if s_key not in self.surfaces:
            assert grid_key in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            target = self.get_dataset(target_key) if index is None else self.get_dataset(target_key)[index].unsqueeze(0)
            if target.shape[0] != 1:
                msg = f'dataset {target_key} shape is {target.shape}. \
A dataset with a single conformation is expected.\
Either pass a key that points to a single structure or pass the index of the \
structure you want, e.g., analyser.scan_error_from_target(key, index=0)'
                raise Exception(msg)
            
            decoded = self.get_decoded(grid_key)
            if align:
                crd_ref = as_numpy(target.permute(0, 2, 1))*self.stdval
                crd_mdl = as_numpy(decoded.permute(0, 2, 1))*self.stdval
                m = deepcopy(self.mol)
                m.coordinates = np.concatenate([crd_ref, crd_mdl])
                m.set_current(0)
                rmsd = np.array([m.rmsd(0, i) for i in range(1, len(m.coordinates))])
            else:
                rmsd = (((decoded-target)*self.stdval)**2).sum(axis=1).mean(axis=-1).sqrt()
            self.surfaces[s_key] = as_numpy(rmsd.reshape(self.n_samples, self.n_samples))
            
        xvals, yvals = self.xyvals[grid_key]
        return self.surfaces[s_key], xvals, yvals

    
    def encode_best(self, target_key, grid_key = 'grid'):
        assert grid_key in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
        encoded_key = f'{target_key}_best_in_{grid_key}'
        if encoded_key in self._encoded:
            return self.get_encoded(encoded_key), encoded_key
        m = deepcopy(self.mol)
        decoded = self.get_decoded(grid_key).permute(0,2,1)*self.stdval
        shape = (int(np.sqrt(decoded.shape[0])), int(np.sqrt(decoded.shape[0])), decoded.shape[1], decoded.shape[2])
        assert shape[0]==64, 'only implemented for 64x64 grid'
        grid_structures = decoded.reshape(*shape)
        dataset = as_numpy(self.get_dataset(target_key).permute(0,2,1)*self.stdval)
        index = []
        for i in tqdm(range(dataset.shape[0]), desc=f'encode best'):
            index.append(self.encode_one_best(dataset[[i]], grid_structures,m))
        best_z = self._encoded[grid_key][index]
        #best_z = self._encoded[grid_key][[self.encode_one_best(dataset[[i]], grid_structures,m) for i in range(dataset.shape[0])]]
        self.set_encoded(encoded_key, best_z)
        return self.get_encoded(encoded_key), encoded_key


    def encode_one_best(self,x, grid_structures,m):
        def calc_rmsd(x,y):
            crd_ref = x
            crd_mdl = y
            m.coordinates = np.concatenate([crd_ref, crd_mdl])
            m.set_current(0)
            return m.rmsd(0,1)
        
        rmsd = np.full([64,64], np.nan)
        for i in range(0,64,8):
            for j in range(0,64,8):
                if np.isnan(rmsd[i,j]):
                    #from IPython import embed
                    #embed(header='huh?')
                    rmsd[i,j] = calc_rmsd(x,grid_structures[[i],[j]])
        best = np.argpartition(rmsd.reshape(-1),8)[:8]
        de = 8
        s = 4
        for b in best:
            index = np.unravel_index(b,(64,64))
            for i in range(max(0,index[0]-de),min(index[0]+de+1,64),s):
                for j in range(max(0,index[1]-de),min(index[1]+de+1,64),s):
                    if np.isnan(rmsd[i,j]):
                        rmsd[i,j] = calc_rmsd(x,grid_structures[[i],[j]])
        best = np.argpartition(rmsd.reshape(-1),8)[:8]
        de = 4
        s = 2
        for b in best:
            index = np.unravel_index(b,(64,64))
            for i in range(max(0,index[0]-de),min(index[0]+de+1,64),s):
                for j in range(max(0,index[1]-de),min(index[1]+de+1,64),s):
                    if np.isnan(rmsd[i,j]):
                        rmsd[i,j] = calc_rmsd(x,grid_structures[[i],[j]])
        best = np.argpartition(rmsd.reshape(-1),8)[:8]
        de = 2
        s = 1
        for b in best:
            index = np.unravel_index(b,(64,64))
            for i in range(max(0,index[0]-de),min(index[0]+de+1,64),s):
                for j in range(max(0,index[1]-de),min(index[1]+de+1,64),s):
                    if np.isnan(rmsd[i,j]):
                        rmsd[i,j] = calc_rmsd(x,grid_structures[[i],[j]])
        return np.nanargmin(rmsd) #MA._encoded
        

    def scan_error(self, grid_key = 'grid', RMSD_key='Network_RMSD', z_drift_key='Network_z_drift'):
        '''
        Calculate RMSD and z-drift on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.
        
        :param str grid_key: Key to the grid to scan error over
        :param str RMSD_key: label for RMSD dataset
        :param str z_drift_key: label for z-drift dataset
        :return: input-to-decoded RMSD latent space NxN surface
        :return: z-drift latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        '''
        s_key = f'{grid_key}_{RMSD_key}'
        z_key = f'{grid_key}_{z_drift_key}'
        if s_key not in self.surfaces:
            assert grid_key in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded(grid_key)            # decode grid 
            # self.set_dataset('grid_decoded', decoded)   # add back as dataset w. different name
            self._datasets[f'{grid_key}_decoded'] = decoded
            decoded_2 = self.get_decoded(f'{grid_key}_decoded')  # encode, and decode a second time
            grid = self.get_encoded(f'{grid_key}')               # retrieve original grid
            grid_2 = self.get_encoded(f'{grid_key}_decoded')     # retrieve decoded encoded grid

            rmsd = (((decoded-decoded_2)*self.stdval)**2).sum(axis=1).mean(axis=-1).sqrt()
            z_drift = ((grid-grid_2)**2).mean(axis=2).mean(axis=1).sqrt()

            self.surfaces[s_key] = rmsd.reshape(self.n_samples, self.n_samples).numpy()
            self.surfaces[z_key] = z_drift.reshape(self.n_samples, self.n_samples).numpy()
        
        xvals, yvals = self.xyvals[grid_key]
            
        return self.surfaces[s_key], self.surfaces[z_key], xvals, yvals

    def _ramachandran_score(self, frame):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'ramachandran_score_class'):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(self.mol, self.processes)
        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1, 0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f, torch.Tensor):
            f = f.data.cpu().numpy()
        
        return self.ramachandran_score_class.get_score(f*self.stdval)
        # nf, na, no, nt = self.ramachandran_score_class.get_score(f*self.stdval)
        # return {'favored':nf, 'allowed':na, 'outliers':no, 'total':nt}

    def _dope_score(self, frame, refine=True, **kwargs):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'dope_score_class'):
            self.dope_score_class = Parallel_DOPE_Score(self.mol, self.processes)

        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1, 0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f,torch.Tensor):
            f = f.data.cpu().numpy()

        return self.dope_score_class.get_score(f*self.stdval, refine=refine, **kwargs)

    def get_all_ramachandran_score(self, tensor):
        '''
        Calculate Ramachandran score of an ensemble of atomic conrdinates.
        
        :param tensor:
        '''
        rama = dict(favored=[], allowed=[], outliers=[], total=[])
        results = []
        for f in tensor:
            results.append(self._ramachandran_score(f))
        for r in tqdm(results,desc='Calc rama'):
            favored, allowed, outliers, total = r.get()
            rama['favored'].append(favored)
            rama['allowed'].append(allowed)
            rama['outliers'].append(outliers)
            rama['total'].append(total)
        return {key:np.array(value) for key, value in rama.items()}       

    def get_all_dope_score(self, tensor, refine=True):
        '''
        Calculate DOPE score of an ensemble of atom coordinates.

        :param tensor:
        :param bool refine: if True, return DOPE score of input and output structure after refinement
        '''
        results = []
        for f in tensor:
            results.append(self._dope_score(f, refine=refine))
        results = np.array([r.get() for r in tqdm(results, desc='Calc Dope')])
        return results

    def reference_dope_score(self, frame):
        '''
        :param numpy.array frame: array with shape [1, N, 3] with Cartesian coordinates of atoms
        :return: DOPE score
        '''
        self.mol.coordinates = deepcopy(frame)
        self.mol.write_pdb('tmp.pdb', split_struc=False)
        env = Environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        mdl = complete_pdb(env, 'tmp.pdb')
        atmsel = Selection(mdl.chains[0])
        score = atmsel.assess_dope()
        return score

    def scan_dope(self, grid_key='grid', refine=True, **kwargs):
        '''
        Calculate DOPE score on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.
        
        :param str grid_key: What grid to calculate dope over
        :param bool refine: if True, structures generated will be energy minimised before DOPE scoring
        :return: DOPE score latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        '''
        
        if refine=='both':
            key = f"DOPE_both"
        elif refine:
            key = "DOPE_refined"
        else:
            key = "DOPE_unrefined"
        key = f'{grid_key}_{key}'
        
        if key not in self.surfaces:
            assert grid_key in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded(grid_key)
            result = self.get_all_dope_score(decoded, refine=refine, **kwargs)
            if refine=='both':
                self.surfaces[key] = as_numpy(result.reshape(self.n_samples, self.n_samples, 2))
            else:
                self.surfaces[key] = as_numpy(result.reshape(self.n_samples, self.n_samples))
            
        xvals, yvals = self.xyvals[grid_key]
        return self.surfaces[key], xvals, yvals

    def scan_ramachandran(self, grid_key='grid'):
        '''
        Calculate Ramachandran scores on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.
        Saves four surfaces in memory, with keys 'Ramachandran_favored', 'Ramachandran_allowed', 'Ramachandran_outliers', and 'Ramachandran_total'.

        :return: Ramachandran_favoured latent space NxN surface (ratio of residues in favourable conformation)
        :return: x-axis values
        :return: y-axis values
        '''
        keys = {i:f'{grid_key}_Ramachandran_{i}' for i in ('favored', 'allowed', 'outliers', 'total')}
        if list(keys.values())[0] not in self.surfaces:
            assert grid_key in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded(grid_key)
            rama = self.get_all_ramachandran_score(decoded)
            for key, value in rama.items():
                self.surfaces[keys[key]] = value
        xvals, yvals = self.xyvals[grid_key]

        return self.surfaces[keys['favored']], xvals, yvals
  
    def scan_custom(self, fct, params, grid_key='grid', surface_key = 'custom'):
        '''
        Generate a surface coloured as a function of a user-defined function.
        
        :param fct: function taking atomic coordinates as input, an optional list of parameters, and returning a single value.
        :param list params: parameters to be passed to function f. If no parameter is needed, pass an empty list.
        :param str grid_key: name of grid to scan over
        :param str surface_key: name of the dataset generated by this function scan
        :return: latent space NxN surface, evaluated according to input function
        :return: x-axis values
        :return: y-axis values
        '''

        skey = f'{grid_key}_{surface_key}'
        if skey in self.surfaces:
            xvals, yvals = self.xyvals[grid_key]
            return self.surfaces[skey], xvals, yvals
        decoded = self.get_decoded(f'{grid_key}_{surface_key}')
        results = []
        for i, j in enumerate(decoded):
            s = (j.view(1, 3, -1).permute(0, 2, 1)*self.stdval).numpy()
            results.append(fct(s, *params))
        self.surfaces[skey] = np.array(results).reshape(self.n_samples, self.n_samples)
        xvals, yvals = self.xyvals[grid_key]
        
        return self.surfaces[skey], xvals, yvals

    def generate(self, crd):
        '''
        Generate a collection of protein conformations, given coordinates in the latent space.
        
        :param numpy.array crd: coordinates in the latent space, as a (Nx2) array
        :return: collection of protein conformations in the Cartesian space (NxMx3, where M is the number of atoms in the protein)
        ''' 
        with torch.no_grad():
            z = torch.tensor(crd.transpose(1, 2, 0)).float()
            key = list(self._datasets)[0]
            s = self.network.decode(z)[:, :, :self._datasets[key].shape[2]].numpy().transpose(0, 2, 1)

        return s*self.stdval + self.meanval

    def __getstate__(self):
        return_dict = {}
        ignore =  ['dope_score_class', 'ramachandran_score_class', '_network']
        for key, value in dict(self.__dict__).items():
            if key in ignore:
                continue
            if key in ['_datasets', '_decoded']:
                import hashlib
                return_dict[f'{key}_hashed'] = {k:hashlib.sha1(np.ascontiguousarray(v.numpy(force=True))).hexdigest() for k,v in value.items()}
            else:
                return_dict[key] = value
        return return_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._datasets = {}
        self._decoded = {}
        self._network = None


