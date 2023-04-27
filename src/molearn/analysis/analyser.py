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
# Author: Matteo Degiacomi

from copy import deepcopy
import numpy as np
import torch.optim

from modeller import *
from modeller.scripts import complete_pdb

from ..scoring import Parallel_DOPE_Score, Parallel_Ramachandran_Score
from ..data import PDBData

from ..utils import as_numpy

import warnings
warnings.filterwarnings("ignore")


class MolearnAnalysis(object):
    
    def __init__(self,):
        self._datasets = {}
        self._encoded = {}
        self._decoded = {}
        self.surfaces = {}
        self.batch_size = 1

    def set_network(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device

    def get_dataset(self, key):
        return self._datasets[key]

    def set_dataset(self, key, data, atomselect="*"):
        if isinstance(data, str) and data.endswith('.pdb'):
            d = PDBData()
            d.import_pdb(data)
            d.atomselect(atomselect)
            d.prepare_dataset()
            _data = d
        elif isinstance(data, PDBData):
            _data = data
        else:
            raise NotImplementedError('No other data typethan PDBData has been implemented for this method')
        for _key, dataset in self._datasets.items():
            assert dataset.shape[2]== _data.dataset.shape[2] and dataset.shape[1]==_data.dataset.shape[1], f'number of d.o.f differes: {key} has shape {_data.shape} while {_key} has shape {dataset.shape}'
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

    def get_encoded(self, key):
        if key not in self._encoded:
            assert key in self._datasets, f'key {key} does not exist in internal _datasets or in _latent_coords, add it with MolearnAnalysis.set_latent_coords(key, torch.Tensor) '\
            'or add the corresponding dataset with MolearnAnalysis.set_dataset(name, PDBDataset)'
            with torch.no_grad():
                dataset = self.get_dataset(key)
                batch_size = self.batch_size
                encoded = None
                for i in range(0, dataset.shape[0], batch_size):
                    z = self.network.encode(dataset[i:i+batch_size].to(self.device)).cpu()
                    if encoded is None:
                        encoded = torch.empty(dataset.shape[0], z.shape[1], z.shape[2])
                    encoded[i:i+batch_size] = z
                self._encoded[key] = encoded
        return self._encoded[key]

    def set_encoded(self, key, coords):
        self._encoded[key] = torch.tensor(coords).float()

    def get_decoded(self, key):
        if key not in self._decoded:
            with torch.no_grad():
                batch_size = self.batch_size
                encoded = self.get_encoded(key)
                decoded = torch.empty(encoded.shape[0], *self.shape).float()
                for i in range(0, encoded.shape[0], batch_size):
                    decoded[i:i+batch_size] = self.network.decode(encoded[i:i+batch_size].to(self.device))[:,:,:self.shape[1]].cpu()
                self._decoded[key] = decoded
        return self._decoded[key]

    def set_decoded(self, key, structures):
        self._decoded[key] = structures

    def num_trainable_params(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def get_error(self, key, align=False):
        '''
        Calculate the reconstruction error of a dataset encoded and decoded by a trained neural network
        '''

        dataset = self.get_dataset(key)
        z = self.get_encoded(key)
        decoded = self.get_decoded(key)

        err = []
        for i in range(dataset.shape[0]):
            crd_ref = as_numpy(dataset[i].permute(1,0).unsqueeze(0))*self.stdval + self.meanval
            crd_mdl = as_numpy(decoded[i].permute(1,0).unsqueeze(0))[:, :dataset.shape[2]]*self.stdval + self.meanval #clip the padding of models  

            if align: # use Molecule Biobox class to calculate RMSD
                self.mol.coordinates = deepcopy(crd_ref)
                self.mol.set_current(0)
                self.mol.add_xyz(crd_mdl[0])
                rmsd = self.mol.rmsd(0, 1)
            else:
                rmsd = np.sqrt(np.sum((crd_ref.flatten()-crd_mdl.flatten())**2)/crd_mdl.shape[1]) # Cartesian L2 norm

            err.append(rmsd)

        return np.array(err)


    def get_dope(self, key, refined=True):

        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)
        
        dope_dataset = self.get_all_dope_score(dataset)
        dope_decoded = self.get_all_dope_score(decoded)

        if refined:
            return dict(dataset_dope_unrefined = dope_dataset[0], 
                        dataset_dope_refined = dope_dataset[1],
                        decoded_dope_unrefined = dope_decoded[0],
                        decoded_dope_refined = dope_dataset[1])
        else:
            return dict(dataset_dope_unrefined = dope_dataset, 
                        decoded_dope_unrefined = dope_decoded,)

    def get_ramachandran(self, key):
        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)

        ramachandran = {f'dataset_{key}':value for key, value in self.get_all_ramachandran_score(dataset).items()}
        ramachandran.update({f'decoded_{key}':value for key, value in self.get_all_ramachandran_score(decoded).items()})
        return ramachandran

    def setup_grid(self, samples=64, bounds_from = None, bounds = None, padding=0.1):
        '''
        :param bounds_from: str, list of strings, or 'all'
        :param bounds: tuple (xmin, xmax, ymin, ymax) or None
        '''
        key = 'grid'
        if bounds_from is not None:
            bounds = self._get_bounds(bounds_from, exclude = key)
        bx = (bounds[1]-bounds[0])*padding
        by = (bounds[3]-bounds[2])*padding
        self.xvals = np.linspace(bounds[0]-bx, bounds[1]+bx, samples)
        self.yvals = np.linspace(bounds[2]-by, bounds[3]+by, samples)
        self.n_samples = samples
        meshgrid = np.meshgrid(self.xvals, self.yvals)
        stack = np.stack(meshgrid, axis=2).reshape(-1,1,2)
        self.set_encoded(key, stack)
        return key

    def _get_bounds(self, bounds_from, exclude = ['grid', 'grid_decoded']):
        if isinstance(exclude, str):
            exclude = [exclude,]
        if bounds_from == 'all':
            bounds_from = [key for key in self._encoded.keys() if key not in exclude]
        elif isinstance(bounds_from,str):
            bounds_from = [bounds_from,]
        xmin, ymin, xmax, ymax = [],[],[],[]
        for key in bounds_from:
            z = self.get_encoded(key)
            xmin.append(z[:,0].min())
            ymin.append(z[:,1].min())
            xmax.append(z[:,0].max())
            ymax.append(z[:,1].max())
        xmin, ymin = min(xmin), min(ymin)
        xmax, ymax = max(xmax), max(ymax)
        return xmin, xmax, ymin, ymax

    def scan_error_from_target(self, key, index=None):
        '''
        experimental function, creating a coloured landscape of RMSD vs single target structure
        target should be a Tensor of a single protein stucture loaded via load_test
        :param key: key pointing to a single structure that has been with MolearnAnalysis.set_dataset(key, structure)
        :param index: Default None, if key corresponds to multiple structures then an index is required to use only one structure as target.
        '''
        s_key = f'RMSD_from_{key}' if index is None else f'RMSD_from_{key}_index_{index}'
        if s_key not in self.surfaces:
            assert 'grid' in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            target = self.get_dataset(key) if index is None else self.get_dataset(key)[index]
            assert target.shape[0] == 1
            decoded = self.get_decoded('grid')
            rmsd = (((decoded-target)*self.stdval)**2).sum(axis=1).mean(axis=-1).sqrt()
            self.surfaces[s_key] = rmsd.reshape(self.n_samples, self.n_samples).numpy()
        return self.surfaces[s_key], self.xvals, self.yvals

    def scan_error(self):
        '''
        grid sample the latent space on a samples x samples grid (64 x 64 by default).
        Boundaries are defined by training set projections extrema, plus/minus 10%
        '''
        s_key = 'Network_RMSD'
        z_key = 'Network_z_drift'
        if s_key not in self.surfaces:
            assert 'grid' in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded('grid')           # decode grid 
            #self.set_dataset('grid_decoded', decoded)    # add back as dataset w. different name
            self._datasets['grid_decoded'] = decoded
            decoded_2 = self.get_decoded('grid_decoded') # encode, and decode a second time
            grid = self.get_encoded('grid')              # retrieve original grid
            grid_2 = self.get_encoded('grid_decoded')    # retrieve decoded encoded grid

            rmsd = (((decoded-decoded_2)*self.stdval)**2).sum(axis=1).mean(axis=-1).sqrt()
            z_drift = ((grid-grid_2)**2).mean(axis=2).mean(axis=1).sqrt()

            self.surfaces[s_key] = rmsd.reshape(self.n_samples, self.n_samples).numpy()
            self.surfaces[z_key] = z_drift.reshape(self.n_samples, self.n_samples).numpy()
        return self.surfaces[s_key], self.surfaces[z_key], self.xvals, self.yvals

    def _ramachandran_score(self, frame, processes=-1):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'ramachandran_score_class'):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(self.mol, processes=processes) #Parallel_Ramachandran_Score(self.mol)
        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1,0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f, torch.Tensor):
            f = f.data.cpu().numpy()
        return self.ramachandran_score_class.get_score(f*self.stdval)
        #nf, na, no, nt = self.ramachandran_score_class.get_score(f*self.stdval)
        #return {'favored':nf, 'allowed':na, 'outliers':no, 'total':nt}


    def _dope_score(self, frame, refine = True, processes=-1):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'dope_score_class'):
            self.dope_score_class = Parallel_DOPE_Score(self.mol, processes=processes)

        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1,0)
        else:
            assert frame.shape[1] ==3
            f = frame
        if isinstance(f,torch.Tensor):
            f = f.data.cpu().numpy()

        return self.dope_score_class.get_score(f*self.stdval, refine = refine)


    def get_all_ramachandran_score(self, tensor, processes = -1):
        '''
        applies _ramachandran_score to an array of data
        '''
        rama = dict(favored=[], allowed=[], outliers=[], total=[])
        results = []
        for f in tensor:
            results.append(self._ramachandran_score(f, processes=processes))
        for r in results:
            favored, allowed, outliers, total = r.get()
            rama['favored'].append(favored)
            rama['allowed'].append(allowed)
            rama['outliers'].append(outliers)
            rama['total'].append(total)
        return {key:np.array(value) for key, value in rama.items()}       

    def get_all_dope_score(self, tensor, refine = True):
        '''
        applies _dope_score to an array of data
        '''
        results = []
        for f in tensor:
            results.append(self._dope_score(f, refine = refine))
        results = np.array([r.get() for r in results])
        if refine:
            return results[:,0], results[:,1]
        return results

    def reference_dope_score(self, frame):
        '''
        give a numpy array with shape [1, N, 3], already scaled to the correct size
        '''
        self.mol.coordinates = deepcopy(frame)
        self.mol.write_pdb('tmp.pdb')
        env = Environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        mdl = complete_pdb(env, 'tmp.pdb')
        atmsel = Selection(mdl.chains[0])
        score = atmsel.assess_dope()
        return score

    def scan_dope(self, **kwargs):
        u_key = f'DOPE_unrefined'
        r_key = f'DOPE_refined'
        if u_key not in self.surfaces:
            assert 'grid' in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded('grid')
            unrefined, refined = self.get_all_dope_score(decoded,**kwargs)
            self.surfaces[u_key] = as_numpy(unrefined.reshape(self.n_samples, self.n_samples))
            self.surfaces[r_key] = as_numpy(refined.reshape(self.n_samples, self.n_samples))
        return self.surfaces[u_key], self.surfaces[r_key], self.xvals, self.yvals

    def scan_ramachandran(self, processes = -1):
        keys = {i:f'Ramachandran_{i}' for i in ('favored', 'allowed', 'outliers', 'total')}
        if list(keys.values())[0] not in self.surfaces:
            assert 'grid' in self._encoded, 'make sure to call MolearnAnalysis.setup_grid first'
            decoded = self.get_decoded('grid')
            rama = self.get_all_ramachandran_score(decoded, processes=processes)
            for key, value in rama.items():
                self.surfaces[keys[key]] = value
        return self.surfaces['Ramachandran_favored'], self.xvals, self.yvals
  
    def scan_custom(self, fct, params, key):
        '''
        param f: function taking atomic coordinates as input, an optional list of parameters. Returns a single value.
        param params: parameters to be passed to function f
        param label: name of the dataset generated by this function scan
        param samples: sampling of grid sampling
        returns: grid scanning of latent space according to provided function, x, and y grid axes
        '''
        decoded = self.get_decoded('grid')
        results = []
        for i,j in enumerate(decoded):
            s = (j.view(1,3,-1).permute(0,2,1)*self.stdval).numpy()
            results.append(fct(s, *params))
        self.surfaces[key] = np.array(results).reshape(self.n_samples, self.n_samples)
        return self.surfaces[key],self.xvals, self.yvals

    def generate(self, crd):
        '''
        generate a collection of protein conformations, given (Nx2) coordinates in the latent space
        ''' 
        with torch.no_grad():
            z = torch.tensor(crd.transpose(1, 2, 0)).float()   
            s = self.network.decode(z)[:, :, :self.training_set.shape[2]].numpy().transpose(0, 2, 1)

        return s*self.stdval + self.meanval

