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

import modeller
from modeller import *
from modeller.scripts import complete_pdb

from ..scoring import Parallel_DOPE_Score, Parallel_Ramachandran_Score
from ..pdb_data import PDBData

from ..utils import as_numpy

import warnings
warnings.filterwarnings("ignore")


class MolearnAnalysis(object):
    
    def __init__(self):
        pass

    def set_train_data(self, data, atomselect="*"):
        if isinstance(data, str) and data.endswith('.pdb'):
            d = PDBData()
            d.import_pdb(data)
            d.atomselect(atomselect)
            d.prepare_dataset()
            self._training_set = d.dataset.float()
            self.meanval = d.mean
            self.stdval = d.std
            self.atoms = d.atoms
            self.mol = d.frame()
        elif isinstance(data, PDBData):
            self._training_set = data.dataset.float()
            self.meanval = data.mean
            self.stdval = data.std
            self.atoms = data.atoms
            self.mol = data.frame()
        else:
            raise NotImplementedError('No other data typethan PDBData has been implemented for this method')

    def set_network(self, network):
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device


    def set_test_data(self, data, use_training_parameters=False):
        if isinstance(data, str) and data.endswith('.pdb'):
            d = PDBData()
            d.import_pdb(data)
            d.atomselect(self.atoms)
            if use_training_parameters:
                d.std = self.stdval
                d.mean = self.meanval
            d.prepare_dataset()
            self._test_set = d.dataset.float()
        elif isinstance(data, PDBData):
            self._test_set = data.dataset.float()
        if self._test_set.shape[2] != self.training_set.shape[2]:
            raise Exception(f'number of d.o.f. differs: training set has {self.training_set.shape[2]}, test set has {self._test_set.shape[2]}')

    def num_trainable_params(self):

        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    @property
    def training_set(self):
        return self._training_set.clone()


    @property
    def training_set_z(self):
        if not hasattr(self, '_training_set_z'):
            with torch.no_grad():
                self._training_set_z = self.network.encode(self.training_set.to(self.device))
        return self._training_set_z.clone()

    @property
    def test_set(self):
        return self._test_set.data
    @property
    def training_set_decoded(self):
        if not hasattr(self, '_training_set_decoded'):
            with torch.no_grad():
                self._training_set_decoded = self.network.decode(self.training_set_z.to(self.device))[:,:,:self.training_set.shape[2]]
        return self._training_set_decoded.clone()

    @property
    def test_set_z(self):
        if not hasattr(self, '_test_set_z'):
            with torch.no_grad():
                self._test_set_z = self.network.encode(self.test_set.to(self.device))
        return self._test_set_z.clone()

    @property
    def test_set_decoded(self):
        if not hasattr(self, '_test_set_decoded'):
            with torch.no_grad():
                self._test_set_decoded = self.network.decode(self.test_set_z.to(self.device))[:,:,:self.test_set.shape[2]]
        return self._test_set_decoded.clone()


    def get_error(self, dataset="", z=None, decoded=None, align=False):
        '''
        Calculate the reconstruction error of a dataset encoded and decoded by a trained neural network
        '''

        if dataset == "" or dataset =="training_set":
            dataset = self.training_set
            z = self.training_set_z
            decoded = self.training_set_decoded
        elif dataset == "test_set":
            dataset = self.test_set
            z = self.test_set_z
            decoded = self.test_set_decoded
        elif z is not None:
            if decoded is None:
                with torch.no_grad():
                    decoded = self.network.decode(z)[:,:,:dataset.shape[2]]
        else:
            with torch.no_grad():
                z = self.network.encode(dataset.float())
                decoded = self.network.decode(z)[:,:,:dataset.shape[2]]

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


    def get_dope(self, dataset="", z=None, decoded = None, refined=True):

        if dataset == "" or dataset == "training_set":
            dataset = self.training_set
            z = self.training_set_z
            decoded = self.training_set_decoded
        elif dataset == "test_set":
            dataset = self.test_set
            z = self.test_set_z
            decoded = self.test_set_decoded
        elif z is not None:
            if decoded is None:
                with torch.no_grad():
                    decoded = self.network.decode(z)[:,:,:dataset.shape[2]]
        else:
            with torch.no_grad():
                z = self.network.encode(dataset.float())
                decoded = self.network.decode(z)[:,:,:dataset.shape[2]]

        
        dope_dataset = self.get_all_dope_score(dataset)
        dope_decoded = self.get_all_dope_score(decoded)

        if refined:
            return dope_dataset, dope_decoded
        else:
            return (dope_dataset,), (dope_decoded,)

    def get_ramachandran(self, dataset="", z = None, decoded = None):
        if dataset == "" or dataset == "training_set":
            dataset = self.training_set
            z = self.training_set_z
            decoded = self.training_set_decoded
        elif dataset == "test_set":
            dataset = self.test_set
            z = self.test_set_z
            decoded = self.test_set_decoded
        elif z is not None:
            if decoded is None:
                with torch.no_grad():
                    decoded = self.network.decode(z)[:,:,:dataset.shape[2]]
        else:
            with torch.no_grad():
                z = self.network.encode(dataset.float())
                decoded = self.network.decode(z)[:, :, :dataset.shape[2]]

        ramachandran_dataset = self.get_all_ramachandran_score(dataset)
        ramachandran_decoded = self.get_all_ramachandran_score(decoded)
        return ramachandran_dataset, ramachandran_decoded


    def _get_sampling_ranges(self, samples):
        
        bx = (np.max(as_numpy(self.training_set_z)[:, 0]) - np.min(as_numpy(self.training_set_z)[:, 0]))*0.1 # 10% margins on x-axis
        by = (np.max(as_numpy(self.training_set_z)[:, 1]) - np.min(as_numpy(self.training_set_z)[:, 1]))*0.1 # 10% margins on y-axis
        xvals = np.linspace(np.min(as_numpy(self.training_set_z)[:, 0])-bx, np.max(as_numpy(self.training_set_z)[:, 0])+bx, samples)
        yvals = np.linspace(np.min(as_numpy(self.training_set_z)[:, 1])-by, np.max(as_numpy(self.training_set_z)[:, 1])+by, samples)
    
        return xvals, yvals
        
    
    def scan_error_from_target(self, target, samples=50):
        '''
        experimental function, creating a coloured landscape of RMSD vs single target structure
        target should be a Tensor of a single protein stucture loaded via load_test
        '''

        target = target.numpy().flatten()*self.stdval + self.meanval
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        surf_compare = np.zeros((len(self.xvals), len(self.yvals)))

        with torch.no_grad():

            for x, i in enumerate(self.xvals):
                for y, j in enumerate(self.yvals):

                    # take latent space coordinate (1) and decode it (2)
                    z = torch.tensor([[[i,j]]]).float()
                    s = self.network.decode(z)[:,:,:self.training_set.shape[2]]*self.stdval + self.meanval

                    surf_compare[x,y] = np.sum((s.numpy().flatten()-target)**2)

        self.surf_target = np.sqrt(surf_compare/len(target))

        return self.surf_target, self.xvals, self.yvals
        
    
    def scan_error(self, samples = 50):
        '''
        grid sample the latent space on a samples x samples grid (50 x 50 by default).
        Boundaries are defined by training set projections extrema, plus/minus 10%
        '''
        
        if hasattr(self, "surf_z"):
            if samples == len(self.surf_z):
                return self.surf_z, self.surf_c, self.xvals, self.yvals
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        surf_z = np.zeros((len(self.xvals), len(self.yvals))) # L2 norms in latent space ("drift")
        surf_c = np.zeros((len(self.xvals), len(self.yvals))) # L2 norms in Cartesian space

        with torch.no_grad():

            for x, i in enumerate(self.xvals):
                for y, j in enumerate(self.yvals):

                    # take latent space coordinate (1) and decode it (2)
                    z1 = torch.tensor([[[i,j]]]).float()
                    s1 = self.network.decode(z1)[:,:,:self.training_set.shape[2]]

                    # take the decoded structure, re-encode it (3) and then decode it (4)
                    z2 = self.network.encode(s1)
                    s2 = self.network.decode(z2)[:,:,:self.training_set.shape[2]]

                    surf_z[x,y] = np.sum((z2.numpy().flatten()-z1.numpy().flatten())**2) # Latent space L2, i.e. (1) vs (3)
                    surf_c[x,y] = np.sum((s2.numpy().flatten()-s1.numpy().flatten())**2) # Cartesian L2, i.e. (2) vs (4)
        
        self.surf_c = np.sqrt(surf_c)
        self.surf_z = np.sqrt(surf_z)
        
        return self.surf_z, self.surf_c, self.xvals, self.yvals


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

        return self.dope_score_class.get_score(f*self.stdval, refine=refine)


    def get_all_ramachandran_score(self, tensor):
        '''
        applies _ramachandran_score to an array of data
        '''
        results = []
        for f in tensor:
            results.append(self._ramachandran_score(f))
        results = np.array([r.get() for r in results])
        return results

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


    def scan_dope(self, samples = 50, refine = True, processes = -1):

        if hasattr(self, "surf_dope_refined") and hasattr(self, "surf_dope_unrefined"):
            if samples == len(self.surf_dope_refined) and samples == len(self.surf_dope_unrefined):
                return self.surf_dope_unrefined, self.surf_dope_refined, self.xvals, self.yvals
        
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        
        X, Y = torch.meshgrid(torch.tensor(self.xvals), torch.tensor(self.yvals))
        z_in = torch.stack((X,Y), dim=2).view(samples*samples, 1, 2, 1).float()

        #surf_dope = np.zeros((len(self.xvals)*len(self.yvals),))
        results = []
        with torch.no_grad():
            for i, j in enumerate(z_in):
                structure = self.network.decode(j)[:,:,:self.training_set.shape[2]]
                results.append(self._dope_score(structure[0], refine=refine, processes=processes))
        
        results = np.array([r.get() for r in results])
        
        if refine:
            self.surf_dope_unrefined = results[:,0].reshape(len(self.xvals), len(self.yvals))
            self.surf_dope_refined = results[:, 1].reshape(len(self.xvals), len(self.yvals))
            return self.surf_dope_unrefined, self.surf_dope_refined, self.xvals, self.yvals
        else:
            self.surf_dope_unrefined = results.reshape(len(self.xvals), len(self.yvals))
            return self.surf_dope_unrefined, self.xvals, self.yvals            


    def scan_all(self, samples = 50, processes = -1):
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        X, Y = torch.meshgrid(torch.tensor(self.xvals), torch.tensor(self.yvals))
        z_in = torch.stack((X,Y), dim=2).view(samples*samples, 1, 2, 1).float()
        z_size = len(z_in)
        dope = []
        rama = []
        error_z = []
        error_c = []
        with torch.no_grad():
            for i,j in enumerate(z_in):
                if i%100==0:
                    print(f'on same {i} out of {z_size}, {100*i/z_size}%')

                structure = self.network.decode(j.to(self.device))[:,:,:self.training_set.shape[2]]
                dope.append(self._dope_score(structure[0], refine=True, processes=processes))
                rama.append(self._ramachandran_score(structure[0], processes=processes))
                z2 = self.network.encode(structure)
                structure2 = self.network.decode(z2)[:,:,:self.training_set.shape[2]]
                error_z.append(np.sum((z2.cpu().numpy().flatten()-j.numpy().flatten())**2)) # Latent space L2, i.e. (1) vs (3)
                error_c.append(np.sum((structure2.cpu().numpy().flatten()-structure.cpu().numpy().flatten())**2)) # Cartesian L2, i.e. (2) vs (4)
        
        print('finish dope')
        dope = np.array([r.get() for r in dope])
        print('finish rama')
        rama = np.array([r.get() for r in rama])
        error_z = np.sqrt(np.array(error_z))
        error_c = np.sqrt(np.array(error_c))

        self.surf_c = error_c.reshape(samples, samples)
        self.surf_z = error_z.reshape(samples, samples)
        self.surf_dope_unrefined = dope[:,0].reshape(samples, samples)
        self.surf_dope_refined = dope[:,1].reshape(samples, samples)
        self.surf_ramachandran_favored = rama[:,0].reshape(samples, samples)
        self.surf_ramachandran_allowed = rama[:,1].reshape(samples, samples)
        self.surf_ramachandran_outliers = rama[:,2].reshape(samples, samples)
        self.surf_ramachandran_total = rama[:,3].reshape(samples, samples)

        return dict(
            landscape_err_latent=self.surf_z,
            landscape_err_3d=self.surf_c,
            landscape_dope_unrefined = self.surf_dope_unrefined,
            landscape_dope_refined = self.surf_dope_refined,
            landscape_ramachandran_favored = self.surf_ramachandran_favored,
            landscape_ramachandran_allowed = self.surf_ramachandran_allowed,
            landscape_ramachandran_outlier = self.surf_ramachandran_outliers,
            landscape_ramachandran_total = self.surf_ramachandran_total,
            xvals = self.xvals,
            yvals = self.yvals,
            )


    def scan_ramachandran(self, samples = 50, processes = -1):
        if hasattr(self, "surf_ramachandran"):
            if samples == len(self.surf_ramachandran):
                return self.surf_ramachandran_favored, self.surf_ramachandran_allowed, self.surf_ramachandran_outliers
        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        X, Y = torch.meshgrid(torch.tensor(self.xvals), torch.tensor(self.yvals))
        z_in = torch.stack((X,Y), dim=2).view(samples*samples,1,2,1).float()

        results = []
        with torch.no_grad():
            for i,j in enumerate(z_in):
                structure = self.network.decode(j)[:,:,:self.training_set.shape[2]]
                results.append(self._ramachandran_score(structure[0], processes=processes))
        results = np.array([r.get() for r in results])
        self.surf_ramachandran_favored = results[:,0].reshape(len(self.xvals), len(self.yvals))
        self.surf_ramachandran_allowed = results[:,1].reshape(len(self.xvals), len(self.yvals))
        self.surf_ramachandran_outliers = results[:,2].reshape(len(self.xvals), len(self.yvals))
        self.surf_ramachandran_total = results[:,3].reshape(len(self.xvals), len(self.yvals))

        return self.surf_ramachandran_favored, self.xvals, self.yvals

  
    def scan_custom(self, fct, params, label, samples = 50):
        '''
        param f: function taking atomic coordinates as input, an optional list of parameters. Returns a single value.
        param params: parameters to be passed to function f
        param label: name of the dataset generated by this function scan
        param samples: sampling of grid sampling
        returns: grid scanning of latent space according to provided function, x, and y grid axes
        '''
        
        if hasattr(self, "custom_data"):
            if label in list(self.custom_data) and samples == len(self.custom_data[label]):
                return self.custom_data[label], self.xvals, self.yvals
        else:
            self.custom_data = {}

        self.xvals, self.yvals = self._get_sampling_ranges(samples)
        X, Y = torch.meshgrid(torch.tensor(self.xvals), torch.tensor(self.yvals))
        z_in = torch.stack((X,Y), dim=2).view(samples*samples,1,2,1).float()

        results = []
        with torch.no_grad():
            for i, j in enumerate(z_in):
                
                structure = self.network.decode(j)[:,:,:self.training_set.shape[2]].numpy().transpose(0, 2, 1)
                results.append(fct(structure*self.stdval + self.meanval, *params))
                
        results = np.array(results)
        self.custom_data[label] = results.reshape(len(self.xvals), len(self.yvals))
        
        return self.custom_data[label], self.xvals, self.yvals

  
    def generate(self, crd):
        '''
        generate a collection of protein conformations, given (Nx2) coordinates in the latent space
        ''' 
        with torch.no_grad():
            z = torch.tensor(crd.transpose(1, 2, 0)).float()   
            s = self.network.decode(z)[:, :, :self.training_set.shape[2]].numpy().transpose(0, 2, 1)

        return s*self.stdval + self.meanval
