import numpy as np
from copy import deepcopy
from multiprocessing import get_context
from scipy.spatial.distance import cdist

from iotbx.data_manager import DataManager
from mmtbx.validation.ramalyze import ramalyze
from scitbx.array_family import flex

from ..utils import random_string
import os


class Ramachandran_Score:
    '''
    This class contains methods that use iotbx/mmtbx to calulate the quality of phi and psi values in a protein.
    '''
    
    def __init__(self, mol, threshold=1e-3):
        '''
        :param biobox.Molecule mol: One example frame to gain access to the topology. Mol will also be used to save a temporary pdb file that will be reloaded to create the initial iotbx Model.
        :param float threshold: (default: 1e-3) Threshold used to determine similarity between biobox.molecule coordinates and iotbx model coordinates. Determine that iotbx model was created successfully.
        '''
        
        tmp_file = f'rama_tmp{random_string()}.pdb'
        mol.write_pdb(tmp_file, split_struc=False)
        filename = tmp_file
        self.mol = mol
        self.dm = DataManager(datatypes=['model'])
        self.dm.process_model_file(filename)
        self.model = self.dm.get_model(filename)
        self.score = ramalyze(self.model.get_hierarchy())  # get score to see if this works
        self.shape = self.model.get_sites_cart().as_numpy_array().shape

        # tests
        x = self.mol.coordinates[0]
        m = self.model.get_sites_cart().as_numpy_array()
        assert m.shape == x.shape
        self.idxs = np.where(cdist(m, x)<threshold)[1]
        assert self.idxs.shape[0] == m.shape[0]
        assert not np.any(((m-x[self.idxs])>threshold))
        os.remove(tmp_file)

    def get_score(self, coords, as_ratio=False):
        '''
            Given coords (corresponding to self.mol) will calculate Ramachandran scores using cctbux ramalyze module
            Returns the counts of number of torsion angles that fall within favored, allowed, and outlier regions and finally the total number of torsion angles analysed.
            :param numpy.ndarray coords: shape (N, 3)
            :returns: (favored, allowed, outliers, total)
            :rtype: tuple of ints
        '''
        
        assert coords.shape == self.shape
        self.model.set_sites_cart(flex.vec3_double(coords[self.idxs].astype(np.double)))
        self.score = ramalyze(self.model.get_hierarchy())
        nf = self.score.n_favored
        na = self.score.n_allowed
        no = self.score.n_outliers
        nt = self.score.n_total
        if as_ratio:
            return nf/nt, na/nt, no/nt
        else:
            return nf, na, no, nt


def set_global_score(score, kwargs):
    '''
    make score a global variable
    This is used when initializing a multiprocessing process
    '''
    
    global worker_ramachandran_score
    worker_ramachandran_score = score(**kwargs)  # mol = mol, data_dir=data_dir, **kwargs)


def process_ramachandran(coords, kwargs):
    '''
    ramachandran worker
    Worker function for multiprocessing class
    '''
    
    return worker_ramachandran_score.get_score(coords, **kwargs)


class Parallel_Ramachandran_Score:
    '''
    A multiprocessing class to get Ramachandran scores. 
    A typical use case would looke like::

        score_class = Parallel_Ramachandran_Score(mol, **kwargs)
        results = []
        for frame in coordinates_array:
            results.append(score_class.get_score(frame))
            # Ramachandran scores will be calculated asynchronously in background
        ...
        # to retrieve the results
        results = np.array([r.get() for r in results])
        favored = results[:,0]
        allowed = results[:,1]
        outliers = results[:,2]
        total = results[:,3]

    '''
    
    def __init__(self, mol, processes=-1):
        '''
        :param biobox.Molecule mol: biobox melucel containing one example fram of the protein to be analysed. This will be passed to Ramachandran_Score instances in each thread.
        :param int processes: (default: -1) Number of processes argument to pass to multiprocessing.pool. This controls the number of therads created.
        '''
        
        # set a number of processes as user desires, capped on number of CPUs
        if processes > 0:
            processes = min(processes, os.cpu_count())
        else:
            processes = os.cpu_count()
        
        self.mol = deepcopy(mol)
        score = Ramachandran_Score
        ctx = get_context('spawn')
        
        self.pool = ctx.Pool(processes=processes, initializer=set_global_score,
                         initargs=(score, dict(mol=mol)))
        self.process_function = process_ramachandran

    def __reduce__(self):
        return (self.__class__, (self.mol,))

    def get_score(self, coords, **kwargs):
        '''
        :param coords: # shape (N, 3) numpy array
        '''
        # is copy necessary?
        return self.pool.apply_async(self.process_function, (coords.copy(), kwargs))
