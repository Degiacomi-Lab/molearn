import numpy as np
from copy import deepcopy
from multiprocessing import Pool, Event, get_context
from scipy.spatial.distance import cdist

from iotbx.data_manager import DataManager
from mmtbx.validation.ramalyze import ramalyze
from scitbx.array_family import flex

from ..utils import cpu_count, random_string


class Ramachandran_Score():
    def __init__(self, mol, threshold=1e-3):
        tmp_file = f'rama_tmp{random_string()}.pdb'
        mol.write_pdb(tmp_file)#'rama_tmp.pdb')
        filename = tmp_file#'rama_tmp.pdb'
        self.mol = mol
        self.dm = DataManager(datatypes = ['model'])
        self.dm.process_model_file(filename)
        self.model = self.dm.get_model(filename)
        self.score = ramalyze(self.model.get_hierarchy()) # get score to see if this works
        self.shape = self.model.get_sites_cart().as_numpy_array().shape

        #tests
        x = self.mol.coordinates[0]
        m = self.model.get_sites_cart().as_numpy_array()
        assert m.shape == x.shape
        self.idxs = np.where(cdist(m, x)<threshold)[1]
        assert self.idxs.shape[0] == m.shape[0]
        assert not np.any(((m-x[self.idxs])>threshold))
        os.remove(tmp_file)

    def get_score(self, coords, as_ratio = False):
        '''
            Given coords (corresponding to self.mol) will calculate Ramachandran scores using cctbux ramalyze module
            :param coords: numpy array (shape (N, 3))
            :returns: (favored, allowed, outliers, total)

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
    '''
    global worker_ramachandran_score
    worker_ramachandran_score = score(**kwargs)#mol = mol, data_dir=data_dir, **kwargs)

def process_ramachandran(coords, kwargs):
    '''
    ramachandran worker
    '''
    return worker_ramachandran_score.get_score(coords,**kwargs)

class Parallel_Ramachandran_Score():
    
    def __init__(self, mol, processes=-1):
        
        # set a number of processes as user desires, capped on number of CPUs
        if processes > 0:
            processes = min(processes, cpu_count())
        else:
            processes = cpu_count()
        
        self.mol = deepcopy(mol)
        score = Ramachandran_Score
        ctx = get_context('spawn')
        
        self.pool = ctx.Pool(processes=processes, initializer=set_global_score,
                         initargs=(score, dict(mol=mol)),
                         )
        self.process_function = process_ramachandran

    def __reduce__(self):
        return (self.__class__, (self.mol,))


    def get_score(self, coords,**kwargs):
        '''
        :param coords: # shape (1, N, 3) numpy array
        '''
        #is copy necessary?
        return self.pool.apply_async(self.process_function, (coords.copy(), kwargs))



