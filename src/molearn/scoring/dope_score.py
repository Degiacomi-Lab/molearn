import numpy as np
from copy import deepcopy

from ..utils import ShutUp, cpu_count, random_string

import modeller
from modeller import *
from modeller.scripts import complete_pdb
from modeller.optimizers import ConjugateGradients

from multiprocessing import Pool, Event, get_context
import os
    
class DOPE_Score:
    '''
    This class contains methods to calculate dope without saving to save and load PDB files for every structure. Atoms in a biobox coordinate tensor are mapped to the coordinates in the modeller model directly.
    '''

    def __init__(self, mol):
        '''
        :param biobox.Molecule mol: One example frame to gain access to the topology. Mol will also be used to save a temporary pdb file that will be reloaded in modeller to create the initial modeller Model.
        '''

        #set residues names with protonated histidines back to generic HIS name (needed by DOPE score function)
        testH = mol.data["resname"].values
        testH[testH == "HIE"] = "HIS"
        testH[testH == "HID"] = "HIS"
        _mol = deepcopy(mol)
        _mol.data["resname"] = testH

        alternate_residue_names = dict(CSS=('CYX',))
        atoms = ' '.join(list(_mol.data['name'].unique()))
        #tmp_file = f'tmp{np.random.randint(1e10)}.pdb'
        tmp_file = f'tmp{random_string()}.pdb'
        _mol.write_pdb(tmp_file, conformations=[0], split_struc = False)
        log.level(0,0,0,0,0)
        env = environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        self.fast_mdl = complete_pdb(env, tmp_file)
        self.fast_fs = selection(self.fast_mdl.chains[0])
        self.fast_ss = self.fast_fs.only_atom_types(atoms)
        atom_residue = _mol.get_data(columns=['name', 'resname', 'resid'])
        atom_order = []
        first_index = next(iter(self.fast_ss)).residue.index
        offset = atom_residue[0,2]-first_index
        for i, j in enumerate(self.fast_ss):
            if i < len(atom_residue):
                for j_residue_name in alternate_residue_names.get(j.residue.name, (j.residue.name,)):
                    if [j.name, j_residue_name, j.residue.index+offset] == list(atom_residue[i]):
                        atom_order.append(i)
                    else:
                        where_arg = (atom_residue==(np.array([j.name, j_residue_name, j.residue.index+offset], dtype=object))).all(axis=1)
                        where = np.where(where_arg)[0]
                        atom_order.append(int(where))
        self.fast_atom_order = atom_order
        # check fast dope atoms
        for i,j in enumerate(self.fast_ss):
            if i<len(atom_residue):
                assert _mol.data['name'][atom_order[i]]==j.name
        self.cg = ConjugateGradients()
        os.remove(tmp_file)

    def get_dope(self, frame, refine=False):
        '''
        Get the dope score. Injects coordinates into modeller and uses ``mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False) to reconstruct missing atoms.
        If a error is thrown by modeller or at any stage, we just return a fixed large value of 1e10.
        :param numpy.ndarray frame: shape [N, 3]
        :param bool refine: (default: False) If True, relax the structures using a maximum of 50 steps of ConjugateGradient descent
        :returns: Dope score as calculated by modeller. If error is thrown we just simply return 1e10.
        :rtype: float
        '''
        # expect coords to be shape [N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        try:
            frame = frame.astype(float)
            self.fast_fs.unbuild()
            for i, j in enumerate(self.fast_ss):
                if i+1<frame.shape[0]:
                    j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
            self.fast_mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)
            if refine == 'both':
                with ShutUp():
                    dope_unrefined = self.fast_fs.assess_dope()
                    self.cg.optimize(self.fast_fs, max_iterations=50)
                    dope_refined = self.fast_fs.assess_dope()
                    return dope_unrefined, dope_refined
            with ShutUp():
                if refine:
                    self.cg.optimize(self.fast_fs, max_iterations=50)
                    
                dope_score = self.fast_fs.assess_dope()

            return dope_score
        except:
            return 1e10
        
    def get_all_dope(self, coords, refine=False):
        '''
        Expect a array of frames. return array of DOPE score value.
        :param numpy.ndarray coords: shape [B, N, 3]
        :param bool refine: (default: False) If True, relax the structures using a maximum of 50 steps of Conjugate Gradient descent
        :returns: float array shape [B]
        :rtype: np.ndarray
        '''
        # expect coords to be shape [B, N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        dope_scores = []
        for frame in coords:
            frame = frame.astype(float)
            self.fast_fs.unbuild()
            for i, j in enumerate(self.fast_ss):
                if i+1<frame.shape[0]:
                    j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
            self.fast_mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)

            if refine:
                self.cg.optimize(self.fast_fs, max_iterations=50)
                
            dope_scores.append(self.fast_fs.assess_dope())
            
        return np.array(dope_scores)

def set_global_score(score, kwargs):
    '''
    make score a global variable
    This is used when initializing a multiprocessing process
    '''
    global worker_dope_score
    worker_dope_score = score(**kwargs)#mol = mol, data_dir=data_dir, **kwargs)

def process_dope(coords, kwargs):
    '''
    dope worker
    Worker function for multiprocessing class
    '''
    return worker_dope_score.get_dope(coords,**kwargs)

class Parallel_DOPE_Score():
    '''
    a multiprocessing class to get modeller DOPE scores.
    A typical use case would looke like::

      score_class = Parallel_DOPE_Score(mol, **kwargs)
      results = []
      for frame in coordinates_array:
          results.append(score_class.get_score(frame))
      .... # DOPE will be calculated asynchronously in background
      #to retrieve the results
      results = np.array([r.get() for r in results])

    '''
    def __init__(self, mol, processes=-1, context = 'spawn', **kwargs):
        '''
        :param biobox.Molecule mol: biobox molecule containing one example frame of the protein to be analysed. This will be passed to DOPE_Score class instances in each thread.
        :param int processes: (default: -1) Number of processes argument to pass to multiprocessing.pool. This controls the number of threads created.
        :param \*\*kwargs: additional kwargs will be passed multiprocesing.pool during initialisation.
        '''
        
        # set a number of processes as user desires, capped on number of CPUs
        if processes > 0:
            processes = min(processes, cpu_count())
        else:
            processes = cpu_count()
        self.processes = processes
        self.mol = deepcopy(mol)
        score = DOPE_Score
        ctx = get_context(context)
        self.pool = ctx.Pool(processes=processes, initializer=set_global_score,
                         initargs=(score, dict(mol=mol)),
                         **kwargs,
                         )
        self.process_function = process_dope

    def __reduce__(self):
        return (self.__class__, (self.mol, self.processes))

    def get_score(self, coords, **kwargs):
        '''
        :param np.array coords: # shape (N, 3) numpy array
        '''
        #is copy necessary?
        return self.pool.apply_async(self.process_function, (coords.copy(), kwargs))
