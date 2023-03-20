import sys, os, glob
import numpy as np
from copy import deepcopy
import modeller
from modeller import *
from modeller.scripts import complete_pdb
from modeller.optimizers import ConjugateGradients

from multiprocessing import Pool, Event, get_context

def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.data.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)

class ShutUp(object):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout =  self._stdout
    
class DOPE_Score:

    def __init__(self, mol):

        #set residues names with protonated histidines back to generic HIS name (needed by DOPE score function)
        testH = mol.data["resname"].values
        testH[testH == "HIE"] = "HIS"
        testH[testH == "HID"] = "HIS"
        _mol = deepcopy(mol)
        _mol.data["resname"] = testH

        alternate_residue_names = dict(CSS=('CYX',))
        atoms = ' '.join(list(_mol.data['name'].unique()))
        _mol.write_pdb('tmp.pdb', conformations=[0])
        log.level(0,0,0,0,0)
        env = environ()
        env.libs.topology.read(file='$(LIB)/top_heav.lib')
        env.libs.parameters.read(file='$(LIB)/par.lib')
        self.fast_mdl = complete_pdb(env, 'tmp.pdb')
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

    def get_dope(self, frame, refine=False):
        # expect coords to be shape [N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        frame = frame.astype(float)
        self.fast_fs.unbuild()
        for i, j in enumerate(self.fast_ss):
            if i+1<frame.shape[0]:
                j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
        self.fast_mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)
        with ShutUp():
            dope_unrefined = self.fast_fs.assess_dope()
            if refine:
                self.cg.optimize(self.fast_fs, max_iterations=50)
                dope_refined = self.fast_fs.assess_dope()
        if refine:
            return dope_unrefined, dope_refined
        return dope_unrefined
    def get_all_dope(self, coords, refine=False):
        # expect coords to be shape [B, N, 3] use .cpu().numpy().copy() before passing here and make sure it is scaled correctly
        dope_scores_unrefined = []
        dope_scores_refined = []
        for frame in coords:
            frame = frame.astype(float)
            self.fast_fs.unbuild()
            for i, j in enumerate(self.fast_ss):
                if i+1<frame.shape[0]:
                    j.x, j.y, j.z = frame[self.fast_atom_order[i], :]
            self.fast_mdl.build(build_method='INTERNAL_COORDINATES', initialize_xyz=False)
            dope_scores_unrefined.append(self.fast_fs.assess_dope())
            if refine:
                self.cg.optimize(self.fast_fs, max_iterations=50)
                dope_scores_refined.append(self.fast_fs.assess_dope())
        if refine:
            return np.array(dope_scores_unrefined), np.array(dope_scores_refined)
        return np.array(dope_scores_unrefined)

def set_global_score(score, kwargs):
    '''
    make score a global variable
    '''
    global worker_dope_score
    worker_dope_score = score(**kwargs)#mol = mol, data_dir=data_dir, **kwargs)

def process_dope(coords, kwargs):
    '''
    dope worker
    '''
    return worker_dope_score.get_dope(coords,**kwargs)

class Parallel_DOPE_Score():
    def __init__(self, mol, **kwargs):
        self.mol = deepcopy(mol)
        score = DOPE_Score
        ctx = get_context('spawn')
        self.pool = ctx.Pool(initializer=set_global_score,
                         initargs=(score, dict(mol=mol)),
                         **kwargs,
                         )
        self.process_function = process_dope

    def __reduce__(self):
        return (self.__class__, (self.mol,))

    def get_score(self, coords,**kwargs):
        '''
        :param coords: # shape (1, N, 3) numpy array
        '''
        #is copy necessary?
        return self.pool.apply_async(self.process_function, (coords.copy(), kwargs))

