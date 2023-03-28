from copy import deepcopy
import time
import pickle
from IPython import display

import numpy as np

#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim

import modeller
from modeller import *
from modeller.scripts import complete_pdb

import MDAnalysis as mda

import warnings
warnings.filterwarnings("ignore")

from ipywidgets import Layout
from ipywidgets import widgets
from tkinter import Tk, filedialog
import plotly.graph_objects as go
import nglview as nv

from .scoring import Parallel_DOPE_Score, Parallel_Ramachandran_Score
from .pdb_data import PDBData


def as_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.data.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        return np.array(tensor)


###############################################################################

class MolearnAnalysis(object):
    
    def __init__(self,):
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


    def get_error(self, dataset="", align=False):
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


    def get_dope(self, dataset="", refined=True):

        if dataset == "" or dataset == "training_set":
            dataset = self.training_set
            z = self.training_set_z
            decoded = self.training_set_decoded
        elif dataset == "test_set":
            dataset = self.test_set
            z = self.test_set_z
            decoded = self.test_set_decoded
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

    def get_ramachandran(self, dataset=""):
        if dataset == "" or dataset == "training_set":
            dataset = self.training_set
            z = self.training_set_z
            decoded = self.training_set_decoded
        elif dataset == "test_set":
            dataset = self.test_set
            z = self.test_set_z
            decoded = self.test_set_decoded
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


    def _ramachandran_score(self, frame):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'ramachandran_score_class'):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(self.mol) #Parallel_Ramachandran_Score(self.mol)
        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1,0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f, torch.Tensor):
            f = f.data.cpu().numpy()

        return self.ramachandran_score_class.get_score(f*self.stdval)


    def _dope_score(self, frame, refine = True):
        '''
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        '''
        if not hasattr(self, 'dope_score_class'):
            self.dope_score_class = Parallel_DOPE_Score(self.mol)

        assert len(frame.shape) == 2, f'We wanted 2D data but got {len(frame.shape)} dimensions'
        if frame.shape[0] == 3:
            f = frame.permute(1,0)
        else:
            assert frame.shape[1] ==3
            f = frame
        if isinstance(f,torch.Tensor):
            f = f.data.cpu().numpy()

        return self.dope_score_class.get_score(f*self.stdval, refine = refine)


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


    def scan_dope(self, samples = 50):

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
                results.append(self._dope_score(structure[0], refine = True))
        results = np.array([r.get() for r in results])
        self.surf_dope_unrefined = results[:,0].reshape(len(self.xvals), len(self.yvals))
        self.surf_dope_refined = results[:, 1].reshape(len(self.xvals), len(self.yvals))
        
        return self.surf_dope_unrefined, self.surf_dope_refined, self.xvals, self.yvals


    def scan_ramachandran(self, samples = 50):
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
                results.append(self._ramachandran_score(structure[0]))
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

###############################################################################

class MolearnGUI(object):
    
    def __init__(self, MA=[]):
        
        if not isinstance(MA, MolearnAnalysis) and MA != []:
            raise Exception(f'Expecting an MolearnAnalysis instance, {type(MA)} found')
        else:
            self.MA = MA

        self.waypoints = [] # collection of all saved waypoints
        
        self.run()

        
    def oversample(self, crd, pts=10):
        '''
        add extra equally spaced points between a list of points ("pts" per interval)
        ''' 
        pts += 1
        steps = np.linspace(1./pts, 1, pts)
        pts = [crd[0,0]]
        for i in range(1, len(crd[0])):
            for j in steps:
                newpt = crd[0, i-1] + (crd[0, i]-crd[0, i-1])*j
                pts.append(newpt)

        return np.array([pts])

        
    def on_click(self, trace, points, selector):
        '''
        control display of training set
        '''
        
        if len(points.xs) == 0:
            return

        # add new waypoint to list
        pt = np.array([[points.xs[0], points.ys[0]]])
        if len(self.waypoints) == 0:
            self.waypoints = pt    
        else:
            self.waypoints = np.concatenate((self.waypoints, pt))

        # update latent space plot
        self.latent.data[3].x = self.waypoints[:, 0]
        self.latent.data[3].y = self.waypoints[:, 1]
        self.latent.update()

        # update textbox (triggering update of 3D representation)
        try:
            pt = np.array([self.latent.data[3].x, self.latent.data[3].y]).T.flatten().round(decimals=4).astype(str)
            self.mybox.value = " ".join(pt)
        except:
            return
        

    def interact_3D(self, mybox, samplebox):
        '''
        generate and display proteins according to latent space trail
        ''' 

        # get latent space path
        try:
            crd = np.array(mybox.split()).astype(float)
            crd = crd.reshape((1, int(len(crd)/2), 2))       
            crd = self.oversample(crd, pts=int(samplebox))
        except Exception:
            self.button_pdb.disabled = True
            return 

        # generate structures along path
        t = time.time()
        gen = self.MA.generate(crd)
        print(f'{crd.shape[1]} struct. in {time.time()-t:.4f} sec.')

        # display generated structures
        self.mymol.load_new(gen)
        view = nv.show_mdanalysis(self.mymol)
        view.add_representation("spacefill")
        display.display(view)

        self.button_pdb.disabled = False


    def drop_background_event(self, change):
        '''
        control colouring style of latent space surface
        '''

        if change.new == "drift":
            try:
                data = self.MA.surf_z.T
            except:
                return
            
            self.block0.children[2].readout_format = '.1f'

        elif change.new == "RMSD":
            try:
                data = self.MA.surf_c.T
            except:
                return
            
            self.block0.children[2].readout_format = '.1f'


        elif change.new == "target RMSD":
            try:
                data = self.MA.surf_target.T
            except:
                return
            
            self.block0.children[2].readout_format = '.1f'


        elif change.new == "DOPE_unrefined":
            try:
                data = self.MA.surf_dope_unrefined.T
            except:
                return
            
            self.block0.children[2].readout_format = 'd'

        elif change.new == "DOPE_refined":
            try:
                data = self.MA.surf_dope_refined.T
            except:
                return
            
            self.block0.children[2].readout_format = 'd'
        
        elif change.new == "ramachandran_favored":
            try:
                data = self.MA.surf_ramachandran_favored.T
            except:
                return
            
            self.block0.children[2].readout_format = 'd'

        elif change.new == "ramachandran_allowed":
            try:
                data = self.MA.surf_ramachandran_allowed.T
            except:
                return
            
            self.block0.children[2].readout_format = 'd'
            
        elif change.new == "ramachandran_outliers":
            try:
                data = self.MA.surf_ramachandran_outliers.T
            except:
                return
            
            self.block0.children[2].readout_format = 'd'
            
        elif "custom" in change.new:
            mykey = change.new.split(":")[1]
            try:
                data = self.MA.custom_data[mykey].T
            except Exception:
                return      
            
            self.block0.children[2].readout_format = 'd'
                 
        self.latent.data[0].z = data
        
        # step below necessary to avoid situations whereby temporarily min>max
        try:
            self.latent.data[0].zmin = np.min(data)
            self.latent.data[0].zmax = np.max(data)
            self.block0.children[2].min = np.min(data)
            self.block0.children[2].max = np.max(data)
        except:
            self.latent.data[0].zmax = np.max(data)
            self.latent.data[0].zmin = np.min(data)
            self.block0.children[2].max = np.max(data)
            self.block0.children[2].min = np.min(data)
                
        self.block0.children[2].value = (np.min(data), np.max(data))
            
        self.latent.update()


    def check_training_event(self, change):
        '''
        control display of training set
        ''' 
        state_choice = change.new
        self.latent.data[1].visible = state_choice
        self.latent.update()


    def check_test_event(self, change):
        '''
        control display of test set
        ''' 
        state_choice = change.new
        self.latent.data[2].visible = state_choice
        self.latent.update()


    def range_slider_event(self, change):
        '''
        update surface colouring upon manipulation of range slider
        '''
        self.latent.data[0].zmin = change.new[0]
        self.latent.data[0].zmax = change.new[1]
        self.latent.update()


    def mybox_event(self, change):
        '''
        control manual update of waypoints
        '''

        try:
            crd = np.array(change.new.split()).astype(float)
            crd = crd.reshape((int(len(crd)/2), 2))
        except:
            self.button_pdb.disabled = False
            return

        self.waypoints = crd.copy()

        self.latent.data[3].x = self.waypoints[:, 0]
        self.latent.data[3].y = self.waypoints[:, 1]
        self.latent.update()


    def button_pdb_event(self, check):
        '''
        save PDB file corresponding to the interpolation shown in the 3D view
        '''

        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        fname = filedialog.asksaveasfilename(defaultextension="pdb", filetypes=[("PDB file", "pdb")])

        if fname == "":
            return

        crd = np.array(self.mybox.value.split()).astype(float)
        crd = crd.reshape((1, int(len(crd)/2), 2))       
        crd = self.oversample(crd, pts=int(self.samplebox.value))

        gen = self.MA.generate(crd)
        self.mymol.load_new(gen)
        protein = self.mymol.select_atoms("all")

        with mda.Writer(fname, protein.n_atoms) as W:
            for ts in self.mymol.trajectory:
                W.write(protein)


    def button_save_state_event(self, check):
        '''
        save class state
        '''

        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        fname = filedialog.asksaveasfilename(defaultextension="p", filetypes=[("pickle file", "p")])

        if fname == "":
            return

        pickle.dump([self.MA, self.waypoints], open( fname, "wb" ) )


    def button_load_state_event(self, check):
        '''
        load class state
        '''

        root = Tk()
        root.withdraw()                                        # Hide the main window.
        root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
        fname = filedialog.askopenfilename(defaultextension="p", filetypes=[("picke file", "p")])

        if fname == "":
            return

        self.MA, self.waypoints = pickle.load( open( fname, "rb" ) )

        self.run()

    #####################################################

    def run(self):

        # create an MDAnalysis instance of input protein (for viewing purposes)
        if hasattr(self.MA, "mol"):
            self.MA.mol.write_pdb("tmp.pdb", conformations=[0])
            self.mymol = mda.Universe('tmp.pdb')
        
        ### MENU ITEMS ###
        
        # surface representation menu
        options = []
        if hasattr(self.MA, "surf_z"):
            options.append("drift")
        if hasattr(self.MA, "surf_c"):
            options.append("RMSD")       
        if hasattr(self.MA, "surf_dope_unrefined"):
            options.append("DOPE_unrefined")
        if hasattr(self.MA, "surf_dope_refined"):
            options.append("DOPE_refined")
        if hasattr(self.MA, "surf_target"): 
            options.append("target RMSD")
        if hasattr(self.MA, "surf_ramachandran_favored"):
            options.append("ramachandran_favored")
        if hasattr(self.MA, "surf_ramachandran_allowed"):
            options.append("ramachandran_allowed")
        if hasattr(self.MA, "surf_ramachandran_outliers"):
            options.append("ramachandran_outliers")
        if hasattr(self.MA, "custom_data"):
            for k in list(self.MA.custom_data):
                options.append(f'custom:{k}')

        if len(options) == 0:
            options.append("none")
        
        self.drop_background = widgets.Dropdown(
            options=options,
            value=options[0],
            description='Surf.:',
            layout=Layout(flex='1 1 0%', width='auto'))

        if "none" in options:
            self.drop_background.disabled = True
        
        self.drop_background.observe(self.drop_background_event, names='value')

        # training set visualisation menu
        self.check_training = widgets.Checkbox(
            value=False,
            description='show training',
            disabled=False,
            indent=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.check_training.observe(self.check_training_event, names='value')

        # test set visualisation menu
        self.check_test = widgets.Checkbox(
            value=False,
            description='show test',
            disabled=False,
            indent=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.check_test.observe(self.check_test_event, names='value')

        # text box holding current coordinates
        self.mybox = widgets.Textarea(placeholder='coordinates',
                                 description='crds:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.mybox.observe(self.mybox_event, names='value')

        self.samplebox = widgets.Text(value='10',
                                 description='sampling:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        # button to save PDB file
        self.button_pdb = widgets.Button(
            description='Save PDB',
            disabled=True, layout=Layout(flex='1 1 0%', width='auto'))

        self.button_pdb.on_click(self.button_pdb_event)


        # button to save state file
        self.button_save_state = widgets.Button(
            description= 'Save state',
            disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.button_save_state.on_click(self.button_save_state_event)


        # button to load state file
        self.button_load_state = widgets.Button(
            description= 'Load state',
            disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.button_load_state.on_click(self.button_load_state_event)


        # latent space range slider
        self.range_slider = widgets.FloatRangeSlider(
            description='cmap range:',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f', layout=Layout(flex='1 1 0%', width='auto'))

        self.range_slider.observe(self.range_slider_event, names='value')


        if self.MA == []:
            self.button_save_state.disabled = True
            self.button_pdb.disabled = True
            
        if self.waypoints == []:
            self.button_pdb.disabled = True

        
        ### LATENT SPACE REPRESENTATION ###

        # coloured background
        if "drift" in options:
            sc = self.MA.surf_z
        elif "target RMSD" in options:
            sc = self.MA.surf_target
        elif "DOPE_unrefined" in options:
            sc = self.MA.surf_dope_unrefined
        elif "DOPE_refined" in options:
            sc = self.MA.surf_dope_refined
        elif "ramachandran_favored" in options:
            sc = self.MA.surf_ramachandran_favored
        elif "ramachandran_allowed" in options:
            sc = self.MA.surf_ramachandran_allowed
        elif "ramachandran_outliers" in options:
            sc = self.MA.surf_ramachandran_outliers
        elif len(options)>0:
            if "custom" in options[0]:
                label = options[0].split(":")[1]
                sc = self.MA.custom_data[label]
            else:
                sc = []
        else:
            sc = []
            
        if len(sc)>0:
            plot1 = go.Heatmap(x=self.MA.xvals, y=self.MA.yvals, z=sc.T, zmin=np.min(sc), zmax=np.max(sc),
                               colorscale='viridis', name="latent_space")   
        else:

            if self.MA:
                xvals, yvals = self.MA._get_sampling_ranges(50)
            else:               
                xvals = np.linspace(0, 1, 10)
                yvals = np.linspace(0, 1, 10)
                    
            surf_empty = np.zeros((len(xvals), len(yvals)))
            plot1 = go.Heatmap(x=xvals, y=yvals, z=surf_empty, opacity=0.0, showscale=False, name="latent_space")   
                      
        # training set
        if hasattr(self.MA, "training_set_z"):
            color = "white" if len(sc)>0 else "black"
            plot2 = go.Scatter(x=as_numpy(self.MA.training_set_z)[:, 0].flatten(),
                               y=as_numpy(self.MA.training_set_z)[:, 1].flatten(),
                   showlegend=False, opacity=0.9, mode="markers",
                   marker=dict(color=color, size=5), name="training", visible=False)
        else:
            print("no data available")
            plot2 = go.Scatter(x=[], y=[])
            self.check_training.disabled = True
            
        # test set
        if hasattr(self.MA, "test_set_z"):
            plot3 = go.Scatter(x=as_numpy(self.MA.test_set_z)[:, 0].flatten(),
                               y=as_numpy(self.MA.test_set_z)[:, 1].flatten(),
                   showlegend=False, opacity=0.9, mode="markers",
                   marker=dict(color='silver', size=5), name="test", visible=False)
        else:
            plot3 = go.Scatter(x=[], y=[])
            self.check_test.disabled = True
      
        # path
        plot4 = go.Scatter(x=np.array([]), y=np.array([]),
                   showlegend=False, opacity=0.9,
                   marker=dict(color='red', size=7))

        self.latent = go.FigureWidget([plot1, plot2, plot3, plot4])
        self.latent.update_layout(xaxis_title="latent vector 1", yaxis_title="latent vector 2",
                         autosize=True, width=400, height=350, margin=dict(l=75, r=0, t=25, b=0))
        self.latent.update_xaxes(showspikes=False)
        self.latent.update_yaxes(showspikes=False)       

        if len(sc)>0:
            self.range_slider.value = (np.min(sc), np.max(sc))
            self.range_slider.min = np.min(sc)
            self.range_slider.max = np.max(sc)
            self.range_slider.step = (np.max(sc)-np.min(sc))/100.0
            self.range_slider.disabled = False

        # 3D protein representation (triggered by update of textbox)
        self.protein = widgets.interactive_output(self.interact_3D, {'mybox': self.mybox, 'samplebox': self.samplebox})

        
        ### WIDGETS ARRANGEMENT ###
        
        self.block0 = widgets.VBox([self.check_training, self.check_test, self.range_slider,
                                    self.drop_background, self.samplebox, self.mybox,
                                    self.button_pdb, self.button_save_state, self.button_load_state],
                              layout=Layout(flex='1 1 2', width='auto', border="solid"))

        self.block1 = widgets.VBox([self.latent], layout=Layout(flex='1 1 auto', width='auto'))
        self.latent.data[0].on_click(self.on_click)

        self.block2 = widgets.VBox([self.protein], layout=Layout(flex='1 5 auto', width='auto'))

        self.scene = widgets.HBox([self.block0, self.block1, self.block2])
        self.scene.layout.align_items = 'center'

        if len(self.waypoints) > 0:
            print(self.waypoints, type(self.waypoints))
            self.mybox.value = " ".join(self.waypoints.flatten().astype(str))

        display.clear_output(wait=True)
        display.display(self.scene)
            
