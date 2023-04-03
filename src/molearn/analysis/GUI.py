import time
import pickle
from IPython import display
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

import numpy as np

import MDAnalysis as mda

import warnings
warnings.filterwarnings("ignore")

from ipywidgets import Layout
from ipywidgets import widgets
from tkinter import Tk, filedialog
import plotly.graph_objects as go
import nglview as nv


from .analyser import as_numpy, MolearnAnalysis
from .path import oversample, get_path


class MolearnGUI(object):
    
    def __init__(self, MA=[]):
        
        if not isinstance(MA, MolearnAnalysis) and MA != []:
            raise Exception(f'Expecting an MolearnAnalysis instance, {type(MA)} found')
        else:
            self.MA = MA

        self.waypoints = [] # collection of all saved waypoints
        
        self.run()

        
    #def oversample(self, crd, pts=10):
    #    '''
    #    add extra equally spaced points between a list of points ("pts" per interval)
    #    ''' 
    #    pts += 1
    #    steps = np.linspace(1./pts, 1, pts)
    #    pts = [crd[0,0]]
    #    for i in range(1, len(crd[0])):
    #        for j in steps:
    #            newpt = crd[0, i-1] + (crd[0, i]-crd[0, i-1])*j
    #            pts.append(newpt)
    #
    #    return np.array([pts])

        
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
            crd = oversample(crd, pts=int(samplebox))
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
        #view.add_representation("cartoon")
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
        crd = oversample(crd, pts=int(self.samplebox.value))

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
        
        # make all items displayed clickable
        for item in self.latent.data:
            item.on_click(self.on_click)
        
        self.block2 = widgets.VBox([self.protein], layout=Layout(flex='1 5 auto', width='auto'))

        self.scene = widgets.HBox([self.block0, self.block1, self.block2])
        self.scene.layout.align_items = 'center'

        if len(self.waypoints) > 0:
            print(self.waypoints, type(self.waypoints))
            self.mybox.value = " ".join(self.waypoints.flatten().astype(str))

        display.clear_output(wait=True)
        display.display(self.scene)
            
