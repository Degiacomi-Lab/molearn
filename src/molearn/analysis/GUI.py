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

import time
import pickle
from IPython import display

import numpy as np
import MDAnalysis as mda

import warnings
warnings.filterwarnings("ignore")

from ipywidgets import Layout
from ipywidgets import widgets
from tkinter import Tk, filedialog
import plotly.graph_objects as go
import nglview as nv

from .analyser import MolearnAnalysis
from .path import oversample, get_path_aggregate
from ..utils import as_numpy


class MolearnGUI(object):
    
    def __init__(self, MA=None):
        '''
        :param MA: :func:`MolearnAnalysis <molearn.analysis.MolearnAnalysis>` instance
        '''
        
        if not isinstance(MA, MolearnAnalysis) and MA is not None:
            raise Exception(f'Expecting an MolearnAnalysis instance, {type(MA)} found')
        else:
            self.MA = MA

        self.waypoints = [] # collection of all saved waypoints
        self.samples = [] # collection of all calculated sampling points
        
        self.run()

     
    def update_trails(self):
        
        try:
            crd = self.get_samples(self.mybox.value, int(self.samplebox.value), self.drop_path.value)
            self.samples = crd.copy()
        except:
            self.button_pdb.disabled = False
            return

        # update latent space plot
        if self.samples == []:
            self.latent.data[2].x = self.waypoints[:, 0]
            self.latent.data[2].y = self.waypoints[:, 1]
        else:
            self.latent.data[2].x = self.samples[:, 0]
            self.latent.data[2].y = self.samples[:, 1]
        
        self.latent.update()

    
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

        # update textbox (triggering update of 3D representation)
        try:
            pt = self.waypoints.flatten().round(decimals=4).astype(str)
            #pt = np.array([self.latent.data[3].x, self.latent.data[3].y]).T.flatten().round(decimals=4).astype(str)
            self.mybox.value = " ".join(pt)
        except:
            return

        self.update_trails()    
        

    def get_samples(self, mybox, samplebox, path):

        if path == "A*":
            use_path = True
        else:
            use_path = False

        try:
            crd = np.array(mybox.split()).astype(float)
            crd = crd.reshape((int(len(crd)/2), 2))
        except Exception:
           raise Exception("Cannot define sampling points")
           return
    
        if use_path:
            # connect points via A*
            try:
                landscape = self.latent.data[0].z
                crd = get_path_aggregate(crd, landscape.T, self.MA.xvals, self.MA.yvals)
            except Exception as e:
               raise Exception(f"Cannot define sampling points: path finding failed. {e})")
               return
                                         
        else:
            # connect points via straight line
            try:  
                crd = oversample(crd, pts=int(samplebox))
            except Exception as e:
                raise Exception(f"Cannot define sampling points: oversample failed. {e}")
                return

        return crd

        
    def interact_3D(self, mybox, samplebox, path):
        '''
        generate and display proteins according to latent space trail
        ''' 

        try:
            crd = self.get_samples(mybox, samplebox, path)
            self.samples = crd.copy()
            crd = crd.reshape((1, len(crd), 2))
        except:
            self.button_pdb.disabled = True
            return

        if crd.shape[1] == 0:
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
        if "custom" in change.new:
            mykey = change.new.split(":")[1]
        else:
            mykey = change.new
   
        try:
           data = self.MA.surfaces[mykey]
        except Exception as e:
            print(f"{e}")
            return      
   
        if np.abs(np.max(data) - np.min(data)) < 100:
            self.block0.children[1].readout_format = '.1f'
        else:         
            self.block0.children[1].readout_format = 'd'
        
        self.latent.data[0].z = data
        
        # step below necessary to avoid situations whereby temporarily min>max
        try:
            self.latent.data[0].zmin = np.min(data)
            self.latent.data[0].zmax = np.max(data)
            self.block0.children[1].min = np.min(data)
            self.block0.children[1].max = np.max(data)
        except:
            self.latent.data[0].zmax = np.max(data)
            self.latent.data[0].zmin = np.min(data)
            self.block0.children[1].max = np.max(data)
            self.block0.children[1].min = np.min(data)
                
        self.block0.children[1].value = (np.min(data), np.max(data))
        
        self.update_trails()


    def drop_dataset_event(self, change):
        '''
        control which dataset is displayed
        '''        

        if change.new == "none":
            self.latent.data[1].x = []
            self.latent.data[1].y = []
            
        else:
            try:
               data = as_numpy(self.MA.get_encoded(change.new).squeeze(2))
            except Exception as e:
                print(f"{e}")
                return      
       
            self.latent.data[1].x = data[:, 0]
            self.latent.data[1].y = data[:, 1]

            self.latent.data[1].visible = True
        
        self.latent.update()


    def drop_path_event(self, change):
        '''
        control way paths are looked for
        '''
        if change.new == "A*":
            self.block0.children[4].disabled = True
        else:
            self.block0.children[4].disabled = False
            
        self.update_trails()


    def range_slider_event(self, change):
        '''
        update surface colouring upon manipulation of range slider
        '''
        self.latent.data[0].zmin = change.new[0]
        self.latent.data[0].zmax = change.new[1]
        self.latent.update()


    def trail_update_event(self, change):
        '''
        update trails (waypoints and way they are connected)
        '''

        try:
            crd = np.array(self.mybox.value.split()).astype(float)
            crd = crd.reshape((int(len(crd)/2), 2))
        except:
            self.button_pdb.disabled = False
            return

        self.waypoints = crd.copy()

        self.update_trails()


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

        crd = self.get_samples(self.mybox.value, self.samplebox.value, self.drop_path.value)
        self.samples = crd.copy()
        crd = crd.reshape((1, len(crd), 2))
        
        if crd.shape[1] == 0:
            return

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

        try:
            self.MA, self.waypoints = pickle.load( open( fname, "rb" ) )
            self.run()
        except Exception as e:
            raise Exception(f"Cannot load state file. {e}")

    #####################################################

    def run(self):

        # create an MDAnalysis instance of input protein (for viewing purposes)
        if hasattr(self.MA, "mol"):
            self.MA.mol.write_pdb("tmp.pdb", conformations=[0], split_struc = False)
            self.mymol = mda.Universe('tmp.pdb')
        
        ### MENU ITEMS ###
        
        # surface representation dropdown menu
        options = []
        if self.MA is not None:
            for f in list(self.MA.surfaces):
                options.append(f)

        if len(options)>0:
            val = options
        else:
            val = ["none"]

        self.drop_background = widgets.Dropdown(
            options=val,
            value=val[0],
            description='Surf.:',
            layout=Layout(flex='1 1 0%', width='auto'))

        if len(options) == 0:
            self.drop_background.disabled = True
        
        self.drop_background.observe(self.drop_background_event, names='value')


        # dataset selector dropdown menu
        options2 = ["none"]
        if self.MA is not None:
            for f in list(self.MA._datasets):
                options2.append(f)

        self.drop_dataset = widgets.Dropdown(
            options=options2,
            value=options2[0],
            description='Dataset:',
            layout=Layout(flex='1 1 0%', width='auto'))

        if len(options2) == 1:
            self.drop_dataset.disabled = True
        else:
            self.drop_dataset.disabled = False
        
        self.drop_dataset.observe(self.drop_dataset_event, names='value')

        # pathfinder method dropdown menu
        self.drop_path = widgets.Dropdown(
            options=["Euclidean", "A*"],
            value="Euclidean",
            description='Path:',
            layout=Layout(flex='1 1 0%', width='auto'))

        self.drop_path.observe(self.drop_path_event, names='value')


        # text box holding current coordinates
        self.mybox = widgets.Textarea(placeholder='coordinates',
                                 description='crds:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.mybox.observe(self.trail_update_event, names='value')

        # text box holding number of sampling points
        self.samplebox = widgets.Text(value='10',
                                 description='sampling:',
                                 disabled=False, layout=Layout(flex='1 1 0%', width='auto'))

        self.samplebox.observe(self.trail_update_event, names='value')


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

        if self.MA is None:
            self.button_save_state.disabled = True
            self.button_pdb.disabled = True
            
        if self.waypoints == []:
            self.button_pdb.disabled = True

        
        ### LATENT SPACE REPRESENTATION ###

        # surface 
        if len(options)>0:
            sc = self.MA.surfaces[options[0]]
        else:
            sc = []
            
        if len(sc)>0:
            plot1 = go.Heatmap(x=self.MA.xvals, y=self.MA.yvals, z=sc, zmin=np.min(sc), zmax=np.max(sc),
                               colorscale='viridis', name="latent_space")   
        else:

            if self.MA is not None:
                self.MA.setup_grid(samples=50)
                xvals, yvals = self.MA.xvals, self.MA.yvals
            else:               
                xvals = np.linspace(0, 1, 10)
                yvals = np.linspace(0, 1, 10)
                    
            surf_empty = np.zeros((len(xvals), len(yvals)))
            plot1 = go.Heatmap(x=xvals, y=yvals, z=surf_empty, opacity=0.0, showscale=False, name="latent_space")   
                      
        # dataset
        if self.MA is not None and len(list(self.MA._datasets))>0:
                  
            mydata = as_numpy(self.MA.get_encoded(options2[1]).squeeze(2))
            color = "white" if len(sc)>0 else "black"
            plot2 = go.Scatter(x=mydata[:, 0].flatten(),
                              y=mydata[:, 1].flatten(),
                  showlegend=False, opacity=0.9, mode="markers",
                  marker=dict(color=color, size=5), name=options2[1], visible=False)
        else:
            plot2 = go.Scatter(x=[], y=[])

        # path
        plot3 = go.Scatter(x=np.array([]), y=np.array([]),
                   showlegend=False, opacity=0.9, mode = 'lines+markers',
                   marker=dict(color='red', size=4))

        self.latent = go.FigureWidget([plot1, plot2, plot3])
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

        # 3D protein representation (triggered by update of textbox, sampling box, or pathfinding method)
        self.protein = widgets.interactive_output(self.interact_3D, {'mybox': self.mybox, 'samplebox': self.samplebox, 'path': self.drop_path})

        
        ### WIDGETS ARRANGEMENT ###
        
        self.block0 = widgets.VBox([self.drop_dataset, self.range_slider,
                                    self.drop_background, self.drop_path, self.samplebox, self.mybox,
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
            self.mybox.value = " ".join(self.waypoints.flatten().astype(str))

        display.clear_output(wait=True)
        display.display(self.scene)
            
