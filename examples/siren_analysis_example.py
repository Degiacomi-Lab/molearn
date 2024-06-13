 #%%
import sys, os
sys.path.insert(0, os.path.join(os.path.abspath(os.pardir),'src'))
from molearn.data import PDBData
from molearn.trainers import OpenMM_Physics_Trainer
from molearn.models.siren import AutoEncoder
import torch
from molearn.analysis import MolearnAnalysis
import matplotlib.pyplot as plt
from molearn.utils import as_numpy

#%%
# Specify FUll_DATASETS
if __name__ =='__main__':
    FULL_DATASETS = True
        
    if not FULL_DATASETS:
        device = torch.device('cuda')
        MA_processes = 16
        #torch.set_num_threads(16)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        MA_processes = 16
#%%
# Load network 
if __name__ == '__main__':
    print("> Loading network parameters...")

    fname = f'siren_checkpoints_nmr_alignres{os.sep}checkpoint_epoch220_loss-0.49344822612859435.ckpt'
    # change 'cpu' to 'cuda' if you have a suitable cuda enabled device
    checkpoint = torch.load(fname, map_location=torch.device('cpu'))
    #checkpoint = torch.load(fname, map_location=torch.device('cuda'))
    net = AutoEncoder(**checkpoint['network_kwargs'])
    net.load_state_dict(checkpoint['model_state_dict'])
# %%
# Import data
if __name__ == '__main__':
    # what follows is a method to re-create the training and test set
    # by defining the manual see and loading the dataset in the same order as when
    #the neural network was trained, the same train-test split will be obtained
    data = PDBData()
    data.import_pdb(f'data{os.sep}aggregated_align_1-84.pdb')
    #data.import_pdb(f'data{os.sep}MurD_closed_selection.pdb')
    #data.import_pdb(f'data{os.sep}MurD_open_selection.pdb')
    #data.import_pdb('/home3/pghw87/trajectories/MurD/MurD_closed.pdb')
    #data.import_pdb('/home3/pghw87/trajectories/MurD/MurD_open.pdb')
    data.fix_terminal()
    data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    data.prepare_dataset()
    data_train, data_valid = data.split(manual_seed=25)
    # data_test = PDBData()
    # data_test.import_pdb(f'data{os.sep}MurD_closed_apo_selection.pdb')
    # data_test.std = data.std
    # data_test.mean = data.mean
    # data_test.fix_terminal()
    # data_test.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
    # data_test.prepare_dataset()
#%%
# Created MA object, import train and valid data
if __name__ == '__main__':
    print("> Loading training data...")

    MA = MolearnAnalysis()
    MA.set_network(net)

    # increasing the batch size makes encoding/decoding operations faster,
    # but more memory demanding
    MA.batch_size = 4

    # increasing processes makes DOPE and Ramachandran scores calculations faster,
    # but more more memory demandingTrue
    MA.processes = 2
#%%
    # print out the size of train, valid, test data
# if __name__ == '__main__':
#     print(f'Train set shape {data_train.dataset.shape}, Valid set shape {data_valid.dataset.shape}, Test set shape {data_test.dataset.shape}')
#%%
# store the training and valid set in the MolearnAnalysis instance
# the second parameter of the sollowing commands can be both a PDBData instance
# or a path to a multi-PDB file
if __name__ == '__main__':
    MA.set_dataset("training", data_train)
    MA.set_dataset("valid", data_valid)
    #MA.set_dataset('test', data_test)

# %%
# Calculate RMSD of training and test set
if __name__ == '__main__':
    print("> calculating RMSD of training, validation and testing sets")

    err_train = MA.get_error('training')
    err_valid = MA.get_error('valid')
    #err_test = MA.get_error('test')

    #print(f'Mean RMSD is {err_train.mean()} for training set and {err_valid.mean()} for valid set and {err_test.mean()} for test set')
    print(f'Mean RMSD is {err_train.mean()} for training set and {err_valid.mean()} for valid set')
#%%
if __name__ == '__main__':
    # Plot RMSD 
    fig, ax = plt.subplots()
    #violin = ax.violinplot([err_train, err_valid, err_test], showmeans = True, )
    violin = ax.violinplot([err_train, err_valid], showmeans = True, )
    #ax.set_xticks([1,2,3])
    ax.set_xticks([1,2])
    #ax.set_title('RMSD of training, validation and testing sets')
    ax.set_title('RMSD of training and validation for residues 1-84 aligned (NMR structures) with Siren')
    #ax.set_xticklabels(['Training', 'Validation','Testing'])
    ax.set_xticklabels(['Training', 'Validation'])
    fig.gca().set_ylabel(r'RMSD ($ \AA$)')
    plt.savefig('RMSD_plot.png')
# %%
# Generate error landscape
if __name__ == '__main__':
    print("> generating error landscape")
    # build a 50x50 grid. By default, it will be 10% larger than the region occupied
    # by all loaded datasets
    if FULL_DATASETS:
        MA.setup_grid(64)
    else:
        MA.setup_grid(10)
    landscape_err_latent, landscape_err_3d, xaxis, yaxis = MA.scan_error()

    # Plot RMSD grid in latent space
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xaxis, yaxis, landscape_err_latent, vmax = 8)
    fig.colorbar(c, label = 'latent space RMSD ($ \AA$)')
    ax.set_title("Error grid (RMSD) in latent space")
    coords = as_numpy(MA.get_encoded('valid'))
    #tcoords = as_numpy(MA.get_encoded('test'))
    plt.scatter(coords[:,0,0], coords[:,1,0], label = 'Validation')
    #plt.scatter(tcoords[:,0,0], tcoords[:,1,0], label ='Testing')
    plt.savefig('Error_rmsd_grid.png')

    # Plot drift grid in latent space
    fig, ax = plt.subplots()
    c = ax.pcolormesh(xaxis, yaxis, landscape_err_3d, vmax = 1)
    fig.colorbar(c, label = 'latent space drfit ($ \AA$)')
    ax.set_title("Error grid (drift) in latent space")
    coords = as_numpy(MA.get_encoded('valid'))
    #tcoords = as_numpy(MA.get_encoded('test'))
    plt.scatter(coords[:,0,0], coords[:,1,0], label = 'Validation')
    #plt.scatter(tcoords[:,0,0], tcoords[:,1,0], label ='Testing')
    plt.savefig('Error_drift_grid.png')

#%%
# Generate dope scores
if __name__ == '__main__':
    dope, xvals, yvals = MA.scan_dope()

#%%
# Plot dope grid
if __name__ == '__main__':
    fig, ax  = plt.subplots()
    c = ax.pcolormesh(xvals, yvals, dope)
    ax.set_title("Dope score of residues 0-83 aligned in latent space of Siren")
    fig.colorbar(c, label = 'latent space DDOPE score ($ \AA$)')
    coords = as_numpy(MA.get_encoded('valid'))
    #tcoords = as_numpy(MA.get_encoded('test'))
    plt.scatter(coords[:,0,0], coords[:,1,0], label = 'Valid', c='white')
    #plt.scatter(tcoords[:,0,0], tcoords[:,1,0], label ='Test')
    #ax.legend()
    plt.savefig('Dope_grid.png')

    ## to visualise the GUI, execute the code above in a Jupyter notebook, then call:
    # from molearn.analysis import MolearnGUI
    # MolearnGUI(MA)
#%%
# Generate ramachandran score
if __name__ == '__main__':
    ramachandran, xvals, yvals = MA.scan_ramachandran()

#%%
# Plot ramachandran grid
if __name__ == '__main__':
    fig, ax  = plt.subplots()
    c = ax.pcolormesh(xvals, yvals, ramachandran)
    fig.colorbar(c)
    ax.set_title("Ramachandran grid in latent space")
    plt.scatter(coords[:,0,0], coords[:,1,0], label = 'Valid')
    plt.scatter(tcoords[:,0,0], tcoords[:,1,0], label ='Test')
    plt.savefig('Ramachandran.png')
# %%
