import torch

from molearn.models.foldingnet import AutoEncoder
from molearn.analysis import MolearnAnalysis
from molearn.data import PDBData


print("> Loading network parameters...")

fname = 'xbb_foldingnet_checkpoints\\checkpoint_no_optimizer_state_dict_epoch167_loss0.003259085263643.ckpt'
checkpoint = torch.load(fname, map_location=torch.device('cpu'))
net = AutoEncoder(**checkpoint['network_kwargs'])
net.load_state_dict(checkpoint['model_state_dict'])

print("> Loading training data...")

MA = MolearnAnalysis()
MA.set_network(net)

# increasing the batch size makes encoding/decoding operations faster,
# but more memory demanding
MA.batch_size = 4

# increasing processes makes DOPE and Ramachandran scores calculations faster,
# but more more memory demanding
MA.processes = 2

# what follows is a method to re-create the training and test set
# by defining the manual see and loading the dataset in the same order as when
#the neural network was trained, the same train-test split will be obtained
data = PDBData()
data.import_pdb('data\\MurD_closed_selection.pdb')
data.import_pdb('data\\MurD_open_selection.pdb')
data.fix_terminal()
data.atomselect(atoms = ['CA', 'C', 'N', 'CB', 'O'])
data.prepare_dataset()
data_train, data_test = data.split(manual_seed=25)

# store the training and test set in the MolearnAnalysis instance
# the second parameter of the sollowing commands can be both a PDBData instance
# or a path to a multi-PDB file
MA.set_dataset("training", data_train)
MA.set_dataset("test", data_test)

print("> calculating RMSD of training and test set")

err_train = MA.get_error('training')
err_test = MA.get_error('test')

print("> generating error landscape")
# build a 50x50 grid. By default, it will be 10% larger than the region occupied
# by all loaded datasets
MA.setup_grid(50)
landscape_err_latent, landscape_err_3d, xaxis, yaxis = MA.scan_error()

## to visualise the GUI, execute the code above in a Jupyter notebook, then call:
# from molearn.analysis import MolearnGUI
# MolearnGUI(MA)