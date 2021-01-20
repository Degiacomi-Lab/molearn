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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from copy import deepcopy
import biobox

from molearn import load_data
from molearn import Auto_potential
from molearn import Autoencoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

floc = ["./test/MurD_test.pdb"] # test protein (contains only 16 conformations of the MurD protein)
batch_size = 4 # if this is too small, gpu utilization goes down
epoch = 0
iter_per_epoch = 5 #use higher iter_per_epoch = 1000 for smoother plots (iter_per_epoch = smoothness  of statistics)
method = 'roll' # 3 method for  available in Auto_potential: 'roll', 'convolutional', 'indexing'

# load multiPDB and create loss function
dataset, meanval, stdval, atom_names, mol, test0, test1 = load_data(floc[0], atoms = ["CA", "C", "N", "CB", "O"], device=device)
lf = Auto_potential(frame=dataset[0]*stdval, pdb_atom_names=atom_names, method = method, device=device)

# Saving test structures (the most extreme conformations in terms of RMSD)
# Remember to rescale with stdval, permute axis from [3,N] to [N,3]
# unsqueeze to [1, N, 3], send back to cpu, and convert to numpy array.
crds =  (test0*stdval).permute(1,0).unsqueeze(0).data.cpu().numpy()
mol.coordinates = crds
mol.write_pdb("TEST0.pdb")

crds =  (test1*stdval).permute(1,0).unsqueeze(0).data.cpu().numpy()
mol.coordinates = crds
mol.write_pdb("TEST1.pdb")

# helper function to make getting another batch of data easier
def cycle(iterable):
   while True:
       for x in iterable:
           yield x

########################################################################
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dataset.float()),
    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=6)
iterator = iter(cycle(train_loader))

# define networks
network = Autoencoder(m=2.0, latent_z=2, r=2).to(device)
print("> Network parameters: ", len(torch.nn.utils.parameters_to_vector(network.parameters())))
# define optimisers
optimiser = torch.optim.Adam(network.parameters(), lr=0.001, amsgrad=True)

#training loop
while (epoch<200):
    print("> epoch: ", epoch)
    for i in range(iter_per_epoch):
        print(i)
        # get two batches of training data
        x0 = next(iterator)[0].to(device)
        x1 = next(iterator)[0].to(device)
        optimiser.zero_grad()

        #encode
        z0 = network.encode(x0)
        z1 = network.encode(x1)

        #interpolate
        alpha = torch.rand(x0.size(0), 1, 1).to(device)
        z_interpolated = (1-alpha)*z0 + alpha*z1

        #decode
        out = network.decode(z0)[:,:,:x0.size(2)]
        out_interpolated = network.decode(z0)[:,:,:x0.size(2)]

        #calculate MSE
        mse_loss = ((x0-out)**2).mean() # reconstructive loss (Mean square error)
        out *= stdval
        out_interpolated *= stdval

        #calculate physics for interpolated samples
        bond_energy, angle_energy, torsion_energy, NB_energy = lf.get_loss(out_interpolated)
        #by being enclosed in torch.no_grad() torch autograd cannot see where this scaling
        #factor came from and hence although mathematically the physics cancels, no gradients
        #are found and the scale is simply redefined at each step
        with torch.no_grad():
            scale = 0.1*mse_loss.item()/(bond_energy.item()+angle_energy.item()+torsion_energy.item()+NB_energy.item())

        network_loss = mse_loss + scale*(bond_energy + angle_energy + torsion_energy + NB_energy)

        #determine gradients
        network_loss.backward()

        #advance the network weights
        optimiser.step()

    #save interpolations between test0 and test1 every 5 epochs
    if epoch%5 == 0:
        interpolation_out = torch.zeros(20, x0.size(2), 3)
        #encode test with each network
        #Not training so switch to eval mode
        network.eval()
        with torch.no_grad(): # don't need gradients for this bit
            test0_z = network.encode(test0.unsqueeze(0).float())
            test1_z = network.encode(test1.unsqueeze(0).float())

            #interpolate between the encoded Z space for each network between test0 and test1
            for idx, t in enumerate(np.linspace(0, 1, 20)):
                interpolation_out[idx] = network.decode(float(t)*test0_z + (1-float(t))*test1_z)[:,:,:x0.size(2)].squeeze(0).permute(1,0).cpu().data
            interpolation_out *= stdval

        # Remember to switch back to train mode when you are done
        network.train()

        #save interpolations
        mol.coordinates = interpolation_out.numpy()
        mol.write_pdb("epoch_%s_interpolation.pdb"%epoch)
