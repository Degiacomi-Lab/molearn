import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np


# create the GON network (a SIREN as in https://vsitzmann.github.io/siren/)
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class Decoder(nn.Module):
    def __init__(self, dimensions, *args, **kwargs):
        '''
        :param dimensions: list the number of nodes in each layer. dimensions[0] will be the number of input features, dimnensions[-1] will be the number of output features. dimensions[1:-1] will determine the number hidden layers depth
        '''
        super().__init__()
        first_layer = SirenLayer(dimensions[0], dimensions[1], is_first=True)
        other_layers = []
        for dim0, dim1 in zip(dimensions[1:-2], dimensions[2:-1]):
            other_layers.append(SirenLayer(dim0, dim1))
        final_layer = SirenLayer(dimensions[-2], dimensions[-1], is_last=True)
        self.model =  nn.Sequential(first_layer, *other_layers, final_layer)

    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, mol, latent_dimensions=2, hidden_dimensions = None, positional_encoding = True, atom_encoding= True,  positional_encoding_dimensions=32, *args, **kwargs):
        super().__init__()
        if hidden_dimensions is None:
            dimensions = [latent_dimensions, 512, 512, 512, 3]
        else:
            dimensions = [latent_dimensions, *hidden_dimensions, 3]
        posenc = self.get_positional_encoding(mol.coordinates.shape[1], positional_encoding_dimensions) if positional_encoding else torch.empty(mol.coordinates.shape[1],0)
        atomenc = self.get_atom_encoding(mol) if atom_encoding else torch.empty(mol.coordinates.shape[1],0)
        encoding = torch.cat([posenc, atomenc], dim=-1).unsqueeze(0)
        self.register_buffer('encoding', encoding)
        self.latent_dimensions = latent_dimensions
        dimensions[0]+=self.encoding.shape[-1]


        self.decoder = Decoder(dimensions, *args, **kwargs)


    @staticmethod
    def get_positional_encoding(number_of_atoms, encoding_dimensions=32):
        '''
        sin cosine Positional encoding as typically used in transformers. see Attention is all you need paper
        This is implemented here because it is widely used in a different task and it's performance in this task has not been profiled. I expect there are much better ways of doing this.
        :param number_of_atoms: How many atoms to encode for
        :param encoding_dimensions: How many encoding dimensions to use
        :returns: torch.tensor shape [number_of_atoms, encoding_dimensions]
        '''
        position = torch.arange(number_of_atoms).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dimensions, 2) * (-math.log(10000.0) / encoding_dimensions)).unsqueeze(0)
        pe = torch.zeros(number_of_atoms, encoding_dimensions)
        pe[:,0::2] = torch.sin(div_term*position)
        pe[:,1::2] = torch.cos(div_term*position)
        return pe

    @staticmethod
    def get_atom_encoding(mol):
        '''
        One hot encoding of atom types
        :param mol: pass a biobox molecule containing at least one frame of the protein as example
        :returns: torch.tensor shape [number_of_atoms, number_of_unique_atomtypes]
        '''
        mol.assign_atomtype()
        atomtypes = mol.data['name'].values
        unique = np.unique(atomtypes)
        encoding = torch.zeros(len(atomtypes),len(unique))
        for i,u in enumerate(unique):
            encoding[atomtypes==u,i] = 1.0
        assert (encoding.sum(dim=1)==1).all()
        return encoding



    def decode(self, z):
        '''
        :param z: shape [B, 1, Z]
        '''
        z = z.reshape(z.shape[0],1, self.latent_dimensions)
        encoding = self.encoding.repeat(z.shape[0],1,1)
        z_rep = z.repeat(1, encoding.size(1), 1)
        return self.decoder(torch.cat((encoding, z_rep), dim=-1)).permute(0,2,1)


    def refine_encoding(self, latent, structure):
        '''
        WARNING: this will set requires_grad_ in place on latent
        :param latent: initial guess of latent tensor shape [B, 1, Z] torch.zeros or torch.randn usually
        :param structure: structure tensor to encode latent to
        '''
        grad_needed = torch.is_grad_enabled()
        with torch.enable_grad():
            g = self.decode(latent.requires_grad_())
            L_inner = ((g-structure)**2).sum(1).mean()
            grad = torch.autograd.grad(L_inner, [latent], create_graph=grad_needed, retain_graph=grad_needed)[0]
        return latent - grad

    def encode(self, structure):
        '''
        :param structure: torch.tensor shape [Batch_size, n_atoms, 3]
        '''
        z = torch.zeros(structure.shape[0], 1, self.latent_dimensions).to(structure.device)
        return self.refine_encoding(z, structure)

    def forward(self, x):
        return self.decode(x)


if __name__=='__main__':
    print('Nothing to see here')