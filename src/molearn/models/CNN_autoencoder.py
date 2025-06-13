# Copyright (c) 2022 Samuel C. Musson
#
# Molightning is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# Molightning is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molightning ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, f):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm1d(f),
                      nn.ReLU(inplace=True),
                      nn.Conv1d(f, f, 3, stride=1, padding=1, bias=False),
                      nn.BatchNorm1d(f)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
        # return torch.relu(x + self.conv_block(x))       #earlier runs were with 'return x + self.conv_block(x)' but not an issue (really?)


class To2D(nn.Module):
    
    def __init__(self):
        super(To2D, self).__init__()
        pass
    
    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2, 1))
        z = torch.sigmoid(z)
        return z


class From2D(nn.Module):
    def __init__(self):
        super(From2D, self).__init__()
        self.f = nn.Linear(2, 26*2)    

    def forward(self, x):
        x = x.view(x.size(0), 2)
        x = self.f(x)
        x = x.view(x.size(0), 2, 26)
        return x


class AutoEncoder(nn.Module):    
    '''
    This is the autoencoder used in our `Ramaswamy 2021 paper <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011052>`_.
    It is largely superseded by :func:`molearn.models.foldingnet.AutoEncoder`.
    '''
    def __init__(self, init_z=32, latent_z=1, depth=4, m=1.5, r=0, droprate=None):    
        '''
        :param int init_z: number of channels in first layer
        :param int latent_z: number of latent space dimensions
        :param int depth: number of layers
        :param float m: scaling factor, dictating number of channels in subsequent layers
        :param int r: number of residual blocks between layers
        :param float droprate: dropout rate
        '''
        
        super(AutoEncoder, self).__init__()    
        # encoder block    
        eb = nn.ModuleList()    
        eb.append(nn.Conv1d(3, init_z, 4, 2, 1, bias=False))    
        eb.append(nn.BatchNorm1d(init_z))    
        if droprate is not None:    
            eb.append(nn.Dropout(p=droprate))    
        eb.append(nn.ReLU(inplace=True))    
        for i in range(depth):
            eb.append(nn.Conv1d(int(init_z*m**i), int(init_z*m**(i+1)), 4, 2, 1, bias=False))
            eb.append(nn.BatchNorm1d(int(init_z*m**(i+1))))
            if droprate is not None:
                eb.append(nn.Dropout(p=droprate))
            eb.append(nn.ReLU(inplace=True))
            for j in range(r):
                eb.append(ResidualBlock(int(init_z*m**(i+1))))
        eb.append(nn.Conv1d(int(init_z*m**depth), latent_z, 4, 2, 1, bias=False))
        eb.append(To2D())
        self.encoder = eb
        # decoder block
        db = nn.ModuleList()
        db.append(From2D())
        db.append(nn.ConvTranspose1d(latent_z, int(init_z*m**(depth+1)), 4, 2, 1, bias=False))
        db.append(nn.BatchNorm1d(int(init_z*m**(depth+1))))
        if droprate is not None:
            db.append(nn.Dropout(p=droprate))
        db.append(nn.ReLU(inplace=True))
        for i in reversed(range(depth+1)):
            db.append(nn.ConvTranspose1d(int(init_z*m**(i+1)), int(init_z*m**i), 4, 2, 1, bias=False))
            db.append(nn.BatchNorm1d(int(init_z*m**i)))
            if droprate is not None:
                db.append(nn.Dropout(p=droprate))
            db.append(nn.ReLU(inplace=True))
            for j in range(r):
                db.append(ResidualBlock(int(init_z*m**i)))
        db.append(nn.ConvTranspose1d(int(init_z*m**(i)), 3, 4, 2, 1))
        self.decoder = db
        
    def encode(self, x):
        for m in self.encoder:
            x = m(x)
        return x
    
    def decode(self, x):
        for m in self.decoder:
            x = m(x)
        return x
