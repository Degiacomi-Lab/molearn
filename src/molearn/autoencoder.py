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


class FromLatentDimensions(nn.Module):
    def __init__(self, latent_size=2, init_n=26, sn=False):
        super().__init__()
        self.f = nn.Linear(latent_size, latent_size*init_n)
        self.latent_size = latent_size
        self.init_n=init_n

    def forward(self, x):
        x = x.view(x.size(0), self.latent_size)
        x = self.f(x)
        x = x.view(x.size(0), self.latent_size, self.init_n)
        return x
class ToLatentDimensions(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()
        self.latent_size = latent_size

    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(self.latent_size,1))
        z = torch.sigmoid(z)
        return  z

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv1d(channels, int(channels), 3,1,1,bias=False),
            nn.BatchNorm1d(int(channels)),
            nn.ReLU(),
            nn.Conv1d(int(channels), channels, 3,1,1,bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.resblock(x))

class Decoder(nn.Module):
    def __init__(self, latent_dim, init_n, n_channels):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            FromLatentDimensions(latent_dim, init_n),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyConvTranspose1d(512, 4, stride=2, padding=0), #64 -> 128
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(512),
            Residual_Block(512),
            nn.LazyConvTranspose1d(256, 4, stride=2, padding=1), #128 -> 256
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(256),
            Residual_Block(256),

            nn.LazyConvTranspose1d(128, 4, stride=2, padding=1), #256 -> 512
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(128),
            Residual_Block(128),
            nn.LazyConvTranspose1d(64, 4, stride=2, padding=1), #512 -> 1024
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(64),
            Residual_Block(64),
            nn.LazyConvTranspose1d(n_channels, 4, stride=2, padding=1) # 1024->2048
        )
    def forward(self, z):
        x = self.net(z) # torch.util.data.checkpoint
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim, init_n):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            nn.LazyConv1d(64, 4, stride=1, padding=0), #2048 -> 1024
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(64),
            Residual_Block(64),
            nn.LazyConv1d(128, 4, stride=2, padding=1), #1024 -> 512
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(128),
            Residual_Block(128),
            nn.LazyConv1d(256, 4, stride=2, padding=1), #512 -> 256
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(256),
            Residual_Block(256),
            nn.LazyConv1d(512, 4, stride=2, padding=1), # 256 -> 128
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            Residual_Block(512),
            Residual_Block(512),
            nn.LazyConv1d(3, 4, stride=2, padding=1), # 128 -> 64
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            ToLatentDimensions(latent_dim),
        )
    def forward(self, z):
        x = self.net(z) # torch.util.data.checkpoint
        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim, init_n,**kwargs):
        super(Autoencoder, self).__init__()
        print('unused kwargs ', kwargs)
        self.encoder = Encoder(latent_dim, init_n)
        self.decoder = Decoder(latent_dim, init_n,3)


    def forward(self,z):
        self.decoder(z)

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)
