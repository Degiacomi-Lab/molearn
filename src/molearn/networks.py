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
import torch.nn.utils.spectral_norm as spectral_norm



class ResidualBlock(nn.Module):
    def __init__(self, f, sn=False, gn=False, num_groups=8):
        super().__init__()
        conv_block = [
                      spectral_norm(nn.Conv1d(f,f, 3, stride=1, padding=1, bias=False)) if sn else \
                                    nn.Conv1d(f,f, 3, stride=1, padding=1, bias=False),
                      nn.GroupNorm(num_groups,f) if gn else nn.BatchNorm1d(f),
                      nn.ReLU(inplace=True)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ToLatentDimensions(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()

    def forward(self, x):
        z = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(2,1))
        z = torch.sigmoid(z)
        return  z

class FromLatentDimensions(nn.Module):
    def __init__(self, latent_size=2, init_n=26, sn=False):
        super().__init__()
        self.f = spectral_norm(nn.Linear(latent_size, latent_size*init_n)) if sn else nn.Linear(latent_size, latent_size*init_n)
        self.latent_size = latent_size
        self.init_n=init_n

    def forward(self, x):
        x = x.view(x.size(0), self.latent_size)
        x = self.f(x)
        x = x.view(x.size(0), self.latent_size, self.init_n)
        return x

class Encoder(nn.Module):
    def __init__(self, init_z=32, latent_z=2, depth=4, m=2.0, r=2, use_spectral_norm=False,use_group_norm=False, num_groups=8,
                init_n=26):
        super().__init__()
        sn = use_spectral_norm # rename for brevity
        # encoder block
        eb = nn.ModuleList()
        eb.append(spectral_norm(nn.Conv1d(3, init_z, 4, 2, 1, bias=False)) if sn else nn.Conv1d(3, init_z, 4, 2, 1, bias=False))
        eb.append(nn.GroupNorm(num_groups, init_z) if use_group_norm else nn.BatchNorm1d(init_z))
        eb.append(nn.ReLU(inplace=True))
        for j in range(r):
            eb.append(ResidualBlock(int(init_z), sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        for i in range(depth):
            eb.append(spectral_norm(nn.Conv1d(int(init_z*m**i), int(init_z*m**(i+1)), 4, 2, 1, bias=False)) if sn else \
                                    nn.Conv1d(int(init_z*m**i), int(init_z*m**(i+1)), 4, 2, 1, bias=False))
            eb.append(nn.GroupNorm(num_groups, int(init_z*m**(i+1))) if use_group_norm else nn.BatchNorm1d(int(init_z*m**(i+1))))
            eb.append(nn.ReLU(inplace=True))
            for j in range(r):
                eb.append(ResidualBlock(int(init_z*m**(i+1)), sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        eb.append(spectral_norm(nn.Conv1d(int(init_z*m**depth), latent_z, 4, 2, 1, bias=False)) if sn else \
                                nn.Conv1d(int(init_z*m**depth), latent_z, 4, 2, 1, bias=False))
        for j in range(r):
            eb.append(ResidualBlock(latent_z, sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        eb.append(ToLatentDimensions(latent_size=latent_z))
        self.model = nn.Sequential(*eb)

    def forward(self, x):
        #expect input of shape [B,3,N]
        return self.model(x)

class Decoder(nn.Module):

    def __init__(self, init_z=32, latent_z=2, depth=4, m=2.0, r=2, use_spectral_norm=False,use_group_norm=False, num_groups=8,
                init_n=26):
        super().__init__()
        sn = use_spectral_norm # rename for brevity
        self.latent_z = latent_z
        # decoder block
        db = nn.ModuleList()
        db.append(FromLatentDimensions(latent_size=latent_z, init_n=init_n, sn=use_spectral_norm))
        db.append(spectral_norm(nn.ConvTranspose1d(latent_z, int(init_z*m**(depth)), 4, 2, 1, bias=False)) if sn else \
                                nn.ConvTranspose1d(latent_z, int(init_z*m**(depth)), 4, 2, 1, bias=False))
        db.append(nn.GroupNorm(num_groups, int(init_z*m**(depth))) if use_group_norm else nn.BatchNorm1d(int(init_z*m**(depth))))
        db.append(nn.ReLU(inplace=True))
        for j in range(r):
            db.append(ResidualBlock(int(init_z*m**(depth)), sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        for i in reversed(range(depth)):
            db.append(spectral_norm(nn.ConvTranspose1d(int(init_z*m**(i+1)), int(init_z*m**i), 4, 2, 1, bias=False)) if sn else \
                                    nn.ConvTranspose1d(int(init_z*m**(i+1)), int(init_z*m**i), 4, 2, 1, bias=False))
            db.append(nn.GroupNorm(num_groups, int(init_z*m**i)) if use_group_norm else nn.BatchNorm1d(int(init_z*m**i)))
            db.append(nn.ReLU(inplace=True))
            for j in range(r):
                db.append(ResidualBlock(int(init_z*m**i), sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        db.append(spectral_norm(nn.ConvTranspose1d(int(init_z), 3, 4, 2, 1)) if sn else \
                                nn.ConvTranspose1d(int(init_z), 3, 4, 2, 1))
        for j in range(r):
            db.append(ResidualBlock(3, sn=use_spectral_norm, gn=use_group_norm, num_groups=num_groups))
        self.model = nn.Sequential(*db)

    def forward(self, x):
        #in lightning, forward defines the prediction/inference actions
        #expect input of shape [B,Latent_size,1]
        return self.model(x.view(x.size(0), self.latent_z,1))

class Autoencoder(nn.Module):
    def __init__(self, init_z=32, latent_z=2, depth=4, m=2.0, r=2, use_spectral_norm=False,use_group_norm=False, num_groups=8,
                init_n=26):
        super().__init__()
        self.encoder = Encoder(init_z, latent_z, depth, m, r, use_spectral_norm, use_group_norm, num_groups, init_n)
        self.decoder = Decoder(init_z, latent_z, depth, m, r, use_spectral_norm, use_group_norm, num_groups, init_n)

    def forward(self, x):
        return self.decoder(self.encoder(x))[:,:,:x.shape[2]]

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)

class NewAutoencoder(nn.Module):
    def __init__(self, kwargs=None, encoder_kwargs=None, decoder_kwargs=None):
        super().__init__()
        self.encoder = Encoder(**dict(kwargs, **encoder_kwargs))
        self.decoder = Decoder(**dict(kwargs, **decoder_kwargs))

    def forward(self, x):
        return self.decoder(self.encoder(x))[:,:,:x.shape[2]]

    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)


