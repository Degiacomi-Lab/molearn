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

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        raise ImportError('The default networks has been changed, import old from molearn.old_networks.autoencoder.ResidualBlock or better networks from molearn.autoencoder.ResidualBlock')

class To2D(nn.Module):
    def __init__(self, *args, **kwargs):
        raise ImportError('The default networks has been changed, import old from molearn.old_networks.autoencoder.To2D or better networks from molearn.autoencoder.ToLatentDimension')

class From2D(nn.Module):
    def __init__(self, *args, **kwargs):
        raise ImportError('The default networks has been changed, import old from molearn.old_networks.autoencoder.From2D or better networks from molearn.autoencoder.FromLatentDimension')

class Autoencoder(nn.Module):
    def __init__(self, *args, **kwargs):
        raise ImportError('The default networks has been changed, import old from molearn.old_networks.autoencoder.Autoencoder or better networks from molearn.autoencoder.Autoencoder')
