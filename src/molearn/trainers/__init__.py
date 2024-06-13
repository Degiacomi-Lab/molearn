# Copyright (c) 2022 Samuel C. Musson
#
# Molearn is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# Molightning is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with molightning ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
"""
trainers holds classes for training networks
"""

from .trainer import *
from .torch_physics_trainer import *



class RaiseErrorOnInit:
    module = 'unknown module is creating an ImportError'
    def __init__(self,*args, **kwargs):
        raise ImportError(f'{self.module}. Therefore {self.__class__.__name__} can not be used')
try:
    from .openmm_physics_trainer import *
except ImportError as e:
    import warnings
    warnings.warn(f"{e}. OpenMM or openmmtorchplugin are not installed. If this is needed please install with `mamba install -c conda-forge openmmtorchplugin=1.1.3 openmm`")
try:
    from .sinkhorn_trainer import *
except ImportError as e:
    warnings.warn(f"{e}. sinkhorn is not installed. If this is needed please install with `pip install geomloss`")

