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


__author__ = "V. K. Ramaswamy, S. C. Musson, C. G. Willcocks, M. T. Degiacomi"
__version__ = '1.1'
__date__ = '$Date: 2022-10-26 $'
from .loss_functions import Auto_potential
from .protein_handler import *
from .networks import *
from .molearn_trainer import Molearn_Trainer, Molearn_Physics_Trainer, OpenMM_Physics_Trainer, Molearn_Constrained
from .pdb_data import PDBData
from .openmm_loss import openmm_energy
<<<<<<< Updated upstream
=======
#from .analysis import MolearnAnalysis, MolearnGUI
>>>>>>> Stashed changes

try:
    from .sinkhorn_trainer import Sinkhorn_Trainer
except Exception as e:
    import warnings
    warnings.warn(f"{e}. Will not be able to use sinkhorn_trainer.")
    
