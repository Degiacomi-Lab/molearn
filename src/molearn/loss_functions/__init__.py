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
`loss_functions` holds the classes for calculating OpenMM energy, and backup methods for calculating energy with just pyTorch
"""
from .openmm_thread import openmm_energy, OpenmmPluginScore, OpenMMPluginScoreSoftForceField
from .torch_protein_energy import TorchProteinEnergy
