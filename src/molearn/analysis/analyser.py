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
#
# Authors: Matteo Degiacomi, Samuel Musson

from __future__ import annotations
import os
from copy import deepcopy
import numpy as np
import torch.optim
from pathlib import Path
from typing import Union

try:
    # from modeller import *
    from modeller.selection import Selection
    from modeller.environ import Environ
    from modeller.scripts import complete_pdb
except Exception as e:
    print("Error importing modeller: ")
    print(e)

try:
    from ..scoring import Parallel_DOPE_Score
except ImportError as e:
    print(
        "Import Error captured while trying to import Parallel_DOPE_Score, it is likely that you dont have Modeller installed"
    )
    print(e)
try:
    from ..scoring import Parallel_Ramachandran_Score
except ImportError as e:
    print(
        "Import Error captured while trying to import Parallel_Ramachandran_Score, it is likely that you dont have cctbx/iotbx installed"
    )
    print(e)
from ..data import PDBData

from ..utils import as_numpy
from tqdm import tqdm
import warnings

from openmm.app.modeller import Modeller
from openmm.app.forcefield import ForceField
from openmm.app.pdbfile import PDBFile

# from openmm.app import PME
from openmm.app import NoCutoff
from openmm.openmm import VerletIntegrator
from openmm.app.simulation import Simulation
from openmm.unit import picoseconds

warnings.filterwarnings("ignore")


class MolearnAnalysis:
    """
    This class provides methods dedicated to the quality analysis of a
    trained model.
    """

    def __init__(self):
        self._datasets = {}
        self._encoded = {}
        self._decoded = {}
        self.surfaces = {}
        self.batch_size = 1
        self.processes = 1

    def set_network(self, network):
        """
        :param network: a trained neural network defined in :func:`molearn.models <molearn.models>`
        """
        self.network = network
        self.network.eval()
        self.device = next(network.parameters()).device

    def get_dataset(self, key):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        """
        return self._datasets[key]

    def set_dataset(self, key, data, atomselect="protein"):
        """
        :param data: :func:`PDBData <molearn.data.PDBData>` object containing atomic coordinates
        :param str key: label to be associated with data
        :param list/str atomselect: list of atom names to load, or 'protein' to indicate that all atoms are loaded.
        """
        if isinstance(data, str) and data.endswith(".pdb"):
            d = PDBData()
            d.import_pdb(data)
            d.atomselect(atomselect)
            d.prepare_dataset()
            _data = d
        elif isinstance(data, PDBData):
            _data = data
        else:
            raise NotImplementedError("data should be an PDBData instance")
        for _key, dataset in self._datasets.items():
            assert (
                dataset.shape[2] == _data.dataset.shape[2]
                and dataset.shape[1] == _data.dataset.shape[1]
            ), f"number of d.o.f differes: {key} has shape {_data.shape} while {_key} has shape {dataset.shape}"
        self._datasets[key] = _data.dataset.float()
        if not hasattr(self, "meanval"):
            self.meanval = _data.mean
        if not hasattr(self, "stdval"):
            self.stdval = _data.std
        if not hasattr(self, "atoms"):
            self.atoms = _data.atoms
        if not hasattr(self, "mol"):
            self.mol = _data.frame()
        if not hasattr(self, "shape"):
            self.shape = (_data.dataset.shape[1], _data.dataset.shape[2])

    def get_encoded(self, key):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :return: array containing the encoding in latent space of dataset associated with key
        """
        if key not in self._encoded:
            assert key in self._datasets, (
                f"key {key} does not exist in internal _datasets or in _latent_coords, add it with MolearnAnalysis.set_latent_coords(key, torch.Tensor) "
                "or add the corresponding dataset with MolearnAnalysis.set_dataset(name, PDBDataset)"
            )
            with torch.no_grad():
                dataset = self.get_dataset(key)
                batch_size = self.batch_size
                encoded = None
                for i in tqdm(
                    range(0, dataset.shape[0], batch_size), desc=f"encoding {key}"
                ):
                    z = self.network.encode(
                        dataset[i : i + batch_size].to(self.device)
                    ).cpu()
                    if encoded is None:
                        encoded = torch.empty(dataset.shape[0], z.shape[1], z.shape[2])
                    encoded[i : i + batch_size] = z
                self._encoded[key] = encoded

        return self._encoded[key]

    def set_encoded(self, key, coords):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        """
        self._encoded[key] = torch.tensor(coords).float()

    def get_decoded(self, key):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        """
        if key not in self._decoded:
            with torch.no_grad():
                batch_size = self.batch_size
                encoded = self.get_encoded(key)
                decoded = torch.empty(encoded.shape[0], *self.shape).float()
                for i in tqdm(
                    range(0, encoded.shape[0], batch_size), desc=f"Decoding {key}"
                ):
                    decoded[i : i + batch_size] = self.network.decode(
                        encoded[i : i + batch_size].to(self.device)
                    )[:, :, : self.shape[1]].cpu()
                self._decoded[key] = decoded
        return self._decoded[key]

    def set_decoded(self, key, structures):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        """
        self._decoded[key] = structures

    def num_trainable_params(self):
        """
        :return: number of trainable parameters in the neural network previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_network>`
        """
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def get_error(self, key, align=True):
        """
        Calculate the reconstruction error of a dataset encoded and decoded by a trained neural network.

        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param bool align: if True, the RMSD will be calculated by finding the optimal alignment between structures
        :return: 1D array containing the RMSD between input structures and their encoded-decoded counterparts
        """
        dataset = self.get_dataset(key)
        z = self.get_encoded(key)
        decoded = self.get_decoded(key)

        err = []
        m = deepcopy(self.mol)
        for i in range(dataset.shape[0]):
            crd_ref = (
                as_numpy(dataset[i].permute(1, 0).unsqueeze(0)) * self.stdval
                + self.meanval
            )
            crd_mdl = (
                as_numpy(decoded[i].permute(1, 0).unsqueeze(0))[:, : dataset.shape[2]]
                * self.stdval
                + self.meanval
            )  # clip the padding of models
            # use Molecule Biobox class to calculate RMSD
            if align:
                m.coordinates = deepcopy(crd_ref)
                m.set_current(0)
                m.add_xyz(crd_mdl[0])
                rmsd = m.rmsd(0, 1)
            else:
                rmsd = np.sqrt(
                    np.sum((crd_ref.flatten() - crd_mdl.flatten()) ** 2)
                    / crd_mdl.shape[1]
                )  # Cartesian L2 norm

            err.append(rmsd)

        return np.array(err)

    def get_dope(self, key, refine=True, **kwargs):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param bool refine: if True, refine structures before calculating DOPE score
        :return: dictionary containing DOPE score of dataset, and its decoded counterpart
        """
        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)

        dataset_dope = self.get_all_dope_score(dataset, refine=refine, **kwargs)
        decoded_dope = self.get_all_dope_score(decoded, refine=refine, **kwargs)

        return dict(dataset_dope=dataset_dope, decoded_dope=decoded_dope)

    def get_ramachandran(self, key):
        """
        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        """

        dataset = self.get_dataset(key)
        decoded = self.get_decoded(key)

        ramachandran = {
            f"dataset_{key}": value
            for key, value in self.get_all_ramachandran_score(dataset).items()
        }
        ramachandran.update(
            {
                f"decoded_{key}": value
                for key, value in self.get_all_ramachandran_score(decoded).items()
            }
        )
        return ramachandran

    def get_bondlengths(self, key):
        """
        Get backbone bond lengths of a dataset and its decoded counterpart.

        """
        # Get the atomic indices to calculate different types of bond lengths
        if set(['CA','C', 'N','CB']).issubset(set(self.atoms)):
            indices = {'N-Ca':[], 'Ca-C':[], 'C-N':[], 'CA-CB':[]}
        elif set(['CA','C', 'N']).issubset(set(self.atoms)):
            indices = {'N-Ca':[], 'Ca-C':[], 'C-N':[]}
        else:
            raise ValueError(f"Selected atoms should contain at least N, CA, and C.")

        mol_df = self.mol.data
        for resid in mol_df.resid.unique():
            resname = mol_df[mol_df['resid'] == resid].resname.unique()[0]
            
            N_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'N')].index[0]
            CA_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'CA')].index[0]    
            C_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'C')].index[0]
            indices['N-Ca'].append((N_id, CA_id))
            indices['Ca-C'].append((CA_id, C_id))
            if resname != 'GLY' and 'CB' in self.atoms:
                CB_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'CB')].index[0]
                indices['Ca-Cb'].append((CA_id, CB_id))

            if resid != len(mol_df.resid.unique()):
                next_N_id = mol_df[(mol_df['resid'] == (resid+1)) & (mol_df['name'] == 'N')].index[0]
                indices['C-N'].append((C_id, next_N_id))
        
        # Look for the key in self._datasets and self._encoded
        if key in self._datasets.keys():
            dataset = self.get_dataset(key)*self.stdval + self.meanval
            decoded = self.get_decoded(key)*self.stdval + self.meanval
            dataset_bondlen = {k: MolearnAnalysis._bond_lengths(dataset, v) for k, v in indices.items()}
            decoded_bondlen = {k: MolearnAnalysis._bond_lengths(decoded, v) for k, v in indices.items()}
            return dict(dataset_bondlen=dataset_bondlen, decoded_bondlen=decoded_bondlen)            
        elif key in self._encoded.keys():
            decoded = self.get_decoded(key)*self.stdval + self.meanval
            decoded_bondlen = {k: MolearnAnalysis._bond_lengths(decoded, v) for k, v in indices.items()}
            return dict(decoded_bondlen=decoded_bondlen)
        else:
            raise ValueError(f"Key {key} not found in _datasets or _encoded. Please load the dataset or setup a grid first.")
        

    def setup_grid(self, samples=64, bounds_from=None, bounds=None, padding=0.1):
        """
        Define a NxN point grid regularly sampling the latent space.

        :param int samples: grid size (build a samples x samples grid)
        :param str/list bounds_from: Name(s) of datasets to use as reference, either as single string, a list of strings, or 'all'
        :param tuple/list bounds: tuple (xmin, xmax, ymin, ymax) or None
        :param float padding: define size of extra spacing around boundary conditions (as ratio of axis dimensions)
        """

        key = "grid"
        if bounds is None:
            if bounds_from is None:
                bounds_from = "all"

            bounds = self._get_bounds(bounds_from, exclude=key)

        bx = (bounds[1] - bounds[0]) * padding
        by = (bounds[3] - bounds[2]) * padding
        self.xvals = np.linspace(bounds[0] - bx, bounds[1] + bx, samples)
        self.yvals = np.linspace(bounds[2] - by, bounds[3] + by, samples)
        self.n_samples = samples
        meshgrid = np.meshgrid(self.xvals, self.yvals)
        stack = np.stack(meshgrid, axis=2).reshape(-1, 1, 2)
        self.set_encoded(key, stack)

        return key

    def _get_bounds(self, bounds_from, exclude=["grid", "grid_decoded"]):
        """
        :param bounds_from: keys of datasets to be considered for identification of boundaries in latent space
        :param exclude: keys of dataset not to consider
        :return: four scalars as edges of x and y axis: xmin, xmax, ymin, ymax
        """
        if isinstance(exclude, str):
            exclude = [
                exclude,
            ]

        if bounds_from == "all":
            bounds_from = [key for key in self._datasets.keys() if key not in exclude]
        elif isinstance(bounds_from, str):
            bounds_from = [
                bounds_from,
            ]

        xmin, ymin, xmax, ymax = [], [], [], []
        for key in bounds_from:
            z = self.get_encoded(key)
            xmin.append(z[:, 0].min())
            ymin.append(z[:, 1].min())
            xmax.append(z[:, 0].max())
            ymax.append(z[:, 1].max())

        xmin, ymin = min(xmin), min(ymin)
        xmax, ymax = max(xmax), max(ymax)
        return xmin, xmax, ymin, ymax

    def scan_error_from_target(self, key, index=None, align=True):
        """
        Calculate landscape of RMSD vs single target structure. Target should be previously loaded datset containing a single conformation.

        :param str key: key pointing to a dataset previously loaded with :func:`set_dataset <molearn.analysis.MolearnAnalysis.set_dataset>`
        :param int index: index of conformation to be selected from dataset containing multiple conformations.
        :param bool align: if True, structures generated from the grid are aligned to target prior RMSD calculation.
        :return: RMSD latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        """
        s_key = (
            f"RMSD_from_{key}" if index is None else f"RMSD_from_{key}_index_{index}"
        )
        if s_key not in self.surfaces:
            assert (
                "grid" in self._encoded
            ), "make sure to call MolearnAnalysis.setup_grid first"
            target = (
                self.get_dataset(key)
                if index is None
                else self.get_dataset(key)[index].unsqueeze(0)
            )
            if target.shape[0] != 1:
                msg = f"dataset {key} shape is {target.shape}. \
A dataset with a single conformation is expected.\
Either pass a key that points to a single structure or pass the index of the \
structure you want, e.g., analyser.scan_error_from_target(key, index=0)"
                raise Exception(msg)

            decoded = self.get_decoded("grid")
            if align:
                crd_ref = as_numpy(target.permute(0, 2, 1)) * self.stdval
                crd_mdl = as_numpy(decoded.permute(0, 2, 1)) * self.stdval
                m = deepcopy(self.mol)
                m.coordinates = np.concatenate([crd_ref, crd_mdl])
                m.set_current(0)
                rmsd = np.array([m.rmsd(0, i) for i in range(1, len(m.coordinates))])
            else:
                rmsd = (
                    (((decoded - target) * self.stdval) ** 2)
                    .sum(axis=1)
                    .mean(axis=-1)
                    .sqrt()
                )
            self.surfaces[s_key] = as_numpy(
                rmsd.reshape(self.n_samples, self.n_samples)
            )

        return self.surfaces[s_key], self.xvals, self.yvals

    def scan_error(self, s_key="Network_RMSD", z_key="Network_z_drift"):
        """
        Calculate RMSD and z-drift on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.

        :param str s_key: label for RMSD dataset
        :param str z_key: label for z-drift dataset
        :return: input-to-decoded RMSD latent space NxN surface
        :return: z-drift latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        """
        s_key = "Network_RMSD"
        z_key = "Network_z_drift"
        if s_key not in self.surfaces:
            assert (
                "grid" in self._encoded
            ), "make sure to call MolearnAnalysis.setup_grid first"
            decoded = self.get_decoded("grid")  # decode grid
            # self.set_dataset('grid_decoded', decoded)   # add back as dataset w. different name
            self._datasets["grid_decoded"] = decoded
            decoded_2 = self.get_decoded(
                "grid_decoded"
            )  # encode, and decode a second time
            grid = self.get_encoded("grid")  # retrieve original grid
            grid_2 = self.get_encoded("grid_decoded")  # retrieve decoded encoded grid

            rmsd = (
                (((decoded - decoded_2) * self.stdval) ** 2)
                .sum(axis=1)
                .mean(axis=-1)
                .sqrt()
            )
            z_drift = ((grid - grid_2) ** 2).mean(axis=2).mean(axis=1).sqrt()

            self.surfaces[s_key] = rmsd.reshape(self.n_samples, self.n_samples).numpy()
            self.surfaces[z_key] = z_drift.reshape(
                self.n_samples, self.n_samples
            ).numpy()

        return self.surfaces[s_key], self.surfaces[z_key], self.xvals, self.yvals

    def _ramachandran_score(self, frame):
        """
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        """
        if not hasattr(self, "ramachandran_score_class"):
            self.ramachandran_score_class = Parallel_Ramachandran_Score(
                self.mol, self.processes
            )
        assert (
            len(frame.shape) == 2
        ), f"We wanted 2D data but got {len(frame.shape)} dimensions"
        if frame.shape[0] == 3:
            f = frame.permute(1, 0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f, torch.Tensor):
            f = f.data.cpu().numpy()

        return self.ramachandran_score_class.get_score(f * self.stdval)
        # nf, na, no, nt = self.ramachandran_score_class.get_score(f*self.stdval)
        # return {'favored':nf, 'allowed':na, 'outliers':no, 'total':nt}

    def _dope_score(self, frame, refine=True, **kwargs):
        """
        returns multiprocessing AsyncResult
        AsyncResult.get() will return the result
        """
        if not hasattr(self, "dope_score_class"):
            self.dope_score_class = Parallel_DOPE_Score(self.mol, self.processes)

        assert (
            len(frame.shape) == 2
        ), f"We wanted 2D data but got {len(frame.shape)} dimensions"
        if frame.shape[0] == 3:
            f = frame.permute(1, 0)
        else:
            assert frame.shape[1] == 3
            f = frame
        if isinstance(f, torch.Tensor):
            f = f.data.cpu().numpy()

        return self.dope_score_class.get_score(f * self.stdval, refine=refine, **kwargs)

    @staticmethod
    def _ca_chirality(N, CA, C, CB):
        """
        Calculate chirality of Cα atom in a protein residue.

        :param N: Cartesian coordinates of N atom
        :param CA: Cartesian coordinates of CA atom
        :param C: Cartesian coordinates of C atom
        :param CB: Cartesian coordinates of CB atom
        :return: dot product of normal vector to the plane defined by N, CA, and C
        """
        ca_c = C - CA
        ca_cb = CB - CA
        ca_n = N - CA
        normal = np.cross(ca_n, ca_c)
        dot_product = np.dot(normal, ca_cb)
        # L if dot_product > 0 else D
        return dot_product

    @staticmethod
    def _bond_lengths(crds, indices):
        bond_lengths = [np.linalg.norm(crds[:,:,i[0]] - crds[:,:,i[1]], axis=1) for i in indices]
        return np.array(bond_lengths)
    
    def get_all_ramachandran_score(self, tensor):
        """
        Calculate Ramachandran score of an ensemble of atomic conrdinates.

        :param tensor:
        """
        rama = dict(favored=[], allowed=[], outliers=[], total=[])
        results = []
        for f in tensor:
            results.append(self._ramachandran_score(f))
        for r in tqdm(results, desc="Calc rama"):
            favored, allowed, outliers, total = r.get()
            rama["favored"].append(favored)
            rama["allowed"].append(allowed)
            rama["outliers"].append(outliers)
            rama["total"].append(total)
        return {key: np.array(value) for key, value in rama.items()}

    def get_all_dope_score(self, tensor, refine=True):
        """
        Calculate DOPE score of an ensemble of atom coordinates.

        :param tensor:
        :param bool refine: if True, return DOPE score of input and output structure after refinement
        """
        results = []
        for f in tensor:
            results.append(self._dope_score(f, refine=refine))
        results = np.array([r.get() for r in tqdm(results, desc="Calc Dope")])
        return results

    def reference_dope_score(self, frame):
        """
        :param numpy.array frame: array with shape [1, N, 3] with Cartesian coordinates of atoms
        :return: DOPE score
        """
        self.mol.coordinates = deepcopy(frame)
        self.mol.write_pdb("tmp.pdb", split_struc=False)
        env = Environ()
        env.libs.topology.read(file="$(LIB)/top_heav.lib")
        env.libs.parameters.read(file="$(LIB)/par.lib")
        mdl = complete_pdb(env, "tmp.pdb")
        atmsel = Selection(mdl.chains[0])
        score = atmsel.assess_dope()
        return score

    def scan_dope(self, key=None, refine=True, **kwargs):
        """
        Calculate DOPE score on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.

        :param str key: label for unrefined DOPE score surface (default is DOPE_unrefined or DOPE_refined)
        :param bool refine: if True, structures generated will be energy minimised before DOPE scoring
        :return: DOPE score latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        """

        if key is None:
            if refine == "both":
                key = "DOPE_both"
            elif refine:
                key = "DOPE_refined"
            else:
                key = "DOPE_unrefined"

        if key not in self.surfaces:
            assert (
                "grid" in self._encoded
            ), "make sure to call MolearnAnalysis.setup_grid first"
            decoded = self.get_decoded("grid")
            result = self.get_all_dope_score(decoded, refine=refine, **kwargs)
            if refine == "both":
                self.surfaces[key] = as_numpy(
                    result.reshape(self.n_samples, self.n_samples, 2)
                )
            else:
                self.surfaces[key] = as_numpy(
                    result.reshape(self.n_samples, self.n_samples)
                )

        return self.surfaces[key], self.xvals, self.yvals

    def scan_ramachandran(self):
        """
        Calculate Ramachandran scores on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.
        Saves four surfaces in memory, with keys 'Ramachandran_favored', 'Ramachandran_allowed', 'Ramachandran_outliers', and 'Ramachandran_total'.

        :return: Ramachandran_favoured latent space NxN surface (ratio of residues in favourable conformation)
        :return: x-axis values
        :return: y-axis values
        """
        keys = {
            i: f"Ramachandran_{i}" for i in ("favored", "allowed", "outliers", "total")
        }
        if list(keys.values())[0] not in self.surfaces:
            assert (
                "grid" in self._encoded
            ), "make sure to call MolearnAnalysis.setup_grid first"
            decoded = self.get_decoded("grid")
            rama = self.get_all_ramachandran_score(decoded)
            for key, value in rama.items():
                self.surfaces[keys[key]] = value

        return self.surfaces["Ramachandran_favored"], self.xvals, self.yvals

    def scan_ca_chirality(self):
        """
        Calculate chiralities of Cα atoms on a grid sampling the latent space.
        Requires a grid system to be defined via a prior call to :func:`set_dataset <molearn.analysis.MolearnAnalysis.setup_grid>`.
        Requires the atom selection includes Cα atoms.
        Saves a surface in memory, with key 'CA_chirality'.

        :return: Number of CA inversions on the latent space NxN surface
        :return: x-axis values
        :return: y-axis values
        """
        assert (
            set(['CA','C','N','CB']).issubset(set(self.atoms))
            ), "Atom selection shoud at least include CA, C, N, and CB"
        
        key = "Chirality"
        if key not in self.surfaces:
            assert (
                "grid" in self._encoded
            ), "make sure to call MolearnAnalysis.setup_grid first"
            decoded = self.get_decoded("grid")

            mol_df = self.mol.data

            # Get atom indices 
            indices = dict()
            for resid in mol_df.resid.unique():
                resname = mol_df[mol_df['resid'] == resid].resname.unique()[0]
                if not resname == 'GLY':
                    N_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'N')].index[0]
                    C_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'C')].index[0]
                    CA_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'CA')].index[0]
                    CB_id = mol_df[(mol_df['resid'] == resid) & (mol_df['name'] == 'CB')].index[0]
                    indices[resname + str(resid)] = (N_id, CA_id, C_id, CB_id)
            
            results = []
            for j in decoded:
                s = (j.view(1, 3, -1).permute(0, 2, 1) * self.stdval).numpy()
                inversions = {}
                for k, v in indices.items():
                    dot_prod = self._ca_chirality(s[0,v[0],:], 
                                                  s[0,v[1],:], 
                                                  s[0,v[2],:], 
                                                  s[0,v[3],:])
                    if dot_prod < 0:
                        inversions[k] = dot_prod
                results.append(len(inversions))
            self.surfaces[key] = np.array(results).reshape(self.n_samples, self.n_samples)

        return self.surfaces[key], self.xvals, self.yvals 

    def scan_custom(self, fct, params, key):
        """
        Generate a surface coloured as a function of a user-defined function.

        :param fct: function taking atomic coordinates as input, an optional list of parameters, and returning a single value.
        :param list params: parameters to be passed to function f. If no parameter is needed, pass an empty list.
        :param str key: name of the dataset generated by this function scan
        :return: latent space NxN surface, evaluated according to input function
        :return: x-axis values
        :return: y-axis values
        """
        decoded = self.get_decoded("grid")
        results = []
        for i, j in enumerate(decoded):
            s = (j.view(1, 3, -1).permute(0, 2, 1) * self.stdval).numpy()
            results.append(fct(s, *params))
        self.surfaces[key] = np.array(results).reshape(self.n_samples, self.n_samples)

        return self.surfaces[key], self.xvals, self.yvals

    def _relax(self, pdb_file: Union[str, Path], out_path: Union[str, Path], maxIterations: int = 1000) -> None:
        """
        Model the sidechains and relax generated structure

        :param str pdb_file: path to the pdb file generated by the model
        :param str out_path: path where the modelled/relaxed structures are be saved
        """

        if not isinstance(pdb_file, str):
            pdb_file = str(pdb_file)
        if not isinstance(out_path, str):
            out_path = str(out_path)

        # Assume sidechain modelling is required if the number of selected atoms is fewer than 6
        if len(self.atoms) < 6:
            modelled_file = out_path + os.sep + (pdb_file.stem + "_modelled.pdb")
            try:
                env = Environ()
                env.libs.topology.read(file='$(LIB)/top_heav.lib')
                env.libs.parameters.read(file='$(LIB)/par.lib')   

                mdl = complete_pdb(env, str(pdb_file))
                mdl.write(str(modelled_file))
                pdb_file = modelled_file
            except Exception as e:
                print(f'Failed to model {pdb_file}\n{e}')
        try:
            relaxed_file = out_path + os.sep + (pdb_file.stem + "_relaxed.pdb")
            # Read pdb
            pdb = PDBFile(pdb_file)
            # Add hydrogens
            forcefield = ForceField("amber99sb.xml")
            modeller = Modeller(pdb.topology, pdb.positions)
            modeller.addHydrogens(forcefield)

            system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff)
            integrator = VerletIntegrator(0.001 * picoseconds)
            simulation = Simulation(modeller.topology, system, integrator)
            simulation.context.setPositions(modeller.positions)
            # Energy minimization
            simulation.minimizeEnergy(maxIterations=maxIterations)
            positions = simulation.context.getState(getPositions=True).getPositions()
            # Write energy minimized file
            PDBFile.writeFile(simulation.topology, positions, open(relaxed_file, "w+"))
        except Exception as e:
            print(f'Failed to relax {pdb_file}\n{e}')
    
    def _pdb_file(
        self,
        prot_coords: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        pdb_file: str,
    ) -> None:
        """
        create pdb file for given coordinates

        :param np.ndarray[tuple[int, int], np.dtype[np.float64]] prot_coords: coordinates of all atoms of a protein
        :param str pdb_file: path where the pdb file should be stored
        """
        pdb_data = self.mol.data
        with open(
            pdb_file,
            "w+",
        ) as cfile:
            for ck, k in enumerate(prot_coords):
                cfile.write(
                    f"{str(pdb_data['atom'][ck]):6s}{int(pdb_data['index'][ck]):5d} {str(pdb_data['name'][ck]):^4s}{'':1s}{str(pdb_data['resname'][ck]):3s} {str(pdb_data['chain'][ck]):1s}{int(pdb_data['resid'][ck]):4d}{'':1s}   {k[0]:8.3f}{k[1]:8.3f}{k[2]:8.3f}{float(pdb_data['occupancy'][ck]):6.2f}{float(pdb_data['beta'][ck]):6.2f}          {str(pdb_data['atomtype'][ck]):>2s}{str(pdb_data['charge'][ck]):2s}\n"
                )

    def generate(
        self,
        crd: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
        pdb_path: str | None = None,
        relax: bool = False,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        """
        Generate a collection of protein conformations, given coordinates in the latent space.

        :param numpy.array crd: coordinates in the latent space, as a (Nx2) array
        :param str pdb_path: path where to pdb_files should be stored as files named sN.pdb where N is the index in the crd array
        :param bool relax: whether relaxed structures should be generated in a sN_relaxed.pdb file

        :return: collection of protein conformations in the Cartesian space (NxMx3, where M is the number of atoms in the protein)
        """
        with torch.no_grad():
            key = list(self._datasets)[0]
            # if not on cpu transfer data back to cpu before converting it to numpy array
            if self.device == "cpu":
                z = torch.tensor(crd.transpose(1, 2, 0)).float()
                s = (
                    self.network.decode(z)[:, :, : self._datasets[key].shape[2]]
                    .numpy()
                    .transpose(0, 2, 1)
                )
            else:
                z = torch.tensor(crd.transpose(1, 2, 0)).float().to(self.device)
                s = (
                    self.network.decode(z)[:, :, : self._datasets[key].shape[2]]
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 1)
                )

        gen_prot_coords = s * self.stdval + self.meanval
        # create pdb file
        if pdb_path is not None:
            for i, coord in enumerate(tqdm(gen_prot_coords, desc="Generating pdb files")):
                struct_path = os.path.join(pdb_path, f"s{i}.pdb")
                self._pdb_file(coord, struct_path)
                # relax and save as new file
                if relax:
                    self._relax(
                        struct_path, pdb_path, maxIterations=1000
                    )

        return gen_prot_coords

    def __getstate__(self):
        return {
            key: value
            for key, value in dict(self.__dict__).items()
            if key not in ["dope_score_class", "ramachandran_score_class"]
        }
