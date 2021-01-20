# Copyright (c) 2014-2017 Matteo Degiacomi
#
# BiobOx is free software ;
# you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
# either version 2 of the License, or (at your option) any later version.
# BiobOx is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with BiobOx ;
# if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
# Author : Matteo Degiacomi, matteothomas.degiacomi@gmail.com

import os
from copy import deepcopy
import numpy as np
import scipy.signal
import pandas as pd

# Definiton of constants for later calculations
epsilon0 = 8.8542 * 10**(-12) # m**-3 kg**-1 s**4 A**2, Permitivitty of free space
kB = 1.3806 * 10**(-23) # m**2 kg s**-2 K-1, Lattice Boltzmann constant
e = 1.602 * 10**(-19) # A s, electronic charge
m = 1 * 10**(-9) # number of nm in 1 m
c = 3.336 * 10**(-30) # conversion from debye to e m
Na = 6.022 * 10**(23) # Avagadros Number

from biobox.classes.structure import Structure

class Molecule(Structure):
    '''
    Subclass of :func:`Structure <structure.Structure>`, allows reading, manipulating and analyzing molecular structures.
    '''

    chain_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd',
                   'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                   'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    def __init__(self):
        '''
        At instantiation, properties associated to every individual atoms are stored in a pandas Dataframe self.data.
        The columns of the self.data have the following names:
        atom, index, name, resname, chain, resid, beta, occupancy, atomtype, radius, charge.

        self.knowledge contains a knowledge base about atoms and residues properties. Default values are:

        * 'residue_mass' property stores the average mass for most common aminoacids (values from Expasy website)
        * 'atom_vdw' vdw radius of common atoms
        * 'atom_mass' mass of common atoms

        The knowledge base can be edited. For instance, to add information about residue "TST" mass in molecule M type: M.knowledge['mass_residue']["TST"]=142.42
        '''

        super(Molecule, self).__init__(r=np.array([]))

        # knowledge base about atoms and residues properties (entry keys:
        # 'residue_mass', 'atom_vdw', 'atom_, mass' can be edited)
        self.knowledge = {}
        self.knowledge['residue_mass'] = {"ALA": 71.0788, "ARG": 156.1875, "ASN": 114.1038, "ASP": 115.0886, "CYS": 103.1388, "CYX": 103.1388, "GLU": 129.1155, "GLN": 128.1307, "GLY": 57.0519,
                                          "HIS": 137.1411, "HSE": 137.1411, "HSD": 137.1411, "HSP": 137.1411, "HIE": 137.1411, "HID": 137.1411, "HIP": 137.1411, "ILE": 113.1594, "LEU": 113.1594,
                                          "LYS": 128.1741, "MET": 131.1926, "MSE": 131.1926, "PHE": 147.1766, "PRO": 97.1167, "SER": 87.0782, "THR": 101.1051, "TRP": 186.2132, "TYR": 163.1760, "VAL": 99.1326}
        self.knowledge['atom_vdw'] = {'H': 1.20, 'N': 1.55, 'NA': 2.27, 'CU': 1.40, 'CL': 1.75, 'C': 1.70, 'O': 1.52, 'I': 1.98, 'P': 1.80, 'B': 1.85, 'BR': 1.85, 'S': 1.80, 'SE': 1.90,
                                      'F': 1.47, 'FE': 1.80, 'K': 2.75, 'MN': 1.73, 'MG': 1.73, 'ZN': 1.39, 'HG': 1.8, 'XE': 1.8, 'AU': 1.8, 'LI': 1.8, '.': 1.8}
        self.knowledge['atom_ccs'] = {'H': 1.2, 'C': 1.91, 'N': 1.91, 'O': 1.91, 'P': 1.91, 'S': 1.91, '.': 1.91}
        self.knowledge['atom_mass'] = {"H": 1.00794, "D": 2.01410178, "HE": 4.00, "LI": 6.941, "BE": 9.01, "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.998403, "NE": 20.18, "NA": 22.989769,
                                       "MG": 24.305, "AL": 26.98, "SI": 28.09, "P": 30.973762, "S": 32.065, "CL": 35.453, "AR": 39.95, "K": 39.0983, "CA": 40.078, "SC": 44.96, "TI": 47.87, "V": 50.94,
                                       "CR": 51.9961, "MN": 54.938045, "FE": 55.845, "CO": 58.93, "NI": 58.6934, "CU": 63.546, "ZN": 65.409, "GA": 69.72, "GE": 72.64, "AS": 74.9216, "SE": 78.96,
                                       "BR": 79.90, "KR": 83.80, "RB": 85.47, "SR": 87.62, "Y": 88.91, "ZR": 91.22, "NB": 92.91, "MO": 95.94, "TC": 98.0, "RU": 101.07, "RH": 102.91, "PD": 106.42,
                                       "AG": 107.8682, "CD": 112.411, "IN": 114.82, "SN": 118.71, "SB": 121.76, "TE": 127.60, "I": 126.90447, "XE": 131.29, "CS": 132.91, "BA": 137.33, "PR": 140.91,
                                       "EU": 151.96, "GD": 157.25, "TB": 158.93, "W": 183.84, "IR": 192.22, "PT": 195.084, "AU": 196.96657, "HG": 200.59, "PB": 207.2, "U": 238.03}
        self.knowledge['atomtype'] = {"C": "C", "CA": "C", "CB": "C", "CG": "C", "CG1": "C", "CG2": "C", "CZ": "C", "CD1": "C", "CD2": "C",
                                      "CD": "C", "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C", "CZ2": "C", "CZ3": "C", "CH2": "C",
                                      "N": "N", "NH1": "N", "NH2": "N", "NZ": "N", "NE": "N", "NE1": "N", "NE2": "N", "ND1": "N", "ND2": "N",
                                      "O": "O", "OG": "O", "OG1": "O", "OG2": "O", "OD1": "O", "OD2": "O", "OE1": "O", "OE2": "O", "OH": "O", "OXT": "O",
                                      "SD": "S", "SG": "S", "H": "H", "HA": "H", "HB1": "H", "HB2": "H", "HE1": "H", "HE2": "H", "HD1": "H", "HD2": "H", 
                                      "H1": "H", "H2": "H", "H3": "H", "HH11": "H", "HH12": "H", "HH21": "H", "HH22": "H", "HG1": "H", "HG2": "H", "HE21": "H", 
                                      "HE22": "H", "HD11": "H", "HD12": "H", "HD13": "H", "HD21": "H", "HD22": "H", "HG11": "H", "HG12": "H", "HG13": "H", 
                                      "HG21": "H", "HG22": "H", "HG23": "H", "HZ2": "H", "HZ3": "H", "HZ": "H", "HA1": "H", "HA2": "H", "HB": "H", "HD3": "H", 
                                      "HG": "H", "HZ1": "H", "HE3": "H", "HB3": "H", "HH1": "H", "HH2": "H", "HD23": "H", "HD13": "H", "HE": "H", "HH": "H", 
                                      "OC1": "O", "OC2": "O", "OW": "O", "HW1": "H", "HW2": "H", "CH3" : "C", "HH31" : "H", "HH32" : "H", "HH33" : "H",
                                      "C00" : "C", "C01" : "C", "C02" : "C", "C04" : "C", "C06" : "C", "C08" : "C", "H03" : "H", "H05" : "H", "H07" : "H",
                                      "H09" : "H", "H0A" : "H", "H0B" : "H", }

    def import_pdb(self, pdb, include_hetatm=False):
        '''
        read a pdb (possibly containing containing multiple models).

        Models are split according to ENDMDL and END statement.
        All alternative coordinates are expected to have the same atoms.
        After loading, the first model (M.current_model=0) will be set as active.

        :param pdb: PDB filename
        :param include_hetatm: if True, HETATM will be included (they get skipped if False)
        '''

        try:
            f_in = open(pdb, "r")
        except Exception as ex:
            raise Exception('ERROR: file %s not found!' % pdb)

        # store filename
        self.properties["filename"] = pdb

        data_in = []
        p = []
        r = []
        e = []
        alternative = []
        biomt = []
        symm = []
        for line in f_in:
            record = line[0:6].strip()

            # load biomatrix, if any is present
            if "REMARK 350   BIOMT" in line:
                try:
                    biomt.append(line.split()[4:8])
                except Exception as ex:
                    raise Exception("ERROR: biomatrix format seems corrupted")

            # load symmetry matrix, if any is present
            if "REMARK 290   SMTRY" in line:
                try:
                    symm.append(line.split()[4:8])
                except Exception as ex:
                    raise Exception("ERROR: symmetry matrix format seems corrupted")

            # if a complete model was parsed store all the saved data into
            # self.data entries (if needed) and temporary alternative
            # coordinates list
            if record == "ENDMDL" or record == "END":

                if len(alternative) == 0:

                    # load all the parsed data in superclass data (Dataframe)
                    # and points data structures
                    try:
                        #building dataframe
                        data = np.array(data_in).astype(str)
                        cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                        idx = np.arange(len(data))
                        self.data = pd.DataFrame(data, index=idx, columns=cols)
                        # Set the index numbers to the idx values to avoid hexadecimal counts
                        self.data["index"] = idx

                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # saving vdw radii
                    try:
                        self.data['radius'] = np.array(r)
                    except Exception as ex:
                        raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' %pdb)

                    # save default charge state
                    self.data['charge'] = np.array(e)

                # save 3D coordinates of every atom and restart the accumulator
                try:
                    if len(p) > 0:
                        alternative.append(np.array(p))
                    p = []
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when loading the structure %s!\nERROR: are all the columns separated?' % pdb)

            if record == 'ATOM' or (include_hetatm and record == 'HETATM'):

                # extract xyz coordinates (save in list of point coordinates)
                p.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                # if no complete model has been yet parsed, load also
                # information about atoms(resid, resname, ...)
                if len(alternative) == 0:
                    w = []
                    # extract ATOM/HETATM statement
                    w.append(line[0:6].strip())
                    w.append(line[6:12].strip())  # extract atom index
                    w.append(line[12:17].strip())  # extract atomname
                    w.append(line[17:21].strip())  # extract resname
                    w.append(line[21].strip())  # extract chain name
                    w.append(line[22:26].strip())  # extract residue id

                    # extract occupancy
                    try:
                        w.append(float(line[54:60]))
                    except Exception as ex:
                        w.append(1.0)

                    # extract beta factor
                    try:
                        # w.append("{0.2f}".format(float(line[60:66])))
                        w.append(float(line[60:66]))
                    except Exception as ex:
                        w.append(0.0)

                    # extract atomtype
                    try:
                        w.append(line[76:78].strip())
                    except Exception as ex:
                        w.append("")

                    # use atomtype to extract vdw radius
                    try:
                        r.append(self.know('atom_vdw')[line[76:78].strip()])
                    except Exception as ex:
                        r.append(self.know('atom_vdw')['.'])

                    # assign default charge state of 0
                    e.append(0.0)

                    data_in.append(w)

        f_in.close()

        # if p list is not empty, that means that the PDB file does not finish with an END statement (like the ones generated by SBT, for instance).
        # In this case, dump all the remaining stuff into alternate coordinates
        # array and (if needed) into properties dictionary.
        if len(p) > 0:

            # if no model has been yet loaded, save also information in
            # properties dictionary.
            if len(alternative) == 0:

                # load all the parsed data in superclass properties['data'] and
                # points data structures
                try:
                    #building dataframe
                    data = np.array(data_in).astype(str)
                    cols = ["atom", "index", "name", "resname", "chain", "resid", "beta", "occupancy", "atomtype"]
                    idx = np.arange(len(data))
                    self.data = pd.DataFrame(data, index=idx, columns=cols)
                    # Set the index numbers to the idx values to avoid hexadecimal counts
                    self.data["index"] = idx

                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving data in %s!\nERROR: are all the columns separated?' %pdb)

                try:
                    self.data['radius'] = np.array(r)
                except Exception as ex:
                    raise Exception('ERROR: something went wrong when saving van der Waals radii in %s!\nERROR: are all the columns separated?' % pdb)

                # save default charge state
                self.properties['charge'] = np.array(e)

            # save 3D coordinates of every atom and restart the accumulator
            try:
                if len(p) > 0:
                    alternative.append(np.array(p))
                p = []
            except Exception as ex:
                raise Exception('ERROR: something went wrong when saving coordinates in %s!\nERROR: are all the columns separated?' %pdb)

        # transform the alternative temporary list into a nice multiple
        # coordinates array
        if len(alternative) > 0:
            try:
                alternative_xyz = np.array(alternative).astype(float)
            except Exception as e:
                alternative_xyz = np.array([alternative[0]]).astype(float)
                print('WARNING: found %s models, but their atom count differs' % len(alternative))
                print('WARNING: treating only the first model in file %s' % pdb)
                #raise Exception('ERROR: models appear not to have the same amount of atoms')

            self.add_xyz(alternative_xyz)
        else:
            raise Exception('ERROR: something went wrong when saving alternative coordinates in %s!\nERROR: no model was loaded... are ENDMDL statements there?' % pdb)

        # if biomatrix information is provided, creat
        if len(biomt) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(biomt), 3):
                raise Exception('ERROR: found %s BIOMT entries. A multiple of 3 is expected'%len(biomt))

            b = np.array(biomt).astype(float).reshape((len(biomt) / 3, 3, 4))
            self.properties["biomatrix"] = b

        # if symmetry information is provided, create entry in properties
        if len(symm) > 0:

            # test whether there are enough lines to create biomatrix
            # statements
            if np.mod(len(symm), 3):
                raise Exception('ERROR: found %s SMTRY entries. A multiple of 3 is expected'%len(symm))

            b = np.array(symm).astype(float).reshape((len(symm) / 3, 3, 4))
            self.properties["symmetry"] = b

        #correctly set types of columns requiring other than string
        self.data["resid"] = self.data["resid"].astype(int)
        self.data["index"] = self.data["index"].astype(int)
        self.data["occupancy"] = self.data["occupancy"].astype(float)
        self.data["beta"] = self.data["beta"].astype(float)

    def write_pdb(self, outname, conformations=[], index=[]):
        '''
        overload superclass method for writing (multi)pdb.

        :param outname: name of pdb file to be generated.
        :param index: indices of atoms to write to file. If empty, all atoms are returned. Index values obtaineable with a call like: index=molecule.atomselect("A", [1, 2, 3], "CA", True)[1]
        :param conformations: list of conformation indices to write to file. By default, a multipdb with all conformations will be produced.
        '''

        # store current frame, so it will be reestablished after file output is
        # complete
        currentbkp = self.current

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        f_out = open(outname, "w")

        for f in frames:
            # get all informations from PDB (for current conformation) in a list
            self.set_current(f)
            d = self.get_pdb_data(index)
            
            # Build our hexidecimal array if num. of atoms > 99999
            idx_val = np.arange(1, len(d) + 1, 1)
            if len(idx_val) > 99999:
                vhex = np.vectorize(hex)
                idx_val = vhex(idx_val)   # convert index values to hexidecimal
                idx_val = [num[2:] for num in idx_val]  # remove 0x at start of hexidecimal number
            
            for i in range(0, len(d), 1):
                # create and write PDB line
                if d[i][2][0].isdigit():
                    L = '%-6s%5s %-5s%-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                else:
                    L = '%-6s%5s %-4s %-4s%1s%4s    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n' % (d[i][0], idx_val[i], d[i][2], d[i][3], d[i][4], d[i][5], float(d[i][6]), float(d[i][7]), float(d[i][8]), float(d[i][9]), float(d[i][10]), d[i][11])
                f_out.write(L)

            f_out.write("END\n")

        f_out.close()

        self.set_current(currentbkp)

        return

    def atomselect(self, chain, res, atom, get_index=False, use_resname=False):
        '''
        Select specific atoms in the protein providing chain, residue ID and atom name.

        :param chain: selection of a specific chain name (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param res: residue ID of desired atoms (accepts * as wildcard). Can also be a list or numpy array of of int.
        :param atom: name of desired atom (accepts * as wildcard). Can also be a list or numpy array of strings.
        :param get_index: if set to True, returns the indices of selected atoms in self.points array (and self.data)
        :param use_resname: if set to True, consider information in "res" variable as resnames, and not resids
        :returns: coordinates of the selected points and, if get_index is set to true, their indices in self.points array.
        '''

        # chain name boolean selector
        if isinstance(chain, str):
            if chain == '*':
                chain_query = np.array([True] * len(self.points))
            else:
                chain_query = self.data["chain"].values == chain
                
        elif isinstance(chain, list) or type(chain).__module__ == 'numpy':
            chain_query = self.data["chain"].values == chain[0]
            for c in range(1, len(chain), 1):
                chain_query = np.logical_or(chain_query, self.data["chain"].values == chain[c])
        else:
            raise Exception("ERROR: wrong type for chain selection. Should be str, list, or numpy")

        if isinstance(res, str):
            if res == '*':
                res_query = np.array([True] * len(self.points))
            elif use_resname:
                res_query = self.data["resname"].values == res
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, int):
            if use_resname:
                res_query = self.data["resname"].values == str(res)
            else:
                res_query = self.data["resid"].values == res

        elif isinstance(res, list) or type(res).__module__ == 'numpy':
            if use_resname:
                res_query = self.data["resname"].values == str(res[0])
            else:
                res_query = self.data["resid"].values == res[0]

            for r in range(1, len(res), 1):
                if use_resname:
                    res_query = np.logical_or(res_query, self.data["resname"].values == str(res[r]))
                else:
                    res_query = np.logical_or(res_query, self.data["resid"].values == res[r])

        else:
            raise Exception("ERROR: wrong type for resid selection. Should be int, list, or numpy")

        # atom name boolean selector
        if isinstance(atom, str):
            if atom == '*':
                atom_query = np.array([True] * len(self.points))
            else:
                atom_query = self.data["name"].values == atom
        elif isinstance(atom, list) or type(atom).__module__ == 'numpy':
            atom_query = self.data["name"].values == atom[0]
            for a in range(1, len(atom), 1):
                atom_query = np.logical_or(atom_query, self.data["name"].values == atom[a])
        else:
            raise Exception("ERROR: wrong type for atom selection. Should be str, list, or numpy")

        # slice data array and return result (colums 5 to 7 contain xyz coords)
        query = np.logical_and(np.logical_and(chain_query, res_query), atom_query)


        if get_index:
            return [self.points[query], np.where(query == True)[0]]
        else:
            return self.points[query]

    def know(self, prop):
        '''
        return information from knowledge base

        :param prop: desired property to extract from knowledge base
        :returns: value associated to requested property, or nan if failed
        '''
        if str(prop) in self.knowledge:
            return self.knowledge[str(prop)]
        else:
            raise Exception("entry %s not found in knowledge base!" % prop)

    def get_pdb_data(self, index=[]):
        '''
        aggregate data and point coordinates, and return in a unique data structure

        Returned data is a list containing strings for points data and floats for point coordinates
        in the same order as a pdb file, i.e.
        ATOM/HETATM, index, name, resname, chain name, residue ID, x, y, z, occupancy, beta factor, atomtype.

        :returns: list aggregated data and coordinates for every point, as string.
        '''

        if len(index) == 0:
            index = range(0, len(self.points), 1)

        # create a list containing all infos contained in pdb (point
        # coordinates and properties)
        d = []
        for i in index:
            d.append([self.data["atom"].values[i],
                      self.data["index"].values[i],
                      self.data["name"].values[i],
                      self.data["resname"].values[i],
                      self.data["chain"].values[i],
                      self.data["resid"].values[i],
                      self.points[i, 0],
                      self.points[i, 1],
                      self.points[i, 2],
                      self.data["beta"].values[i],
                      self.data["occupancy"].values[i],
                      self.data["atomtype"].values[i]])

        return d

    def get_subset(self, idxs, conformations=[]):
        '''
        Return a :func:`Molecule <molecule.Molecule>` object containing only the selected atoms and frames

        :param ixds: atoms to extract
        :param conformations: frames to extract (by default, all)
        :returns: :func:`Molecule <molecule.Molecule>` object
        '''

        # if a subset of all available frames is requested to be written,
        # select them first
        if len(conformations) == 0:
            frames = range(0, len(self.coordinates), 1)
        else:
            if np.max(conformations) < len(self.coordinates):
                frames = conformations
            else:
                raise Exception("ERROR: requested coordinate index %s, but only %s are available" %(np.max(conformations), len(self.coordinates)))

        idx = np.arange(len(idxs))

        # create molecule, and push created data information
        M = Molecule()
        postmp = self.coordinates[:, idxs]
        M.coordinates = postmp[frames]
        M.data = self.data.loc[idxs]
        M.data = M.data.reset_index(drop=True)
        M.data["index"] = idx
        M.current = 0
        M.points = M.coordinates[M.current]

        M.properties['center'] = M.get_center()

        return M

    def get_data(self, indices=[], columns=[]):
        '''
        Return information about atoms of interest (i.e., slice the data DataFrame)

        :param indices: list of indices, if not provided all atom data is returned
        :param columns: list of columns (e.g. ["resname", "resid", "chain"]), if not provided all columns are returned
        :returns: numpy array containing a slice of molecule's data
        '''

        if len(indices) == 0 and len(columns) == 0:
            return self.data.values

        elif len(indices) == 0 and len(columns) != 0:
            return self.data[columns].values

        elif len(indices) != 0 and len(columns) == 0:
            return self.data.loc[indices].values

        else:
            return self.data.loc[indices, columns].values   

