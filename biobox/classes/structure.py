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

from copy import deepcopy
import numpy as np
import pandas as pd

class Structure(object):
    '''
    A Structure consists of an ensemble of points in 3D space, and metadata associated to each of them.
    '''

    def __init__(self, p=np.array([[], []]), r=1.0):
        '''
        Point coordinates and properties data structures are first initialized.
        properties is a dictionary initially containing an entry for 'center' (center of geometry) and 'radius' (average radius of points).

        :param p: coordinates data structure as a mxnx3 numpy array (alternative conformation x atom x 3D coordinate). nx3 numpy array can be supplied, in case a single conformation is present.
        :param r: average radius of every point in dataset (float), or radius of every point (numpy array)
        '''
        if p.ndim == 3:
            self.coordinates = p
            '''numpy array containing an ensemble of alternative coordinates in 3D space'''

        elif p.ndim == 2:
            self.coordinates = np.array([p])
        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %p.ndim)

        self.current = 0
        '''index of currently selected conformation'''

        self.points = self.coordinates[self.current]
        '''pointer to currently selected conformation'''

        self.properties = {}
        '''collection of properties. By default, 'center' (geometric center of the Structure) is defined'''

        self.properties['center'] = self.get_center()
   
        idx = np.arange(len(self.points))     
        if isinstance(r, list) or type(r).__module__ == 'numpy':
            if len(r) > 0:
                self.data = pd.DataFrame(r, index=idx, columns=["radius"])
        else:
                rad = r*np.ones(len(self.points))           
                self.data = pd.DataFrame(rad, index=idx, columns=["radius"])
                ''' metadata about each atom (pandas Dataframe)'''

    def __len__(self, dim="atoms"):
        if dim == "atoms":
            return len(self.points)

    def __getitem__(self, key):
        return self.coordinates[key]

    def set_current(self, pos):
        '''
        select current frame (place frame pointer at desired position)

        :param pos: number of alternative conformation (starting from 0)
        '''
        if pos < self.coordinates.shape[0]:
            self.current = pos
            self.points = self.coordinates[self.current]
            self.properties['center'] = self.get_center()
        else:
            raise Exception("ERROR: position %s requested, but only %s conformations available" %(pos, self.coordinates.shape[0]))

    def get_xyz(self, indices=[]):
        '''
        get points coordinates.

        :param indices: indices of points to select. If none is provided, all points coordinates are returned.
        :returns: coordinates of all points indexed by the provided indices list, or all of them if no list is provided.
        '''
        if indices == []:
            return self.points
        else:
            return self.points[indices]

    def set_xyz(self, coords):
        '''
        set point coordinates.

        :param coords: array of 3D points
        '''
        self.coordinates[self.current] = deepcopy(coords)
        self.points = self.coordinates[self.current]

    def add_xyz(self, coords):
        '''
        add a new alternative conformation to the database

        :param coords: array of 3D points, or array of arrays of 3D points (in case multiple alternative coordinates must be added at the same time)
        '''
        # self.coordinates numpy array containing an ensemble of alternative
        # coordinates in 3D space

        if self.coordinates.size == 0 and coords.ndim == 3:
            self.coordinates = deepcopy(coords)
            self.set_current(0)

        elif self.coordinates.size == 0 and coords.ndim == 2:
            self.coordinates = deepcopy(np.array([coords]))
            self.set_current(0)

        elif self.coordinates.size > 0 and coords.ndim == 3:
            self.coordinates = np.concatenate((self.coordinates, coords))
            # set new frame to the first of the newly inserted ones
            self.set_current(self.current + 1)

        elif self.coordinates.size > 0 and coords.ndim == 2:
            self.coordinates = np.concatenate((self.coordinates, np.array([coords])))
            # set new frame to the first of the newly inserted ones
            self.set_current(self.current + 1)

        else:
            raise Exception("ERROR: expected numpy array with 2 or three dimensions, but %s dimensions were found" %np.ndim)

    def delete_xyz(self, index):
        '''
        remove one conformation from the conformations database.

        the new current conformation will be the previous one.

        :param index: alternative coordinates set to remove
        '''
        self.coordinates = np.delete(self.coordinates, index, axis=0)
        if index > 0:
            self.set_current(index - 1)
        else:
            self.set_current(0)

    def clear(self):
        '''
        remove all the coordinates and empty metadata
        '''
        self.coordinates = np.array([[[], []], [[], []]])
        self.points = self.coordinates[0]
        self.data = pd.DataFrame(index=[], columns=[])

    def rmsd(self, i, j, points_index=[], full=False):
        '''
        Calculate the RMSD between two structures in alternative coordinates ensemble.
        uses Kabsch alignement algorithm.

        :param i: index of the first structure
        :param j: index of the second structure
        :param points_index: if set, only specific points will be considered for comparison
        :param full: if True, RMSD an rotation matrx are returned, RMSD only otherwise
        :returns: RMSD of the two structures. If full is True, the rotation matrix is also returned
        '''

        # see: http://www.pymolwiki.org/index.php/Kabsch#The_Code

        if i >= len(self.coordinates):
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(i, len(self.coordinates)))

        if j >= len(self.coordinates):
            raise Exception("ERROR: index %s requested, but only %s exist in database" %(j, len(self.coordinates)))

        # get first structure and center it
        if len(points_index) == 0:
            m1 = deepcopy(self.coordinates[i])
        elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
            m1 = deepcopy(self.coordinates[i, points_index])
        else:
            raise Exception("ERROR: give me a list of indices to compute RMSD, or nothing at all, please!")

        # get second structure
        if len(points_index) == 0:
            m2 = deepcopy(self.coordinates[j])
        elif isinstance(points_index, list) or type(points_index).__module__ == 'numpy':
            m2 = deepcopy(self.coordinates[j, points_index])
        else:
            raise Exception("ERROR: give me a list of indices to compute RMSD, or nothing at all, please!")

        L = len(m1)
        COM1 = np.sum(m1, axis=0) / float(L)
        m1 -= COM1
        m1sum = np.sum(np.sum(m1 * m1, axis=0), axis=0)

        COM2 = np.sum(m2, axis=0) / float(L)
        m2 -= COM2

        E0 = m1sum + np.sum(np.sum(m2 * m2, axis=0), axis=0)

        # This beautiful step provides the answer. V and Wt are the orthonormal
        # bases that when multiplied by each other give us the rotation matrix, U.
        # S, (Sigma, from SVD) provides us with the error!  Isn't SVD great!
        V, S, Wt = np.linalg.svd(np.dot(np.transpose(m2), m1))

        reflect = float(str(float(np.linalg.det(V) * np.linalg.det(Wt))))

        if reflect == -1.0:
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]

        rmsdval = E0 - (2.0 * sum(S))
        if full:
            return np.sqrt(abs(rmsdval / L)), np.matmul(V, Wt)
        else:
            return np.sqrt(abs(rmsdval / L))           

    def rmsd_distance_matrix(self, points_index=[], flat=False):
        '''
        compute distance matrix between structures (using RMSD as metric).

        :param points_index: if set, only specific points will be considered for comparison
        :param flat: if True, returns flattened distance matrix
        :returns: RMSD distance matrix
        '''

        if flat:
            rmsd = []
        else:
            rmsd = np.zeros((len(self.coordinates), len(self.coordinates)))

        for i in range(0, len(self.coordinates) - 1, 1):
            for j in range(i + 1, len(self.coordinates), 1):
                r = self.rmsd(i, j, points_index)

                if flat:
                    rmsd.append(r)
                else:
                    rmsd[i, j] = r
                    rmsd[j, i] = r

        if flat:
            return np.array(rmsd)
        else:
            return rmsd

    def get_center(self):
        '''
        compute protein center of geometry (also assigns it to self.properties["center"] key).
        '''
        if len(self.points) > 0:
            self.properties['center'] = np.mean(self.points, axis=0)
        else:
            self.properties['center'] = np.array([0.0, 0.0, 0.0])

        return self.properties['center']

