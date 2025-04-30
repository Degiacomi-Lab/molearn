import numpy as np

def get_ca_chirality(n, ca, c, cb):
    """
    Compute the chirality of Cα atoms given vectors of coordinates of N, Cα, C, and Cβ atoms.

    :param numpy.array n: Cartesian coordinates of N atom
    :param numpy.array ca: Cartesian coordinates of CA atom
    :param numpy.array c: Cartesian coordinates of C atom
    :param numpy.array cb: Cartesian coordinates of CB atom

    :return: Angles between the normal vector and the vector from Cα to Cβ. Positive for L-aa.
    """
    ca_n = n - ca
    ca_c = c - ca
    cb_ca = cb - ca
    normal = np.cross(ca_n, ca_c)
    dot = np.einsum("ij,ij->i", normal, cb_ca)

    return dot