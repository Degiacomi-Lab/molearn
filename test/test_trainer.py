import unittest
import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), 'src'))
import molearn

import torch



class Test_Trainers(unittest.TestCase):
    def setUp(self):
        self.data = molearn.PDBData()
        self.data.import_pdb('MurD_test.pdb')
        self.data.atomselect(atoms = ['CA', 'C', 'CB', 'O', 'N'])
        self.data.prepare_dataset()
    def test_init(self):
        pass


if __name__ =='__main__':
    unittest.main()
