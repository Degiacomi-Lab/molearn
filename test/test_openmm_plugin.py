import unittest
import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), 'src'))
import molearn

import torch

class Test_PDBData_Basics(unittest.TestCase):
    def setUp(self):
        self.data = molearn.PDBData()
        self.data.import_pdb('MurD_test.pdb')
        self.data.prepare_dataset()

    def test_dataset_is_tensor(self,):
        self.assertIsInstance(self.data.dataset, torch.Tensor)

    def test_dataset_dimensions(self,):
        self.assertTrue(self.data.dataset.shape[0]==16)
        self.assertTrue(self.data.dataset.shape[1]==3)
        self.assertTrue(self.data.dataset.shape[2]==3286)

    def test_mean(self):
        self.assertIsInstance(self.data.mean, float)
        self.assertTrue(torch.isclose(torch.zeros(1), self.data.dataset.mean()))

    def test_std(self):
        self.assertIsInstance(self.data.std, float)
        self.assertTrue(torch.isclose(torch.ones(1), self.data.dataset.std()))

    def test_atominfo(self):
        atominfo = self.data.get_atominfo()
        self.assertTrue(atominfo.shape[0] == self.data.dataset.shape[2])
        self.assertTrue(atominfo.shape[1]==3)
        # Are the types correct (str, str, int)
        self.assertTrue(all([all([isinstance(a, str), isinstance(b, str), isinstance(c, int)]) for a,b,c in atominfo]))

    def test_frame(self):
        frame = self.data.frame()
        self.assertTrue(frame.coordinates.shape[0] == 1)
        self.assertTrue(frame.coordinates.shape[1] == self.data.dataset.shape[2])
        self.assertTrue(frame.coordinates.shape[2] == 3)

    def test_get_dataloader(self):
        bs = 4
        train, valid = self.data.get_dataloader(batch_size=bs, validation_split=0.5)
        self.assertIsInstance(train, torch.utils.data.DataLoader)
        self.assertIsInstance(valid, torch.utils.data.DataLoader)
        train_batch = next(iter(train))
        valid_batch = next(iter(valid))

        self.assertTrue(train_batch[0].shape[0]==bs)
        self.assertTrue(valid_batch[0].shape[0]==bs)
        self.assertTrue(train_batch[0].shape[1]==3)
        self.assertTrue(valid_batch[0].shape[1]==3)
        self.assertTrue(train_batch[0].shape[2]==self.data.dataset.shape[2])
        self.assertTrue(valid_batch[0].shape[2]==self.data.dataset.shape[2])

    def test_split(self):
        train, valid = self.data.split(validation_split = 0.5)
        self.assertIsInstance(train, molearn.PDBData)
        self.assertIsInstance(valid, molearn.PDBData)
        self.assertTrue(train.dataset.shape[0]==int(self.data.dataset.shape[0]/2))

    def test_get_datasets(self):
        train, valid = self.data.get_datasets(validation_split = 0.5)
        cdist = torch.cdist(self.data.dataset.reshape(16,-1).unsqueeze(0), train.reshape(8,-1).unsqueeze(0))[0]

        mask = torch.isclose(cdist, torch.zeros_like(cdist))
        axis_0 = mask.sum(axis=0)
        self.assertTrue(torch.allclose(axis_0, torch.ones_like(axis_0)))
        axis_1 = mask.sum(axis=1)
        self.assertTrue(axis_1.max()==1)
        self.assertTrue(axis_1.sum() == 8)

    def test_atoms(self):
        atoms = self.data.atoms
        self.assertTrue(len(atoms)==37)
        self.assertTrue(all([isinstance(atom, str) for atom in atoms]))

class Test_PDBData_atomselect_bb(Test_PDBData_Basics):
    def setUp(self):
        self.data = molearn.PDBData()
        self.data.import_pdb('MurD_test.pdb')
        self.data.atomselect(atoms = ['CA', 'C', 'CB', 'O', 'N'])
        self.data.prepare_dataset()

    def test_dataset_dimensions(self,):
        self.assertTrue(self.data.dataset.shape[0]==16)
        self.assertTrue(self.data.dataset.shape[1]==3)
        self.assertTrue(self.data.dataset.shape[2]==2145)

    def test_atoms(self):
        atoms = self.data.atoms
        self.assertTrue(len(atoms)==5)
        self.assertTrue(all([isinstance(atom, str) for atom in atoms]))

class Test_PDBData_atomselect_no_hydrogen(Test_PDBData_Basics):
    def setUp(self):
        self.data = molearn.PDBData()
        self.data.import_pdb('MurD_test.pdb')
        self.data.atomselect(atoms = 'no_hydrogen')
        self.data.prepare_dataset()

    def test_dataset_dimensions(self,):
        self.assertTrue(self.data.dataset.shape[0]==16)
        self.assertTrue(self.data.dataset.shape[1]==3)
        self.assertTrue(self.data.dataset.shape[2]==3286)

    def test_atoms(self):
        atoms = self.data.atoms
        self.assertTrue(len(atoms)==37)
        self.assertTrue(all([isinstance(atom, str) for atom in atoms]))


class Test_Trainers(unittest.TestCase):
    def setUp(self):
        self.data = molearn.PDBData()
        self.data.import_pdb('MurD_test.pdb')
        self.data.atomselect(atoms = ['CA', 'C', 'CB', 'O', 'N'])
        self.data.prepare_dataset()
    def test_init(self):
        pass

class Test_openmm_plugin(unittest.TestCase):
    def test_openmm_energy(self,):

        class mymodule(torch.nn.Module):
            def __init__(self, coords, **kwargs):
                super().__init__()
                self.para = torch.nn.Parameter(coords)
            def forward(self, x):
                return x + self.para

        data = molearn.PDBData()
        data.import_pdb('MurD_test.pdb')
        #data.atomselect(atoms='no_hydrogen')
        data.atomselect(atoms=['CA', 'C', 'CB', 'N', 'O'])
        data.prepare_dataset()
        #dataloader = data.get_dataloader(batch_size=4)
        device = torch.device('cpu')
        _d = data.dataset.to(device).float()
        #d = next(iter(dataloader))[0].to(device).float()
        para = mymodule(_d.clone())
        from molearn.loss_functions import openmm_energy
        openmmscore = openmm_energy(data.mol, data.std, clamp=None, platform = 'CUDA') #xml_file = ['modified_amber_protein.xml',])
        opt = torch.optim.SGD(para.parameters(), lr=0.0001)
        scores = []
        for i in range(1000):
            opt.zero_grad()
            d = torch.zeros_like(_d)
            x = para(d)
            energy = openmmscore(x)
            loss = 1e-4*torch.nansum(energy)
            loss.backward()
            opt.step()
            self.assertFalse(torch.isnan(energy).any())
            if i > 20:
                ratio = (torch.tensor(scores)>loss.item()).sum()/i
                self.assertGreater(ratio, 0.9)
                if i%100==0:
                    print(ratio)
                    print(loss)
            scores.append(loss.item())


if __name__ =='__main__':
    unittest.main()
