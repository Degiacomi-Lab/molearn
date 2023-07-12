import unittest
import sys
import os
import glob
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), 'src'))
import molearn

import torch


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
