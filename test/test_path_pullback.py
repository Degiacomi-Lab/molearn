import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))
from molearn.analysis.path import oversample, pullback_oversample

import numpy as np
import torch


class NonlinearDecoderNet(torch.nn.Module):
    def __init__(self, latent_dim=2, n_atoms=8, hidden=16):
        super().__init__()
        self.n_atoms = n_atoms
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_atoms * 3),
        )
        with torch.no_grad():
            self.net[0].weight.mul_(3.0)
            self.net[2].weight.mul_(2.0)

    def decode(self, z):
        return self.net(z).view(-1, self.n_atoms, 3)


def _consecutive_drmsd_gaps(network, samples, n_atoms):
    with torch.no_grad():
        z = torch.as_tensor(samples, dtype=torch.float32)
        coords = network.decode(z)[:, :n_atoms, :]
    iu = torch.triu_indices(n_atoms, n_atoms, offset=1)
    dm = torch.norm(coords[:, iu[0]] - coords[:, iu[1]], dim=-1)
    diffs = dm[1:] - dm[:-1]
    return torch.sqrt((diffs ** 2).mean(dim=-1)).numpy()


class Test_PullbackOversample(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n_atoms = 8
        self.latent_dim = 2
        self.network = NonlinearDecoderNet(self.latent_dim, self.n_atoms)
        self.pts = 10
        self.n_quad = 32

    def test_short_segment_matches_oversample(self):
        crd = np.array([[0.0, 0.0], [1e-3, 5e-4]])
        euclid = oversample(crd, pts=self.pts)
        pullback = pullback_oversample(crd, self.pts,
                                       network=self.network,
                                       n_atoms=self.n_atoms,
                                       n_quad=self.n_quad)
        self.assertEqual(euclid.shape, pullback.shape)
        self.assertTrue(np.allclose(euclid, pullback, atol=1e-5))

    def test_long_segment_diverges_from_oversample(self):
        crd = np.array([[-2.0, 1.0], [2.5, -1.5]])
        euclid = oversample(crd, pts=self.pts)
        pullback = pullback_oversample(crd, self.pts,
                                       network=self.network,
                                       n_atoms=self.n_atoms,
                                       n_quad=self.n_quad)
        self.assertEqual(euclid.shape, pullback.shape)
        self.assertFalse(np.allclose(euclid, pullback, atol=1e-2))

    def test_consecutive_decoded_gaps_more_uniform_than_euclidean(self):
        crd = np.array([[-2.0, 1.0], [2.5, -1.5]])
        euclid = oversample(crd, pts=self.pts)
        pullback = pullback_oversample(crd, self.pts,
                                       network=self.network,
                                       n_atoms=self.n_atoms,
                                       n_quad=self.n_quad)
        gaps_e = _consecutive_drmsd_gaps(self.network, euclid, self.n_atoms)
        gaps_p = _consecutive_drmsd_gaps(self.network, pullback, self.n_atoms)
        cv_euclid = gaps_e.std() / gaps_e.mean()
        cv_pullback = gaps_p.std() / gaps_p.mean()
        self.assertLess(cv_pullback, 0.15)
        self.assertGreater(cv_euclid, 0.30)


if __name__ == "__main__":
    unittest.main()
