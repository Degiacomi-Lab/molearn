import os
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))

import torch

from molearn.data import PDBData_RNA
from molearn.trainers import Trainer_RNA


RNA_PDB = """\
MODEL        1
ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  P     U A   2       2.000   0.000   0.000  1.00  0.00           P
ATOM      4  C4'   U A   2       3.000   0.000   0.000  1.00  0.00           C
ATOM      5  P     A A   3       4.000   0.000   0.000  1.00  0.00           P
ATOM      6  C4'   A A   3       5.000   0.000   0.000  1.00  0.00           C
ATOM      7  P     C A   4       6.000   0.000   0.000  1.00  0.00           P
ATOM      8  C4'   C A   4       7.000   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  P     G A   1       0.100   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.100   0.000   0.000  1.00  0.00           C
ATOM      3  P     U A   2       2.100   0.000   0.000  1.00  0.00           P
ATOM      4  C4'   U A   2       3.100   0.000   0.000  1.00  0.00           C
ATOM      5  P     A A   3       4.100   0.000   0.000  1.00  0.00           P
ATOM      6  C4'   A A   3       5.100   0.000   0.000  1.00  0.00           C
ATOM      7  P     C A   4       6.100   0.000   0.000  1.00  0.00           P
ATOM      8  C4'   C A   4       7.100   0.000   0.000  1.00  0.00           C
ENDMDL
END
"""


class TestRNAData(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
        handle.write(RNA_PDB)
        handle.close()
        self.filename = handle.name

    def tearDown(self):
        os.unlink(self.filename)

    def test_prepare_and_set_data_RNA(self):
        data = PDBData_RNA()
        data.import_pdb_RNA(self.filename)
        data.atomselect_RNA(["P", "C4'"])
        dataset = data.prepare_dataset_RNA()

        self.assertEqual(dataset.shape, (2, 8, 3))
        self.assertEqual(set(data.indices), {"P", "C4'"})

        trainer = Trainer_RNA(device=torch.device("cpu"))
        trainer.set_data_RNA(data, batch_size=1, validation_split=0.5, manual_seed=1)
        self.assertEqual(set(trainer.indices_RNA), {"P", "C4'"})

    def test_combine_ensembles_RNA(self):
        data = PDBData_RNA()
        data.import_trajectories_RNA(
            filenames=[self.filename, self.filename],
            topologies=[None, None],
        )
        data.atomselect_RNA(["P", "C4'"])
        dataset = data.prepare_dataset_RNA()

        self.assertEqual(dataset.shape, (4, 8, 3))
        self.assertEqual(data.frames_per_trajectory_RNA, [2, 2])

    def test_full_dataset_can_be_used_for_training_and_validation(self):
        data = PDBData_RNA()
        data.import_pdb_RNA(self.filename)
        data.atomselect_RNA(["P", "C4'"])
        data.prepare_dataset_RNA()

        train_loader, valid_loader = data.get_dataloader_RNA(
            batch_size=1,
            manual_seed=1,
            full_dataset_validation_RNA=True,
        )

        self.assertEqual(len(train_loader.dataset), 2)
        self.assertEqual(len(valid_loader.dataset), 2)
        self.assertTrue(torch.equal(data.train_indices, torch.arange(2)))
        self.assertTrue(torch.equal(data.valid_indices, torch.arange(2)))

    def test_validation_split_is_applied_to_each_ensemble(self):
        data = PDBData_RNA()
        data.import_trajectories_RNA(
            filenames=[self.filename, self.filename],
            topologies=[None, None],
        )
        data.atomselect_RNA(["P", "C4'"])
        data.prepare_dataset_RNA()

        train_loader, valid_loader = data.get_dataloader_RNA(
            batch_size=1,
            validation_split=0.5,
            manual_seed=1,
            split_by_ensemble_RNA=True,
        )

        self.assertEqual(len(train_loader.dataset), 2)
        self.assertEqual(len(valid_loader.dataset), 2)
        for offset in (0, 2):
            self.assertEqual(
                int(((data.train_indices >= offset)
                     & (data.train_indices < offset + 2)).sum()),
                1,
            )
            self.assertEqual(
                int(((data.valid_indices >= offset)
                     & (data.valid_indices < offset + 2)).sum()),
                1,
            )

    def test_shared_kabsch_alignment_RNA(self):
        reference = np.asarray([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        rotation = np.asarray([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        moved = reference @ rotation + np.asarray([12.0, -4.0, 7.0])

        data = PDBData_RNA()
        aligned = data.align_coordinates_RNA(
            np.stack([reference, moved]), reference=reference
        )
        np.testing.assert_allclose(aligned[0], reference, atol=1e-10)
        np.testing.assert_allclose(aligned[1], reference, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
