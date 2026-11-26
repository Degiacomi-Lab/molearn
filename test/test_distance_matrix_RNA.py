import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))

from molearn.models import DistanceMatrix_AE, DistanceMatrix_AE_RNA
from molearn.trainers import (
    DistanceMatrix_AE_Trainer_Config_RNA,
    DistanceMatrix_AE_Trainer_RNA,
)
from molearn.trainers.distance_matrix_ae_trainer import DistanceMatrix_AE_Trainer


class TestDistanceMatrixRNA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.residues_RNA = 4
        self.coordinates_RNA = torch.randn(
            2, self.residues_RNA * 5, 3
        )
        self.model_RNA = DistanceMatrix_AE_RNA(
            residue_count_RNA=self.residues_RNA,
            latent_dimension=3,
            init_c=8,
            m=2,
            min_size=2,
        )

    def test_full_matrix_and_lossless_triangular_pack_RNA(self):
        matrix_RNA = self.model_RNA.coords_to_distance_matrix_RNA(
            self.coordinates_RNA
        )
        self.assertEqual(
            tuple(matrix_RNA.shape),
            (2, 1, self.residues_RNA * 5, self.residues_RNA * 5),
        )
        expected_RNA = DistanceMatrix_AE.coords_to_dm(
            self.coordinates_RNA
        )
        torch.testing.assert_close(
            matrix_RNA, expected_RNA,
        )
        folded_RNA = self.model_RNA.fold_dm(matrix_RNA)
        restored_RNA = self.model_RNA.unfold_dm(
            folded_RNA, self.residues_RNA * 5
        )
        diagonal_RNA = torch.eye(
            self.residues_RNA * 5, dtype=torch.bool
        )[None, None]
        torch.testing.assert_close(
            restored_RNA.masked_fill(diagonal_RNA, 0.0),
            matrix_RNA.masked_fill(diagonal_RNA, 0.0),
        )

    def test_maps_are_translation_invariant_RNA(self):
        shift_RNA = torch.tensor([17.0, -4.0, 9.0])
        original_RNA = self.model_RNA.coords_to_distance_matrix_RNA(
            self.coordinates_RNA
        )
        translated_RNA = self.model_RNA.coords_to_distance_matrix_RNA(
            self.coordinates_RNA + shift_RNA
        )
        torch.testing.assert_close(
            translated_RNA, original_RNA, atol=1e-4, rtol=2e-4
        )

    def test_model_and_trainer_backward_RNA(self):
        output_RNA = self.model_RNA(self.coordinates_RNA)
        self.assertEqual(
            tuple(output_RNA.shape),
            tuple(self.coordinates_RNA.shape),
        )
        config_RNA = DistanceMatrix_AE_Trainer_Config_RNA(
            dm_weight=1.0,
            local_k=4,
            local_weight_RNA=0.9,
            nonlocal_weight_RNA=0.1,
        )
        trainer_RNA = DistanceMatrix_AE_Trainer_RNA(
            residue_count_RNA=self.residues_RNA,
            device=torch.device("cpu"),
            config_RNA=config_RNA,
        )
        trainer_RNA.set_autoencoder(
            DistanceMatrix_AE_RNA,
            residue_count_RNA=self.residues_RNA,
            sites_per_residue_RNA=5,
            latent_dimension=3,
            init_c=8,
            m=2,
            min_size=2,
        )
        weights_RNA = trainer_RNA.distance_weights_RNA[0, 0]
        self.assertTrue(torch.equal(weights_RNA, weights_RNA.T))
        self.assertTrue(torch.all(torch.diagonal(weights_RNA) == 0.9))
        first_matrix_RNA = self.model_RNA.coords_to_dm(self.coordinates_RNA)
        second_matrix_RNA = first_matrix_RNA + 0.25
        torch.testing.assert_close(
            trainer_RNA._get_dm_loss_RNA(
                first_matrix_RNA, second_matrix_RNA
            ),
            DistanceMatrix_AE_Trainer._get_dm_loss(
                trainer_RNA, first_matrix_RNA, second_matrix_RNA
            ),
        )
        result_RNA = trainer_RNA.common_step_RNA(self.coordinates_RNA)
        self.assertEqual(
            set(result_RNA),
            {
                "dm_loss",
                "dm_rmse_A",
                "openmm_simrna_energy",
                "simrna_sp_bond_energy",
                "simrna_sp_angle_energy",
                "simrna_sp_torsion_energy",
                "simrna_sp_energy",
                "loss",
            },
        )
        expected_rmse_RNA = (
            result_RNA["dm_loss"] / trainer_RNA.W.sum().sqrt()
        )
        torch.testing.assert_close(
            result_RNA["dm_rmse_A"], expected_rmse_RNA
        )
        self.assertTrue(torch.isfinite(result_RNA["loss"]))
        result_RNA["loss"].backward()
        gradients_RNA = [
            parameter_RNA.grad
            for parameter_RNA in trainer_RNA.autoencoder.parameters()
            if parameter_RNA.grad is not None
        ]
        self.assertTrue(gradients_RNA)
        self.assertTrue(
            all(torch.isfinite(gradient_RNA).all() for gradient_RNA in gradients_RNA)
        )


if __name__ == "__main__":
    unittest.main()
