import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from molearn.loss_functions.simrna_sp_RNA import SimRNALocalStatisticalPotential_RNA
from molearn.trainers.trainer_RNA import Trainer_RNA


class _IdentityAutoencoder_RNA(torch.nn.Module):
    def encode(self, coordinates_RNA):
        return coordinates_RNA

    def decode(self, latent_RNA):
        return latent_RNA


def _fake_data_RNA():
    atoms_RNA = []
    names_RNA = ("P", "C4'", "N9", "C2", "C6")
    for residue_RNA in range(1, 5):
        atoms_RNA.extend(
            SimpleNamespace(
                name=name_RNA,
                resid=residue_RNA,
                segid="A",
                chainID="A",
            )
            for name_RNA in names_RNA
        )
    return SimpleNamespace(
        std=2.0,
        mean=-1.0,
        _mol=SimpleNamespace(atoms=atoms_RNA),
    )


def _coordinates_RNA():
    residues_RNA = []
    for index_RNA in range(4):
        offset_RNA = torch.tensor([3.7 * index_RNA, 0.3 * index_RNA, 0.0])
        residues_RNA.extend(
            (
                offset_RNA + torch.tensor([0.0, 0.0, 0.0]),
                offset_RNA + torch.tensor([1.8, 3.2, 0.5]),
                offset_RNA + torch.tensor([2.7, 4.0, 1.2]),
                offset_RNA + torch.tensor([3.2, 4.8, 1.6]),
                offset_RNA + torch.tensor([2.4, 5.3, 2.2]),
            )
        )
    physical_RNA = torch.stack(residues_RNA).unsqueeze(0)
    return (physical_RNA + 1.0) / 2.0


class TestSimRNALocalStatisticalPotentialRNA(unittest.TestCase):
    def test_tables_are_bundled_and_energy_has_gradients(self):
        potential_RNA = SimRNALocalStatisticalPotential_RNA(_fake_data_RNA())
        self.assertEqual(potential_RNA.dist_CP_RNA.values_RNA.numel(), 15)
        self.assertEqual(potential_RNA.dist_PC_RNA.values_RNA.numel(), 13)
        self.assertEqual(tuple(potential_RNA.eta_theta_RNA.values_RNA.shape), (50, 50))

        coordinates_RNA = _coordinates_RNA().requires_grad_(True)
        components_RNA = potential_RNA.components_RNA(coordinates_RNA)
        self.assertEqual(
            set(components_RNA),
            {
                "simrna_sp_bond_energy",
                "simrna_sp_angle_energy",
                "simrna_sp_torsion_energy",
                "simrna_sp_energy",
            },
        )
        expected_RNA = (
            components_RNA["simrna_sp_bond_energy"]
            + components_RNA["simrna_sp_angle_energy"]
            + components_RNA["simrna_sp_torsion_energy"]
        )
        torch.testing.assert_close(components_RNA["simrna_sp_energy"], expected_RNA)
        components_RNA["simrna_sp_energy"].mean().backward()
        self.assertTrue(torch.isfinite(coordinates_RNA.grad).all())
        self.assertGreater(torch.count_nonzero(coordinates_RNA.grad).item(), 0)

    def test_trainer_combines_mse_and_statistical_energy(self):
        data_RNA = _fake_data_RNA()
        trainer_RNA = Trainer_RNA(device=torch.device("cpu"))
        trainer_RNA._data = data_RNA
        trainer_RNA.autoencoder = _IdentityAutoencoder_RNA()
        trainer_RNA.prepare_simrna_sp_RNA(weight_RNA=0.25)

        result_RNA = trainer_RNA.common_step_RNA(_coordinates_RNA())
        expected_RNA = (
            result_RNA["mse_loss"]
            + 0.25 * result_RNA["simrna_sp_energy"]
        )
        torch.testing.assert_close(result_RNA["loss"], expected_RNA)

    def test_zero_weight_disables_the_potential(self):
        trainer_RNA = Trainer_RNA(device=torch.device("cpu"))
        trainer_RNA.prepare_simrna_sp_RNA(weight_RNA=0.0)
        self.assertIsNone(trainer_RNA.simrna_sp_loss_RNA)

    def test_epoch_report_is_printed_on_one_line(self):
        trainer_RNA = Trainer_RNA(device=torch.device("cpu"))
        trainer_RNA._display_patience_RNA = 16
        trainer_RNA.progress.epochs_since_improve = 3
        metrics_RNA = {
            "epoch": 7,
            "train_mse_loss": 0.07,
            "train_simrna_sp_energy": -4.1,
            "train_loss": 0.029,
            "valid_mse_loss": 0.08,
            "valid_simrna_sp_energy": -4.0,
            "valid_loss": 0.04,
            "total_seconds": 12.34,
        }
        with TemporaryDirectory() as directory_RNA:
            log_RNA = Path(directory_RNA) / "log.dat"
            output_RNA = StringIO()
            with redirect_stdout(output_RNA):
                trainer_RNA.log(str(log_RNA), metrics_RNA)
            lines_RNA = output_RNA.getvalue().splitlines()
            self.assertEqual(len(lines_RNA), 1)
            self.assertIn("train_SimRNA_local: -4.100000", lines_RNA[0])
            self.assertIn("valid_SimRNA_local: -4.000000", lines_RNA[0])
            self.assertIn("no_improve: 3/16", lines_RNA[0])
            self.assertTrue(log_RNA.is_file())


if __name__ == "__main__":
    unittest.main()
