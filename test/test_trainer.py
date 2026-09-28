import unittest
import sys
import os
import tempfile
import shutil
import torch

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))
from molearn.trainers import FitResult, Trainer

class DummyData:
    def __init__(self, n_frames=16, n_atoms=5):
        self.dataset = torch.randn(n_frames, n_atoms, 3)
        self.std = torch.tensor(1.0)
        self.mean = torch.tensor(0.0)
        self.standardise = True
        self.mol = object()
        self.atoms = ["N", "CA", "C", "O", "CB"]
        self.indices = {
            "N": torch.tensor([0], dtype=torch.long),
            "CA": torch.tensor([1], dtype=torch.long),
            "C": torch.tensor([2], dtype=torch.long),
            "O": torch.tensor([3], dtype=torch.long),
            "CB": torch.tensor([4], dtype=torch.long),
        }

    def get_dataloader(self, batch_size=4, validation_split=0.25, **kwargs):
        split = int(len(self.dataset) * (1 - validation_split))
        split = max(1, min(split, len(self.dataset) - 1))
        train_data = self.dataset[:split]
        valid_data = self.dataset[split:]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valid_data),
            batch_size=batch_size,
            shuffle=False,
        )
        return train_loader, valid_loader


class TinyAutoEncoder(torch.nn.Module):
    def __init__(self, n_atoms=5, latent_dim=2):
        super().__init__()
        self.n_atoms = n_atoms
        in_dim = n_atoms * 3
        self.encoder = torch.nn.Linear(in_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, in_dim)

    def encode(self, batch):
        return self.encoder(batch.reshape(batch.shape[0], -1))

    def decode(self, latent):
        decoded = self.decoder(latent)
        return decoded.reshape(latent.shape[0], self.n_atoms, 3)


def _checkpoint_payload(model, epoch, loss):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "loss": loss,
        "network_kwargs": {},
        "atoms": ["N", "CA", "C", "O", "CB"],
        "std": 1.0,
        "mean": 0.0,
    }


class TestTrainers(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.data = DummyData(n_frames=20, n_atoms=5)
        self.tmp_dir = tempfile.mkdtemp(prefix="molearn_trainer_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_init_defaults(self):
        trainer = Trainer(device=self.device)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.device.type, "cpu")

    def test_run_returns_fit_result_and_writes_checkpoints(self):
        trainer = Trainer(device=self.device)
        trainer.set_autoencoder(TinyAutoEncoder, n_atoms=5, latent_dim=2)
        trainer.set_data(self.data, batch_size=4)
        trainer.prepare_optimiser(lr=1e-3)

        log_dir = os.path.join(self.tmp_dir, "logs")
        ckpt_dir = os.path.join(self.tmp_dir, "ckpts")
        result = trainer.run(
            epochs=2,
            log_folder=log_dir,
            checkpoint_folder=ckpt_dir,
            verbose=False,
        )

        self.assertIsInstance(result, FitResult)
        self.assertEqual(result.epochs_run, 2)
        self.assertTrue(os.path.exists(result.last_checkpoint))
        self.assertTrue(os.path.exists(result.best_checkpoint))
        self.assertTrue(os.path.exists(result.log_file))

    def test_load_checkpoint_best_selects_lowest_loss_filename(self):
        trainer = Trainer(device=self.device)
        trainer.set_autoencoder(TinyAutoEncoder, n_atoms=5, latent_dim=2)

        ckpt_dir = os.path.join(self.tmp_dir, "manual_ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            _checkpoint_payload(trainer.autoencoder, epoch=3, loss=2.0),
            os.path.join(ckpt_dir, "checkpoint_epoch3_loss2.0.ckpt"),
        )
        torch.save(
            _checkpoint_payload(trainer.autoencoder, epoch=7, loss=1e-03),
            os.path.join(ckpt_dir, "checkpoint_epoch7_loss1e-03.ckpt"),
        )

        trainer.load_checkpoint(
            checkpoint_name="best",
            checkpoint_folder=ckpt_dir,
            load_optimiser=False,
            weights_only=True,
        )

        self.assertEqual(trainer.epoch, 8)

    def test_update_optimiser_hyperparameters(self):
        trainer = Trainer(device=self.device)
        trainer.set_autoencoder(TinyAutoEncoder, n_atoms=5, latent_dim=2)
        trainer.prepare_optimiser(lr=1e-3)
        trainer.update_optimiser_hyperparameters(lr=5e-4)
        self.assertAlmostEqual(trainer.optimiser.param_groups[0]["lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
