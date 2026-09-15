"""Train the distance-matrix autoencoder (DistanceMatrix_AE) with DistanceMatrix_AE_Trainer.

The encoder reads an inter-atomic distance matrix and the decoder emits Cartesian
coordinates, so the model is invariant to rotation and translation by construction.
``fold_symmetric`` exploits the symmetry of the distance matrix to halve the encoder
input, which cuts its parameters and activations by roughly 4x.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
import torch

from molearn.data import PDBData
from molearn.models import DistanceMatrix_AE
from molearn.trainers import (DistanceMatrix_AE_Trainer,
                              DistanceMatrix_AE_Trainer_Config)


def main():
    ##### Load Data #####
    # No standardisation: the encoder sees a distance matrix, which is already invariant
    # to translation, so subtracting a mean achieves nothing and dividing by the standard
    # deviation only puts dm_loss in arbitrary units instead of Angstrom.
    data = PDBData(standardise=False)
    data.import_pdb(["./data/MurD_open.pdb", "./data/MurD_closed.pdb"])
    data.fix_terminal()
    data.atomselect(atoms=["N", "CA", "CB", "C", "O"])
    data.prepare_dataset()
    # A separate file from the foldingnet example: with standardise=False the statistics
    # are mean 0 / std 1, and writing them to the shared data_statistics.json would
    # overwrite the values analysis_example.ipynb needs for the foldingnet model.
    data.write_statistics("DMAE_data_statistics.json")

    n_atoms = data.dataset.shape[1]
    print(f"{len(data.dataset)} frames, {n_atoms} atoms")

    ##### Prepare Trainer #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dm_loss dominates; the dihedral term supplies chirality, which a distance matrix
    # alone cannot express. Physics is off here to keep the example quick -- set
    # physics_weight and call trainer.prepare_physics() to enable it.
    config = DistanceMatrix_AE_Trainer_Config(
        dm_weight=1.0,
        local_k=4,
        local_weight=0.9,
        nonlocal_weight=0.1,
        dihed_weight=1.0,
        dihed_bb_weight=1.0,
        dihed_imp_weight=1.0,
        physics_weight=0.0,
        physics_inter_weight=0.0,
    )

    trainer = DistanceMatrix_AE_Trainer(dm_dim=n_atoms, device=device, config=config)
    trainer.set_data(
        data,
        batch_size=4,
        validation_split=0.1,
        manual_seed=25,
        save_indices=False,
    )

    trainer.set_autoencoder(DistanceMatrix_AE, dm_dim=n_atoms, latent_dim=2,
                            fold_symmetric=True)
    n_params = sum(p.numel() for p in trainer.autoencoder.parameters())
    print(f"autoencoder: {n_params:,} parameters")
    trainer.prepare_optimiser()

    ##### Training Loop #####
    # 3 epochs so the example runs in about a minute. For a real model use
    # trainer.run_until_converge(patience=16, ...) as in train_foldingnet_withCB.py.
    fit_results = trainer.run(
        epochs=3,
        log_filename="log.dat",
        log_folder="DMAE_checkpoints",
        checkpoint_folder="DMAE_checkpoints",
        verbose=True,
    )
    print(fit_results)


if __name__ == "__main__":
    main()
