import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.trainers import Transformer_Trainer
from molearn.models.transformer import TransformerCoordGen
import torch


def main():
    ##### Load Data #####
    data = PDBData()
    data.import_pdb(
        "./clustered/MurDopen_CLUSTER_aggl_train.dcd",
        "./clustered/MurDopen_NEW_TOPO.pdb",
    )
    data.fix_terminal()
    data.atomselect(atoms=["CA", "C", "N", "CB", "O"])

    ##### Prepare Trainer #####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Transformer_Trainer(device=device)

    trainer.set_data(data, batch_size=8, validation_split=0.1, manual_seed=25)

    trainer.set_autoencoder(TransformerCoordGen)
    trainer.prepare_optimiser()

    ##### Training Loop #####
    # Keep training until loss does not improve for 32 consecutive epochs

    runkwargs = dict(
        log_filename="log_file.dat",
        log_folder="transformer_checkpoints",
        checkpoint_folder="transformer_checkpoints",
    )

    best = 1e24
    while True:
        trainer.run(max_epochs=32 + trainer.epoch, **runkwargs)
        if not best > trainer.best:
            break
        best = trainer.best
    print(f"best {trainer.best}, best_filename {trainer.best_name}")


if __name__ == "__main__":
    main()
