import sys
import os
import argparse

import torch

from torch.optim import Adam
import numpy as np

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import PDBData
from molearn.models.transformer import (
    TransformerCoordGen,
    generate_coordinates,
    gen_xyz,
)

torch.manual_seed(42)
np.random.seed(42)
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def qr_full(num_samples=1):
    z = np.random.randn(num_samples, 3, 3)
    q, r = np.linalg.qr(z)
    sign = 2 * (np.diagonal(r, axis1=-2, axis2=-1) >= 0) - 1
    rot = q
    rot *= sign[..., None, :]
    rot[:, 0, :] *= np.linalg.det(rot)[..., None]
    return rot


# Initialize and train the model
model = TransformerCoordGen().to(device)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-tr", "--trajectory", type=str, required=True, help="path to trajectory"
)
parser.add_argument(
    "-to", "--topology", type=str, required=True, help="path to topology"
)
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    required=True,
    help="path to directory where things should get stored",
)

args = parser.parse_args()

if not os.path.isdir(args.outpath):
    os.mkdir(args.outpath)
##### Load Data #####

data = PDBData()
data.import_pdb(args.trajectory, args.topology)
data.fix_terminal()
data.atomselect(atoms=["CA", "C", "N", "CB", "O"])
data.prepare_dataset()
data.dataset = data.dataset.permute(0, 2, 1)
train_loader, test_loader = data.get_dataloader(
    batch_size=32, validation_split=0.1, manual_seed=25
)

checkpoint = torch.load(
    "./transformer_checkpoints/checkpoint_epoch1110_loss0.0009468746138736606.ckpt"
)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = Adam(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
epoch = checkpoint["epoch"]
loss = checkpoint["loss"]
model.eval()


raw_coord = []
with open("./transformer/coords/init.txt", "r") as cfile:
    for i in cfile:
        raw_coord.append(i.strip().split("  "))
raw_coord = np.asarray(raw_coord, dtype=float)


seq_len = list(data.dataset.shape)[1]
n_residues = [5, 20, 50, 100, 150, 200, 320]
given_seq = 120 * 5
for g in n_residues:
    given_seq = g * 5
    for bs, batch in enumerate(test_loader):
        for cs, sample in enumerate(batch[0]):
            print(f"Batch {bs} protein {cs}")
            sample = sample.permute(0, 1).cpu().numpy()
            start_sequence = sample[:given_seq, :]
            n_rand = start_sequence.shape[0] * start_sequence.shape[1]
            noise = (np.random.ranf(n_rand) * 0.05).reshape(start_sequence.shape)
            # add noise and randomly rotate coodrdiantes
            new_coords = start_sequence + noise  # @ qr_full()
            generated_coords = (
                # use this when rotation is used
                # generate_coordinates(model, new_coords[0], device, seq_len - given_seq)
                generate_coordinates(model, new_coords, device, seq_len - given_seq)
                * data.std
                + data.mean
            )
            sample = sample * data.std + data.mean
            gen_xyz(generated_coords, f"{args.outpath}/{g}lss_{cs}_pred.xyz")
            gen_xyz(sample, f"{args.outpath}/{g}lss_{cs}_gt.xyz")
            if cs > 5:
                break
        break
