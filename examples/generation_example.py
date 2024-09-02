import os
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.models.foldingnet import AutoEncoder
from molearn.analysis import MolearnAnalysis
from molearn.data import PDBData


def main():
    # Note: running the code below within a function is necessary to ensure that
    # multiprocessing (used to calculate DOPE and Ramachandran) runs correctly

    print("> Loading network parameters...")

    fname = f"xbb_foldingnet_checkpoints{os.sep}checkpoint_epoch208_loss-4.205589803059896.ckpt"
    # if GPU is available we will use the GPU else the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(fname, map_location=device)
    net = AutoEncoder(**checkpoint["network_kwargs"])
    net.load_state_dict(checkpoint["model_state_dict"])

    # the network is currently on CPU. If GPU is available, move it there
    if torch.cuda.is_available():
        net.to(device)

    print("> Loading training data...")

    MA = MolearnAnalysis()
    MA.set_network(net)

    # increasing the batch size makes encoding/decoding operations faster,
    # but more memory demanding
    MA.batch_size = 4

    # increasing processes makes DOPE and Ramachandran scores calculations faster,
    # but more more memory demanding
    MA.processes = 2

    # what follows is a method to re-create the training and test set
    # by defining the manual see and loading the dataset in the same order as when
    # the neural network was trained, the same train-test split will be obtained
    data = PDBData()
    data.import_pdb(
        "./clustered/MurD_open_selection_CLUSTER_aggl_train.dcd",
        "./clustered/MurD_open_selection_NEW_TOPO.pdb",
    )
    data.fix_terminal()
    data.atomselect(atoms=["CA", "C", "N", "CB", "O"])
    data.prepare_dataset()
    data_train, data_test = data.split(manual_seed=25)

    # store the training and test set in the MolearnAnalysis instance
    # the second parameter of the following commands can be both a PDBData instance
    # or a path to a multi-PDB file
    MA.set_dataset("training", data_train)
    MA.set_dataset("test", data_test)

    print("> generating error landscape")
    # build a 50x50 grid. By default, it will be 10% larger than the region occupied
    # by all loaded datasets
    grid_side_len = 50
    MA.setup_grid(grid_side_len, padding=1.0)
    landscape_err_latent, landscape_err_3d, xaxis, yaxis = MA.scan_error()

    # argsort all errors as a 1D array
    sort_by_err = landscape_err_latent.ravel().argsort()

    # number of structures to generate
    n_structs = 10
    # only generate structures that have a low enough error - error threshold
    err_th = 1.5
    # boolean array which latent coords from the 1D array have a low enough error
    err_not_reached = landscape_err_latent.ravel()[sort_by_err] < err_th
    # get the latent coords with the lowest errors
    coords_oi = np.asarray(
        [
            [xaxis[i // grid_side_len], yaxis[i % grid_side_len]]
            for i in sort_by_err[:n_structs]
        ]
    )

    # still mask them to be below the error threshold
    coords_oi = coords_oi[err_not_reached[:n_structs]].reshape(1, -1, 2)
    assert (
        coords_oi.shape[1] > 0
    ), "No latent coords available, try raising your error threshold (err_th)"

    # generated structures will be created in ./newly_generated_structs
    if not os.path.isdir("newly_generated_structs"):
        os.mkdir("newly_generated_structs")
    # use relax=True to also relax the generated structures
    # !!! relax=True will only work when trained on all atoms !!!
    MA.generate(coords_oi, "newly_generated_structs")


if __name__ == "__main__":
    main()
