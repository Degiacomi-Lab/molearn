import os
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.models.foldingnet import AutoEncoder
from molearn.analysis import MolearnAnalysis, get_path, oversample
from molearn.data import PDBData


def main():
    # Note: running the code below within a function is necessary to ensure that
    # multiprocessing (used to calculate DOPE and Ramachandran) runs correctly

    print("> Loading network parameters...")
    fname = f"xbb_foldingnet_checkpoints{os.sep}checkpoint_epoch208_loss-4.205589803059896.ckpt"
    # if GPU is available we will use the GPU else the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(fname, map_location=device, weights_only=False)
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
        "./clustered/MurDopen_CLUSTER_aggl_train.dcd",
        "./clustered/MurDopen_NEW_TOPO.pdb",
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
    MA.setup_grid(grid_side_len)
    landscape_err_latent, landscape_err_3d, xaxis, yaxis = MA.scan_error()

    # OPTIONAL START - only for demonstration - use own start and end points for path
    # sort landscape by error
    flat_sort = landscape_err_latent.ravel().argsort()
    # flat index of the lowest error point on grid
    start = flat_sort[0]
    # how many structures to test to find the most distant point in high quality (low error) grid points
    n_test = 200
    # end of path
    end = flat_sort[:n_test][np.argmax(np.abs(flat_sort[:n_test] - start))]
    start_idx = np.unravel_index(start, landscape_err_latent.shape)
    end_idx = np.unravel_index(end, landscape_err_latent.shape)
    # OPTIONAL END

    # linear interpolation
    # use your true start and endpoint latent space coordinates as start and end
    latent_path = oversample(
        np.asarray(
            [
                # start coordinates
                [xaxis[start_idx[1]], yaxis[start_idx[0]]],
                # end coordinates
                [xaxis[end_idx[1]], yaxis[end_idx[0]]],
            ]
        )
    )
    # use A* to find the best path between start and end
    latent_path_astar = get_path(
        np.asarray(start_idx)[::-1],
        np.asarray(end_idx)[::-1],
        landscape_err_latent,
        xaxis,
        yaxis,
    )[0]

    # OPTIONAL START plotting of landscape with start end and path
    fig, ax = plt.subplots()
    sm = ax.pcolormesh(xaxis, yaxis, landscape_err_latent)
    cbar = fig.colorbar(sm, orientation="vertical")
    cbar.ax.set_ylabel("RMSD in Å", rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    ax.scatter(
        latent_path[:, 0], latent_path[:, 1], label="direct", s=3, color="firebrick"
    )
    ax.scatter(
        latent_path_astar[:, 0],
        latent_path_astar[:, 1],
        label="astar",
        s=3,
        color="forestgreen",
    )
    ax.scatter(
        xaxis[start_idx[1]], yaxis[start_idx[0]], label="start", s=10, color="black"
    )
    ax.scatter(xaxis[end_idx[1]], yaxis[end_idx[0]], label="end", s=10, color="yellow")
    plt.legend()
    plt.savefig("Error_grid_sampling.png", dpi=150)
    # OPTIONAL END

    # generating new structures
    # !!! relax=True will only work when trained on all atoms !!!
    latent_path = latent_path.reshape(1, -1, 2)
    latent_path_astar = latent_path_astar.reshape(1, -1, 2)
    if not os.path.isdir("newly_generated_structs_linear"):
        os.mkdir("newly_generated_structs_linear")
    if not os.path.isdir("newly_generated_structs_astar"):
        os.mkdir("newly_generated_structs_astar")
    MA.generate(latent_path, "newly_generated_structs_linear", relax=False)
    MA.generate(latent_path_astar, "newly_generated_structs_astar", relax=False)


if __name__ == "__main__":
    main()
