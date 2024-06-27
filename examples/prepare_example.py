#%%
import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
from molearn.data import DataAssembler


def main():
    storage_path = "./clustered"
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)
    tm = DataAssembler(
        # trajectories
        [
            "./data/preparation/calmodulin_mega.pdb",
            # "./data/preparation/MurDopen.dcd",
            # "./data/preparation/MurDclosed.dcd",
        ],
        # topologies
<<<<<<< HEAD
        # [
        #     "./data/preparation/topo_MurDopen1F.pdb",
        #     "./data/preparation/topo_MurDclosed1F.pdb",
        # ],
        test_size=0.15,
        n_cluster=10,
=======
        [
            "./data/preparation/topo_MurDopen1F.pdb",
            "./data/preparation/topo_MurDclosed1F.pdb",
        ],
        test_size=0.0,
        n_cluster=15,
>>>>>>> 28aff03 (added ability to align on subset of atoms)
        outpath=storage_path,
        verbose=True,
        image_mol=True,
    )
    # reading in the trajectories and removing of all atoms apart from protein atoms
    tm.read_traj()
    # using agglomerative clustering to sample the trajectories
    tm.distance_cluster()
    # creating a dendrogram of the agglomerative clustering
    tm.create_dendrogram()
    # creating the new trajectory as dcd file and a new topology as pdb file
    tm.create_trajectories()
    # using PCA and the first n components for KMeans clustering to sample the trajectories
    tm.pca_cluster()
    tm.create_trajectories()
    # simply striding over the trajectories with a step size computed to result in n_cluster frames
    tm.stride()
    tm.create_trajectories()
    tm.create_trajectories_by_cluster(test_cluster=1)
    """
    the given example wilc prel create the following files in a new directory named 'clustered'
    *   MurDopen_CLUSTER_aggl_train.dcd
    *   MurDopen_CLUSTER_aggl_train_frames.txt
    *   MurDopen_CLUSTER_pca_train.dcd
    *   MurDopen_CLUSTER_pca_train_frames.txt
    *   MurDopen_NEW_TOPO.pdb
    *   MurDopen_STRIDE_5_train.dcd
    *   MurDopen_STRIDE_5_train_frames.txt
    the txt files contain the indices of frames of the original trajectory
    the dcd files contain the new trajectory
    the pdb file is the new topology for the new trajectory
    all atoms apart from protein atoms will be removed
    """


if __name__ == "__main__":
    main()

# %%
