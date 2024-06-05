from __future__ import annotations
import os

import mdtraj as md
import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

np.random.seed(42)


class DataAssembler:
    def __init__(
        self,
        traj_path: str,
        topo_path: str | None = None,
        test_size: float = 0.15,
        n_cluster: int = 1500,
        image_mol: bool = False,
        outpath: str = "",
        verbose: bool = False,
    ):
        """
        Create clustered trajectories, stride trajectories and randomly sampled test frames
        and change either the respective trajectory frame indices or create a new trajectory.
        Will also concatenate multiple trajectories into one and center protein in the water box.
        The topology of the newly created trajectory will be saved in self.outpath/trajectory_name_NEW_TOPO.pdb

        :param str traj_path: path to the trajectory
        :param str topo_path: path to the topology
        :param int test_size: size of the test dataset (0.0 if no test set should be created)
        :param int n_cluster: number of clusters to be created (representative frames)
        :param bool image_mol: True to image to molecule (center it in the box)
        :param str outpath: directory path where the new trajector(y)ies should be stored
        :param bool verbose: True to get info which steps are currently performed
        """
        self.traj_path = traj_path
        self.topo_path = topo_path
        self.test_size = test_size
        self.n_cluster = n_cluster
        self.image_mol = image_mol
        self.outpath = outpath
        self.verbose = verbose
        assert os.path.exists(self.outpath), "Outpath does not exist"

    def _loading_fallback(self, traj_path, topo_path):
        """
        try loading trajectories that are not supported by md.load

        :param str traj_path: path(s) to the trajector(y)ies
        :param str traj_path: path(s) to the topolog(y)ies
        """
        ext = os.path.splitext(traj_path)[-1]

        match_dict = {
            # ".binops": md.load_binpos,
            ".xml": md.load_xml,
            ".dcd": md.load_dcd,
            ".dtr": md.load_dtr,
            ".nc": md.load_netcdf,
            ".ncrst": md.load_netcdf,
            ".trr": md.load_trr,
            ".xtc": md.load_xtc,
            ".xyz": md.load_xyz,
            ".lammpstrj": md.load_lammpstrj,
        }
        try:
            load_func = match_dict[ext]
        except KeyError:
            raise KeyError(
                "Even fallback loading was not able to load the supplied trajectory format"
            )

        """
        # use that for python versions >=3.10
        match ext:
            # case ".binops":
                # load_func = md.load_binpos
            case ".xml":
                load_func = md.load_xml
            case ".dcd":
                load_func = md.load_dcd
            case ".dtr":
                load_func = md.load_dtr
            case ".nc":
                load_func = md.load_netcdf
            case ".ncrst":
                load_func = md.load_netcdf
            case ".trr":
                load_func = md.load_trr
            case ".xtc":
                load_func = md.load_xtc
            case ".xyz":
                load_func = md.load_xyz
            case ".lammpstrj":
                load_func = md.load_lammpstrj
            case _:
                raise KeyError(
                    "Even fallback loading was not able to load the supplied trajectory format"
                )
        """
        return load_func(traj_path, topo_path)

    def read_traj(self) -> None:
        """
        Read in one or multiple trajectories, remove everything but protein atoms and
        image the molecule to center it in the water box, and create a training/validation
        and test split.
        """
        if self.verbose:
            print("Reading trajectory")
        self.traj_name = os.path.splitext(
            os.path.basename(
                self.traj_path if isinstance(self.traj_path, str) else self.traj_path[0]
            )
        )[0]
        # loading the trajectory if it is only one file
        if isinstance(self.traj_path, str):
            try:
                # do not enforce topology file on this formats
                fext = os.path.splitext(self.traj_path)[-1]
                if any([fext == ".pdb", fext == "h5", fext == "lh5"]):
                    self.traj = md.load(self.traj_path)
                else:
                    self.traj = md.load(self.traj_path, self.topo_path)
            except OSError:
                self.traj = self._loading_fallback(self.traj_path, self.topo_path)
            # select only protein atoms from the trajectory
            self.traj = self.traj.atom_slice(self.traj.top.select("protein"))
        # loading multiple trajectories
        elif isinstance(self.traj_path, list):
            if isinstance(self.traj_path, list) and self.topo_path is None:
                fext = os.path.splitext(self.traj_path[0])[-1]
                if any([fext == ".pdb", fext == "h5", fext == "lh5"]):
                    self.topo_path = [None] * len(self.traj_path)
            assert isinstance(
                self.topo_path, list
            ), "If trajectories are supplied as a list the topologies must be too"
            assert len(self.traj_path) == len(
                self.topo_path
            ), "there must be as many topologies supplied as trajectories"
            multi_traj = []
            top0 = None
            for ct, (trp, top) in enumerate(zip(self.traj_path, self.topo_path)):
                loaded = None
                try:
                    # do not enforce topology file on this formats
                    trp_ext = os.path.splitext(trp)
                    if any([trp_ext == ".pdb", trp_ext == "h5", trp_ext == "lh5"]):
                        loaded = md.load(trp)
                    else:
                        loaded = md.load(trp, top)
                except OSError:
                    loaded = self._loading_fallback(trp, top)
                # select only protein atoms from the trajectory
                loaded = loaded.atom_slice(loaded.top.select("protein"))
                # to be able to join them
                if ct == 0:
                    top0 = loaded.topology
                else:
                    if top0.n_atoms != loaded.n_atoms:
                        raise ValueError(
                            f"topologies do not match - topology [{ct}] has {loaded.n_atoms} instead of {top0.n_atoms}"
                        )
                    loaded.topology = top0
                multi_traj.append(loaded)
            self.traj = md.join(multi_traj)
        # converts ELEMENT names from eg "Cd" -> "C" to avoid later complications
        topo_table, topo_bonds = self.traj.topology.to_dataframe()
        topo_table["element"] = topo_table["element"].apply(
            lambda x: x if len(x.strip()) <= 1 else x.strip()[0]
        )
        self.traj.topology = md.Topology.from_dataframe(topo_table, topo_bonds)
        # save new topology
        self.traj[0].save_pdb(
            os.path.join(self.outpath, f"./{self.traj_name}_NEW_TOPO.pdb")
        )
        # Recenter and apply periodic boundary
        if self.image_mol:
            try:
                self.traj.image_molecules(inplace=True)
            except ValueError as e:
                print(f"Unable to image molecule due to '{e}' - will just recenter it")
        # maybe not needed after image_molecules
        self.traj.center_coordinates()
        n_frames = self.traj.n_frames
        # which index separated indices from training and test dataset
        self.test_border = int(n_frames * (1.0 - self.test_size))
        # get indices and shuffle them to obtain indices for use as training/validation
        # and one as test dataset
        self.frame_idx = np.arange(n_frames)
        np.random.shuffle(self.frame_idx)
        self.train_idx = self.frame_idx[: self.test_border]
        train_traj = self.traj[self.train_idx]
        # number of frames for training/validation set
        n_train_frames = len(self.train_idx)

        # remove H atoms from distance calculation
        atom_indices = [
            a.index for a in train_traj.topology.atoms if a.element.symbol != "H"
        ]
        # distance matrix between all frames
        self.traj_dists = np.empty((n_train_frames, n_train_frames))
        for i in range(n_train_frames):
            self.traj_dists[i] = md.rmsd(
                train_traj, train_traj, i, atom_indices=atom_indices
            )

    def calc_rmsd_f(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.int_]],
        np.ndarray[tuple[int], np.dtype[np.int_]],
        np.ndarray[tuple[int], np.dtype[np.int_]],
    ]:
        """
        calculate rmsd and rmsf over the course of the trajectory

        :return: tuple[
                np.ndarray[tuple[int], np.dtype[np.int_]],
                np.ndarray[tuple[int], np.dtype[np.int_]],
                np.ndarray[tuple[int], np.dtype[np.int_]],
            ]
        """
        assert hasattr(self, "traj"), "you first need to read in the trajectory"
        if self.verbose:
            print("Calculating rmsd")
        rmsd_analysis = md.rmsd(self.traj, self.traj, 0)
        rmsf_analysis = md.rmsf(self.traj, self.traj, 0)
        return np.arange(len(rmsd_analysis)), rmsd_analysis, rmsf_analysis

    def _find_representatives(self, idx_idx, labels) -> None:
        """
        search trough each cluster and find the representative frame
        with the highest similarity to all other frames

        param: list[int] idx_idx: indices of all frames in the trajectory
        param: list[int] labels: cluster label for each frame
        """
        if self.verbose:
            print("Finding representatives")
        self.cluster_idx = []
        # iterate over each cluster
        for i in np.unique(labels):
            # boolean mask for which frame is member of cluster i
            label_bool = labels == i
            # if cluster has only one memeber
            if label_bool.sum() == 1:
                iidx = 0
            else:
                # index the whole distance matrix to get a subdistance matirx
                # containing only the distances of cluster members
                i_dists = self.traj_dists[np.ix_(label_bool, label_bool)]
                # transform distances to similarities and find the frame wit the highest
                # similarity to all other cluster members
                iidx = np.exp(-1 * i_dists / i_dists.std()).sum(axis=1).argmax()
            # add the trajectory frame index of the representative to the list of used frames
            self.cluster_idx.append(idx_idx[label_bool][iidx])

    def distance_cluster(
        self,
    ) -> None:
        """
        cluster the trajectory with AgglomerativeClustering based on the rmsd between the frames
        """
        assert hasattr(
            self, "traj_dists"
        ), "No pairweise frame distances present - read in trajectory first"
        if self.verbose:
            print("Distance clustering")
        # replace AgglomerativeClustering with any distance matrix based clustering function
        cluster_func = AgglomerativeClustering(
            n_clusters=self.n_cluster, metric="precomputed", linkage="average"
        )
        # cluster the frames with KMeans and find the representative frame for each cluster
        cluster_func.fit(self.traj_dists)
        idx_idx = np.arange(len(cluster_func.labels_))
        self._find_representatives(idx_idx, cluster_func.labels_)
        self.cluster_method = "CLUSTER_aggl"

    def pca_cluster(self, n: int = 3) -> None:
        """
        cluster the trajectory with KMeans based on the first 5 principal components
        of a PCA of the trajectory

        :param int n: number of principal components (per frame) to use for the clustering
        """
        assert hasattr(self, "traj"), "No traj found - read in trajectory first"
        if self.verbose:
            print("PCA clustering")
        train_traj = self.traj[self.train_idx]
        # align everything to frame 0 and calculate the
        train_traj.superpose(train_traj, 0)
        pca = PCA(n_components=n)
        reduced_cartesian = pca.fit_transform(
            train_traj.xyz.reshape(len(self.train_idx), train_traj.n_atoms * 3)
        )
        # cluster the frames with KMeans and find the representative frame for each cluster
        kmeans = KMeans(n_clusters=self.n_cluster, n_init="auto")
        kmeans.fit(reduced_cartesian)
        idx_idx = np.arange(len(kmeans.labels_))
        self._find_representatives(idx_idx, kmeans.labels_)
        self.cluster_method = "CLUSTER_pca"

    def stride(self) -> None:
        """
        reduce the training dataset size to n samples using stride as 'sampling method'
        """
        if self.verbose:
            print("Reducing size")
        n_train_frames = len(self.train_idx)
        # indices that create a stride over the remaining training trajectory
        stride_idx = np.arange(
            0,
            n_train_frames,
            step=np.floor(n_train_frames / self.n_cluster).astype(int),
        )
        # train_idx need to be sorted so the strides are again over a normal trajectory
        # and not a shuffled one
        stride_pre_idx = self.train_idx.copy()
        stride_pre_idx.sort()
        # true indices of the train_traj
        ori_frame_train_idx_stride = stride_pre_idx[stride_idx]
        # but we need the indices that index the train_traj indices to be in line with
        # clustering methods that return a list of indices for the train_traj indices
        _, stride_idx, _ = np.intersect1d(
            self.frame_idx, ori_frame_train_idx_stride, return_indices=True
        )
        # randomly remove frames if stride calculation yielded in to many frames due to np.floor
        if len(stride_idx) > self.n_cluster:
            stride_idx = np.random.choice(
                stride_idx, size=self.n_cluster, replace=False
            )
        self.cluster_idx = stride_idx
        self.cluster_method = f"STRIDE_{self.n_cluster}"

    def _save_idx(
        self, filepath: str, idxs: list[int] | np.ndarray[tuple[int], np.dtype[np.int_]]
    ) -> None:
        """
        save frame indices to file (overwrites file if it already exists)

        :param str filepath: path where the indices will be stored
        :param list[int] idxs: the indices to be stored
        """
        if self.verbose:
            print("Saving indices")
        with open(filepath, "w+") as idx_file:
            for i in idxs:
                idx_file.write(f"{i}\n")

    def create_trajectories(
        self,
    ) -> None:
        """
        Saves clustered or strided indices for the present training trajectory
        and optionally saves them as new trajectory
        """
        assert all(
            [
                hasattr(self, "traj"),
                hasattr(self, "traj_name"),
                hasattr(self, "traj_dists"),
                hasattr(self, "train_idx"),
                hasattr(self, "frame_idx"),
                hasattr(self, "test_border"),
            ]
        ), "Read in trajectory first"

        assert hasattr(
            self, "cluster_idx"
        ), "No cluster indices optained by now - use any clustering method or stride to get frames from the trajectory"
        ori_frame_train_idx = self.train_idx[self.cluster_idx]

        if self.verbose:
            print("Creating trajectories")

        # create trajectories and the indices of the frames used
        self._save_idx(
            os.path.join(
                self.outpath,
                f"./{self.traj_name}_{self.cluster_method}_train_frames.txt",
            ),
            ori_frame_train_idx,
        )
        self.traj[ori_frame_train_idx].save_dcd(
            os.path.join(
                self.outpath, f"{self.traj_name}_{self.cluster_method}_train.dcd"
            )
        )

        # only create test trajectory if test size is greater than 0
        if self.test_size > 0.0:
            self._save_idx(
                os.path.join(self.outpath, f"./{self.traj_name}_test_frames.txt"),
                self.frame_idx[self.test_border :],
            )
            self.traj[self.frame_idx[self.test_border :]].save_dcd(
                os.path.join(self.outpath, f"./{self.traj_name}_test.dcd")
            )


if __name__ == "__main__":
    pass
