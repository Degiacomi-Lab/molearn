import sys
import os


try:
    sys.path.insert(0, os.path.join(os.path.abspath(os.pardir), "src"))
    from molearn.data import PDBData
except ModuleNotFoundError:
    sys.path.insert(
        0, os.path.join(os.path.split(os.path.abspath(os.pardir))[0], "src")
    )
    from molearn.data import PDBData


def main():
    data0 = PDBData()
    data0.import_pdb(
        "./data/preparation/MurDopen.dcd",
        "./data/preparation/topo_MurDopen1F.pdb",
    )
    data0.fix_terminal()
    data0.atomselect(atoms=["CA", "C", "N", "O", "CB"])
    data0.prepare_dataset()

    data1 = PDBData()
    data1.import_pdb(
        "./data/preparation/MurDopen.dcd",
        "./data/preparation/topo_MurDopen1F.pdb",
    )
    data1.fix_terminal()
    data1.atomselect(atoms=["CA", "C", "N", "O", "CB"])
    data1.prepare_dataset()

    print(f"\nOriginal shape of one frame data0: {data0.dataset[0].shape}")
    print(f"Original shape of one frame data1: {data1.dataset[0].shape}")

    # which dataset has the bigger (more atoms) protein
    ts = max(data0.dataset.shape[2], data1.dataset.shape[2])
    # in this example both trajectories have the same (sized) protein so there would be no effect
    # !!! FOR DEMONSTRATION ONLY adding needed patting size to ts
    ts = ts + 20

    data1.pad_and_cat([data0.dataset, data1.dataset], ts)
    print(f"     New shape of one frame data1: {data1.dataset[0].shape}")

    print(f"New size of data1 dataset:              {data1.dataset.shape}")
    print(f"New size of data0 dataset (unaffected): {data0.dataset.shape}")

    print(f"Padding shape: {(data1.dataset[0] == 0.0).shape}")
    nopad = data1.dataset[0][data1.dataset[0] != 0.0]
    print(f"Removed padding of one frame shape: {nopad.reshape((3,-1)).shape}")


if __name__ == "__main__":
    main()
