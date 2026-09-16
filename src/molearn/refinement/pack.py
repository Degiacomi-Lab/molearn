"""Side-chain packing with FASPR."""
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

#: environment variable consulted when no explicit path is given
FASPR_ENV = "FASPR_BIN"


def faspr_path(faspr=None):
    """Locate the FASPR binary: explicit argument, then ``$FASPR_BIN``, then ``PATH``.

    :raises FileNotFoundError: with installation guidance if none of the three resolve.
    """
    for cand in (faspr, os.environ.get(FASPR_ENV), shutil.which("FASPR")):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise FileNotFoundError(
        "FASPR executable not found. Pass faspr=<path>, set the "
        f"{FASPR_ENV} environment variable, or put FASPR on PATH. "
        "Source: https://github.com/tommyhuangthu/FASPR"
    )


def write_backbone_pdbs(coords, mol, out_dir, prefix="bb"):
    """Write one PDB per structure from ``[B, n_atoms, 3]`` coordinates.

    :param mol: biobox molecule matching the atom selection, used for topology.
    """
    from copy import deepcopy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = deepcopy(mol)
    paths = []
    for i, frame in enumerate(coords):
        m.coordinates = frame.reshape(1, -1, 3)
        m.set_current(0)
        p = out_dir / f"{prefix}_{i:05d}.pdb"
        m.write_pdb(str(p))
        paths.append(str(p))
    return paths


def _faspr_one(args):
    binary, pdb_in, pdb_out = args
    r = subprocess.run([binary, "-i", pdb_in, "-o", pdb_out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pdb_out if (r.returncode == 0 and os.path.exists(pdb_out)) else None


def pack_sidechains(pdb_paths, out_dir, faspr=None, workers=None, prefix="packed"):
    """Add side chains to backbone-only PDBs.

    FASPR is single-threaded, so one process per core is used.

    :returns: ``(packed_paths, packable)`` where ``packable`` is a per-input bool list.
    """
    binary = faspr_path(faspr)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    jobs = [(binary, p, str(out_dir / f"{prefix}_{Path(p).stem.split('_')[-1]}.pdb"))
            for p in pdb_paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_faspr_one, jobs))
    return [p for p in out if p is not None], [p is not None for p in out]
