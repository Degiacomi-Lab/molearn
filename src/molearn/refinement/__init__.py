"""Structure refinement: side-chain packing (FASPR) and energy minimisation (OpenMM).

``refine`` runs both and reports, per structure, whether it was *packable* and
*minimisable* -- the pass/fail pair used as a quality metric alongside
:mod:`molearn.scoring.geometry_score`.
"""
import tempfile
from contextlib import ExitStack
from pathlib import Path

from .minimise import (DEFAULT_FORCEFIELD, minimise_staged, minimise_structures,
                       pick_platform)
from .pack import FASPR_ENV, faspr_path, pack_sidechains, write_backbone_pdbs

__all__ = ["refine", "pack_sidechains", "write_backbone_pdbs", "faspr_path",
           "minimise_structures", "minimise_staged", "pick_platform",
           "DEFAULT_FORCEFIELD", "FASPR_ENV"]


def refine(coords, mol, out_dir=None, faspr=None, workers=None, keep_intermediate=False,
           **minimise_kwargs):
    """Pack side chains onto backbone coordinates, then minimise.

    :param coords: ``[B, n_atoms, 3]`` in Angstrom.
    :param mol: biobox molecule matching the atom selection.
    :param out_dir: where minimised PDBs are written; a temporary directory is used and
        discarded if omitted.
    :param keep_intermediate: keep the backbone and packed PDBs alongside the output.
    :returns: one record per input structure with ``packable``, ``minimisable``,
        ``e_before``, ``e_after``, ``rms_force`` and the force-field strain summary.
    """
    with ExitStack() as stack:
        if out_dir is None:
            out_dir = stack.enter_context(tempfile.TemporaryDirectory())
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # every temporary directory goes on the stack, so none is left behind on an
        # exception or when keep_intermediate is False
        work = out_dir if keep_intermediate else Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="molearn_refine_")))

        bb_paths = write_backbone_pdbs(coords, mol, work)
        packed, packable = pack_sidechains(bb_paths, work, faspr=faspr, workers=workers)
        minimised = minimise_structures(packed, out_dir=out_dir, **minimise_kwargs)

        # minimise_structures only sees the structures FASPR produced, so realign its
        # records with the full input list
        it = iter(minimised)
        records = []
        for i, ok in enumerate(packable):
            if ok:
                rec = dict(next(it))
                rec["packable"] = True
            else:
                rec = {"packable": False, "minimisable": False,
                       "reason": "faspr_failed"}
            rec["index"] = i
            records.append(rec)
        return records
