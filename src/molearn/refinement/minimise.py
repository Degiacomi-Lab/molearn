"""Protonation and energy minimisation of packed structures with OpenMM.

Default force field is ff14SB with GBn2 implicit solvent. The topology is identical for
every frame, so the System and Context are built once and only positions change;
rebuilding per frame otherwise dominates the run.
"""
from pathlib import Path

import numpy as np

from ..scoring.forcefield_score import KT_298, harmonic_params, strain

#: default ff14SB + GBn2
DEFAULT_FORCEFIELD = ("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")


def pick_platform(preferred=("CUDA", "OpenCL", "CPU")):
    """First OpenMM platform that loads, as ``(platform, name)``."""
    import openmm

    for name in preferred:
        try:
            return openmm.Platform.getPlatformByName(name), name
        except Exception:
            continue
    raise RuntimeError("no usable OpenMM platform")


def minimise_staged(sim, positions, bonds, angles, unit, block=100, max_iter=1000,
                    tolerance=10.0, good_frac=0.005, patience=2, rel_improve=0.02):
    """Minimise in blocks, stopping when backbone bond strain converges or plateaus.

    Minimising to a force tolerance spends most of its effort on side chains and
    long-range GB terms: a structure already at 0.00% of bonds beyond 5 kT can still
    report 49 kJ/mol/nm RMS force, at ~45 ms per iteration.

    L-BFGS history resets on each ``minimizeEnergy`` call, so ``block`` should be large
    enough for the optimiser to build curvature information.

    :returns: dict with ``ok``, ``reason`` (``converged``/``plateau``/``maxiter``/
        ``diverged:*``), ``iterations``, and on success ``energy``, ``rms_force``,
        ``positions``, ``pos_nm``, ``e_bond``, ``e_angle``.
    """
    sim.context.setPositions(positions)
    used, stall, prev = 0, 0, None
    reason, state = "maxiter", None
    e_bond = e_angle = None

    while used < max_iter:
        n = min(block, max_iter - used)
        try:
            sim.minimizeEnergy(
                tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=n)
        except Exception as exc:
            # a structure the force field cannot relax is a result, not an error
            return {"ok": False, "reason": f"diverged:{type(exc).__name__}",
                    "iterations": used}
        used += n
        state = sim.context.getState(getEnergy=True, getPositions=True, getForces=True)
        pos_nm = np.array(state.getPositions().value_in_unit(unit.nanometer))
        e_bond, e_angle = strain(pos_nm, bonds, angles)
        frac = float((e_bond > 5 * KT_298).mean())

        if frac <= good_frac:
            reason = "converged"
            break
        if prev is not None and (prev - frac) < rel_improve * max(prev, 1e-12):
            stall += 1
            if stall >= patience:
                reason = "plateau"
                break
        else:
            stall = 0
        prev = frac

    f = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    return {"ok": True, "reason": reason, "iterations": used,
            "energy": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            "rms_force": float(np.sqrt((np.linalg.norm(f, axis=1) ** 2).mean())),
            "positions": state.getPositions(),
            "pos_nm": np.array(state.getPositions().value_in_unit(unit.nanometer)),
            "e_bond": e_bond, "e_angle": e_angle}


def minimise_structures(pdb_paths, out_dir=None, forcefield=DEFAULT_FORCEFIELD,
                        max_iter=1000, tolerance=10.0, device="0", staged=True,
                        pH=7.4, score_forcefield=True):
    """Protonate and minimise each PDB, reporting whether it survived.

    :param out_dir: if given, minimised structures are written here.
    :param staged: use :func:`minimise_staged` rather than a single ``minimizeEnergy``.
    :param score_forcefield: also return bond/angle strain statistics per structure.
    :returns: one dict per input with ``file``, ``minimisable``, ``e_before``,
        ``e_after``, ``rms_force``, ``reason`` and optionally the strain summary.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from pdbfixer import PDBFixer

    from ..scoring.forcefield_score import strain_summary, term_kinds

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    platform, pname = pick_platform()
    props = {"DeviceIndex": str(device)} if pname in ("CUDA", "OpenCL") else None
    ff = app.ForceField(*forcefield)

    sim = top_ref = variants = bonds = angles = None
    n_ref = None
    results = []

    for p in pdb_paths:
        rec = {"file": str(p), "minimisable": False}
        try:
            fixer = PDBFixer(str(p))
            fixer.findMissingResidues()
            fixer.missingResidues = {}          # never invent residues
            fixer.findMissingAtoms()            # adds the terminal OXT
            fixer.addMissingAtoms()

            # Hydrogens via Modeller rather than PDBFixer so protonation states can be
            # pinned: left alone, the HID/HIE choice varies per frame, the atom count
            # changes with it, and the shared System no longer applies.
            modeller = app.Modeller(fixer.topology, fixer.positions)
            if variants is None:
                variants = modeller.addHydrogens(ff, pH=pH)
            else:
                modeller.addHydrogens(ff, pH=pH, variants=variants)
            topology, positions = modeller.topology, modeller.positions

            if sim is None or topology.getNumAtoms() != n_ref:
                system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff,
                                         constraints=None)
                integrator = openmm.LangevinMiddleIntegrator(
                    298 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
                sim = app.Simulation(topology, system, integrator, platform, props)
                n_ref, top_ref = topology.getNumAtoms(), topology
                bonds, angles = harmonic_params(system, topology)

            sim.context.setPositions(positions)
            rec["e_before"] = sim.context.getState(
                getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole)

            if staged:
                res = minimise_staged(sim, positions, bonds, angles, unit,
                                      max_iter=max_iter, tolerance=tolerance)
            else:
                sim.minimizeEnergy(
                    tolerance=tolerance * unit.kilojoule_per_mole / unit.nanometer,
                    maxIterations=max_iter)
                st = sim.context.getState(getEnergy=True, getPositions=True,
                                          getForces=True)
                f = st.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer)
                pos_nm = np.array(st.getPositions().value_in_unit(unit.nanometer))
                e_b, e_a = strain(pos_nm, bonds, angles)
                res = {"ok": True, "reason": "maxiter", "iterations": max_iter,
                       "energy": st.getPotentialEnergy().value_in_unit(
                           unit.kilojoule_per_mole),
                       "rms_force": float(np.sqrt(
                           (np.linalg.norm(f, axis=1) ** 2).mean())),
                       "positions": st.getPositions(), "pos_nm": pos_nm,
                       "e_bond": e_b, "e_angle": e_a}

            rec["reason"] = res["reason"]
            rec["iterations"] = res["iterations"]
            if not res["ok"]:
                results.append(rec)
                continue

            rec.update(minimisable=True, e_after=res["energy"],
                       rms_force=res["rms_force"])
            if score_forcefield:
                rec.update(strain_summary(res["e_bond"], "bond", term_kinds(bonds)))
                rec.update(strain_summary(res["e_angle"], "angle", term_kinds(angles)))
            if out_dir is not None:
                out_pdb = out_dir / f"{Path(p).stem}_minimised.pdb"
                with open(out_pdb, "w") as fh:
                    app.PDBFile.writeFile(top_ref, res["positions"], fh, keepIds=True)
                rec["file"] = str(out_pdb)
        except Exception as exc:
            rec["reason"] = f"failed:{type(exc).__name__}: {exc}"
        results.append(rec)

    return results
