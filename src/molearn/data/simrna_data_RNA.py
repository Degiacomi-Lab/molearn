"""Streaming reader for SimRNA text ``.trc`` trajectories."""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import torch

from .pdb_data_RNA import PDBData_RNA


class SimRNAData_RNA(PDBData_RNA):
    """Read SimRNA TRC coordinates using its HDR and PDB metadata files.

    The PDB is authoritative for physical atom identity and order. The HDR
    supplies the trajectory-header fields and chain labels. Any trailing TRC
    coordinates not represented in the PDB are SimRNA dummy sites used for
    pseudo-torsion calculations and are deliberately excluded from training.
    TRC files are streamed so frame slicing can avoid parsing coordinates for
    frames that are not selected.
    """

    @staticmethod
    def read_hdr_RNA(hdr_file):
        """Parse the named sections of a SimRNA HDR file."""
        hdr_file = Path(hdr_file)
        sections = {}
        current = None
        for raw_line in hdr_file.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("$"):
                current = line[1:]
                sections[current] = []
            elif line and current is not None:
                sections[current].append(line)

        required = {"simulation", "trcHeader", "moleculeTypes", "chains"}
        missing = required.difference(sections)
        if missing:
            raise ValueError(f"Missing HDR sections in {hdr_file}: {sorted(missing)}")
        if not sections["trcHeader"]:
            raise ValueError(f"Empty $trcHeader section in {hdr_file}")

        sections["trcHeader_fields_RNA"] = sections["trcHeader"][0].split()
        sections["coordinate_labels_RNA"] = [
            "".join(line.split()) for line in sections["chains"]
        ]
        return sections

    def import_simrna_trajectories_RNA(self, trc_files, hdr_files, pdb_files):
        """Register one or more matching TRC/HDR/PDB trajectory triplets."""
        if not (len(trc_files) == len(hdr_files) == len(pdb_files)):
            raise ValueError("Each SimRNA TRC must have one HDR and one PDB file")
        if not trc_files:
            raise ValueError("At least one SimRNA trajectory is required")

        self._simrna_sources_RNA = []
        self._mols_RNA = []
        for trc_file, hdr_file, pdb_file in zip(trc_files, hdr_files, pdb_files):
            paths = [Path(trc_file), Path(hdr_file), Path(pdb_file)]
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError("Missing SimRNA files: " + ", ".join(missing))
            universe = mda.Universe(str(paths[2]))
            self._mols_RNA.append(universe)
            self._simrna_sources_RNA.append(
                {
                    "trc": paths[0],
                    "hdr": paths[1],
                    "pdb": paths[2],
                    "metadata": self.read_hdr_RNA(paths[1]),
                    "pdb_atom_count": len(universe.atoms),
                }
            )
        self._mol = self._mols_RNA[0]

    def import_simrna_RNA(self, trc_file, hdr_file, pdb_file):
        """Register a single SimRNA TRC/HDR/PDB trajectory."""
        return self.import_simrna_trajectories_RNA(
            [trc_file], [hdr_file], [pdb_file]
        )

    def atomselect_RNA(self, atoms=("P", "C4'")):
        """Select PDB-mapped atoms while retaining their TRC coordinate indices."""
        super().atomselect_RNA(atoms)
        self._simrna_selected_indices_RNA = [
            molecule.atoms.indices.astype(int).copy() for molecule in self._mols_RNA
        ]

    def _read_trc_RNA(self, source, selected_indices, frame_slice):
        """Stream selected frames and return coordinates plus HEADER metadata."""
        start = 0 if frame_slice.start is None else frame_slice.start
        stop = frame_slice.stop
        step = 1 if frame_slice.step is None else frame_slice.step
        if start < 0 or (stop is not None and stop < 0) or step < 1:
            raise ValueError("SimRNA frame slicing requires non-negative values")

        fields = source["metadata"]["trcHeader_fields_RNA"]
        valid_labels = set(source["metadata"]["coordinate_labels_RNA"])
        atom_count = source["pdb_atom_count"]
        coordinates = []
        headers = []
        unmapped_atom_counts = set()

        with source["trc"].open() as trajectory:
            frame = 0
            while True:
                header_line = trajectory.readline()
                if not header_line:
                    break
                coordinate_line = trajectory.readline()
                if not coordinate_line:
                    raise ValueError(f"Incomplete final frame in {source['trc']}")
                if stop is not None and frame >= stop:
                    break
                selected = frame >= start and (frame - start) % step == 0
                if selected:
                    header_values = header_line.split()
                    if not header_values or header_values[0] != "HEADER":
                        raise ValueError(
                            f"Expected HEADER at frame {frame} in {source['trc']}"
                        )
                    values = header_values[1:]
                    if len(values) != len(fields):
                        raise ValueError(
                            f"Frame {frame} has {len(values)} header values; "
                            f"HDR declares {len(fields)}"
                        )

                    label, separator, coordinate_text = coordinate_line.partition(" ")
                    if not separator or label not in valid_labels:
                        raise ValueError(
                            f"Unexpected coordinate label {label!r} at frame {frame}"
                        )
                    flat = np.fromstring(coordinate_text, sep=" ", dtype=np.float32)
                    if flat.size % 3:
                        raise ValueError(
                            f"Frame {frame} coordinate count is not divisible by 3"
                        )
                    trc_atom_count = flat.size // 3
                    if trc_atom_count < atom_count:
                        raise ValueError(
                            f"Frame {frame} has {trc_atom_count} TRC atoms but "
                            f"the PDB defines {atom_count}"
                        )
                    unmapped_atom_counts.add(trc_atom_count - atom_count)
                    coordinates.append(
                        flat.reshape(-1, 3)[:atom_count][selected_indices].copy()
                    )
                    headers.append(dict(zip(fields, values)))
                frame += 1

        if not coordinates:
            raise ValueError(f"The requested slice contains no frames in {source['trc']}")
        return np.asarray(coordinates), headers, unmapped_atom_counts

    def prepare_dataset_RNA(
        self,
        std=None,
        mean=None,
        start=None,
        stop=None,
        step=None,
        align_RNA=True,
        alignment_reference_RNA=None,
    ):
        """Stream, align, and normalize registered SimRNA trajectories."""
        if not hasattr(self, "_simrna_sources_RNA"):
            raise ValueError("Call import_simrna_RNA before preparing the dataset")
        if not hasattr(self, "_simrna_selected_indices_RNA"):
            self.atomselect_RNA("all")
        if std is not None and mean is not None:
            self.std, self.mean = std, mean
        elif (std is None) != (mean is None):
            raise ValueError("std and mean must be supplied together")

        frame_slice = slice(start, stop, step)
        results = [
            self._read_trc_RNA(source, indices, frame_slice)
            for source, indices in zip(
                self._simrna_sources_RNA, self._simrna_selected_indices_RNA
            )
        ]
        coordinate_sets = [result[0] for result in results]
        self.frame_headers_RNA = [result[1] for result in results]
        self.unmapped_trc_atoms_RNA = [sorted(result[2]) for result in results]
        # Temporary compatibility name while the RNA API is under development.
        self.extra_trc_coordinates_RNA = self.unmapped_trc_atoms_RNA
        coords = np.concatenate(coordinate_sets, axis=0)

        if align_RNA:
            if alignment_reference_RNA is None:
                alignment_reference_RNA = coords[0].copy()
            coords = self.align_coordinates_RNA(coords, alignment_reference_RNA)
        elif alignment_reference_RNA is not None:
            raise ValueError(
                "alignment_reference_RNA cannot be used when align_RNA is False"
            )
        else:
            alignment_reference_RNA = coords[0].copy()
        self.alignment_reference_RNA = np.asarray(alignment_reference_RNA).copy()

        self._compute_indices_RNA()
        coords = self._standardise_coordinates(coords)
        self.dataset = torch.from_numpy(coords).float()
        self._atom_names = list(np.unique(self._mol.atoms.names))
        self.frame_slice_RNA = frame_slice
        self.frames_per_trajectory_RNA = [len(values) for values in coordinate_sets]
        print(f"SimRNA dataset shape: {self.dataset.shape}")
        print(
            "Unmapped SimRNA dummy sites per frame: "
            f"{self.unmapped_trc_atoms_RNA}"
        )
        return self.dataset

    prepare_dataset = prepare_dataset_RNA
