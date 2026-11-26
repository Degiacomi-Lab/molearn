"""PDB connectivity helpers for Molearn's reduced RNA representation."""

from pathlib import Path


def _atom_records_RNA(lines):
    records = []
    for line_index, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")):
            records.append(
                {
                    "line_index": line_index,
                    "name": line[12:16].strip(),
                    "resname": line[17:21].strip(),
                    "chain": line[21:22],
                    "resid": line[22:26].strip(),
                }
            )
    return records


def add_connect_RNA(pdb_file, output_file=None):
    """Add coarse-grained RNA ``CONECT`` records to a generated PDB.

    The graph is ``P--C4'--N`` with the base triangle
    ``N--C2--C4/C6--N`` within each nucleotide, plus ``C4'(i)--P(i+1)``
    along each chain. These links make the reduced model
    display as a connected RNA in molecular viewers; they are not all-atom
    covalent bonds and must not be used as an OpenMM topology.

    Atom serial numbers are also rewritten to valid one-based PDB serials.
    The input file is updated in place unless ``output_file`` is supplied.
    """
    pdb_file = Path(pdb_file)
    destination = Path(output_file) if output_file is not None else pdb_file
    lines = pdb_file.read_text().splitlines()
    records = _atom_records_RNA(lines)
    if not records:
        raise ValueError(f"No ATOM or HETATM records found in {pdb_file}")

    # Molearn/biobox writes zero-based serials. Renumber before making CONECT.
    for serial, record in enumerate(records, start=1):
        record["serial"] = serial
        index = record["line_index"]
        line = lines[index]
        lines[index] = f"{line[:6]}{serial:5d}{line[11:]}"

    residues = []
    residue_lookup = {}
    for record in records:
        key = (record["chain"], record["resid"], record["resname"])
        if key not in residue_lookup:
            residue_lookup[key] = {}
            residues.append((key, residue_lookup[key]))
        residue_lookup[key][record["name"]] = record["serial"]

    bonds = set()

    def bond_RNA(first, second):
        if first is not None and second is not None and first != second:
            bonds.add(tuple(sorted((first, second))))

    previous_key = None
    previous_atoms = None
    for key, atoms in residues:
        base_n = atoms.get("N9", atoms.get("N1"))
        base_outer = atoms.get("C6", atoms.get("C4"))
        bond_RNA(atoms.get("P"), atoms.get("C4'"))
        bond_RNA(atoms.get("C4'"), base_n)
        bond_RNA(base_n, atoms.get("C2"))
        bond_RNA(atoms.get("C2"), base_outer)
        bond_RNA(base_outer, base_n)

        if previous_atoms is not None and previous_key[0] == key[0]:
            bond_RNA(previous_atoms.get("C4'"), atoms.get("P"))
        previous_key, previous_atoms = key, atoms

    neighbours = {record["serial"]: [] for record in records}
    for first, second in sorted(bonds):
        neighbours[first].append(second)
        neighbours[second].append(first)

    body = [
        line for line in lines
        if not line.startswith("CONECT") and line.strip() not in {"END", "ENDMDL"}
    ]
    conect = [
        "CONECT" + f"{serial:5d}" + "".join(f"{other:5d}" for other in linked)
        for serial, linked in neighbours.items()
        if linked
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body + conect + ["END"]) + "\n")
    return destination


def add_connect_directory_RNA(pdb_directory, pattern="*.pdb"):
    """Add RNA connectivity to every matching PDB in a directory."""
    pdb_directory = Path(pdb_directory)
    files = sorted(pdb_directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files found in {pdb_directory}")
    return [add_connect_RNA(filename) for filename in files]
