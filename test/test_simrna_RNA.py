import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))

from molearn.data import SimRNAData_RNA


class TestSimRNADataRNA(unittest.TestCase):
    def test_stream_trc_with_trailing_coordinate_RNA(self):
        hdr = """$simulation
SimRNA

$trcHeader
ID_RUN ID_FRM TEMP

$moleculeTypes
RNA

$chains
RNA _A

$restraints
"""
        pdb = """ATOM      1  P     G A   1       0.000   0.000   0.000  1.00  0.00           P
ATOM      2  C4'   G A   1       1.000   0.000   0.000  1.00  0.00           C
END
"""
        trc = """HEADER run 1 0.9
RNA_A 0 0 0 1 0 0 99 99 99
HEADER run 2 0.9
RNA_A 0 1 0 1 1 0 99 99 99
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test.hdr").write_text(hdr)
            (root / "test.pdb").write_text(pdb)
            (root / "test.trc").write_text(trc)
            data = SimRNAData_RNA()
            data.import_simrna_RNA(
                root / "test.trc", root / "test.hdr", root / "test.pdb"
            )
            data.atomselect_RNA(["P", "C4'"])
            dataset = data.prepare_dataset_RNA(align_RNA=False)

        self.assertEqual(tuple(dataset.shape), (2, 2, 3))
        self.assertEqual(data.frames_per_trajectory_RNA, [2])
        self.assertEqual(data.unmapped_trc_atoms_RNA, [[1]])
        self.assertEqual(data.frame_headers_RNA[0][1]["ID_FRM"], "2")


if __name__ == "__main__":
    unittest.main()
