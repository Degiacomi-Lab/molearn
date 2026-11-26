import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))

from molearn.analysis.pdb_connect_RNA import add_connect_RNA


class TestPDBConnectRNA(unittest.TestCase):
    def test_add_connect_RNA(self):
        pdb = """\
ATOM      0  P   G       2       0.000   0.000   0.000  1.00  0.00           P
ATOM      1 C4'  G       2       1.000   0.000   0.000  1.00  0.00           C
ATOM      2  N9  G       2       2.000   0.000   0.000  1.00  0.00           N
ATOM      3  C2  G       2       3.000   0.000   0.000  1.00  0.00           C
ATOM      4  C6  G       2       4.000   0.000   0.000  1.00  0.00           C
ATOM      5  P   U       3       5.000   0.000   0.000  1.00  0.00           P
ATOM      6 C4'  U       3       6.000   0.000   0.000  1.00  0.00           C
ATOM      7  N1  U       3       7.000   0.000   0.000  1.00  0.00           N
ATOM      8  C2  U       3       8.000   0.000   0.000  1.00  0.00           C
ATOM      9  C4  U       3       9.000   0.000   0.000  1.00  0.00           C
"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "rna.pdb"
            filename.write_text(pdb)
            add_connect_RNA(filename)
            result = filename.read_text()

        self.assertIn("ATOM      1", result)
        self.assertIn("ATOM     10", result)
        self.assertIn("CONECT    2    1    3    6", result)
        self.assertIn("CONECT    3    2    4    5", result)
        self.assertIn("CONECT    9    8   10", result)
        self.assertTrue(result.endswith("END\n"))


if __name__ == "__main__":
    unittest.main()
