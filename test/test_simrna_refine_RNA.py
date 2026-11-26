import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(sys.path[0]), "src"))

from molearn.analysis.simrna_refine_RNA import (
    minimum_energy_frame_RNA,
    minimum_energy_trafl_frame_RNA,
    write_transition_trafl_RNA,
)


class TestOfficialSimRNARefinementRNA(unittest.TestCase):
    def test_minimum_unrestrained_energy_trafl_frame_RNA(self):
        # Header columns are: write, replica, restrained energy,
        # unrestrained energy, temperature. The restrained-energy minimum is
        # frame 1, while the required unrestrained-energy minimum is frame 2.
        trafl_RNA = """1 1 -100.0 -5.0 0.9
0 0 0 1 1 1
2 1 -10.0 -20.0 0.9
2 2 2 3 3 3
3 1 -50.0 -8.0 0.9
4 4 4 5 5 5
"""
        with tempfile.TemporaryDirectory() as directory_RNA:
            path_RNA = Path(directory_RNA) / "refinement.trafl"
            path_RNA.write_text(trafl_RNA)
            result_RNA = minimum_energy_trafl_frame_RNA(path_RNA)
            compatibility_result_RNA = minimum_energy_frame_RNA(path_RNA)

        self.assertEqual(result_RNA["frame_RNA"], 2)
        self.assertEqual(result_RNA["write_number_RNA"], 2)
        self.assertEqual(result_RNA["replica_number_RNA"], 1)
        self.assertEqual(result_RNA["frames_RNA"], 3)
        self.assertAlmostEqual(result_RNA["energy_RNA"], -20.0)
        self.assertAlmostEqual(result_RNA["restrained_energy_RNA"], -10.0)
        self.assertAlmostEqual(result_RNA["temperature_RNA"], 0.9)
        np.testing.assert_allclose(
            result_RNA["coordinates_RNA"], [[2, 2, 2], [3, 3, 3]]
        )
        np.testing.assert_allclose(
            compatibility_result_RNA["coordinates_RNA"],
            result_RNA["coordinates_RNA"],
        )

    def test_write_transition_trafl_RNA(self):
        results_RNA = [
            {
                "minimum_coordinate_line_RNA": "0 0 0 1 1 1",
                "minimum_replica_number_RNA": 3,
                "minimum_restrained_energy_RNA": -10.5,
                "minimum_energy_RNA": -11.5,
                "minimum_temperature_RNA": 0.9,
            },
            {
                "minimum_coordinate_line_RNA": "2 2 2 3 3 3",
                "minimum_replica_number_RNA": 4,
                "minimum_restrained_energy_RNA": -20.5,
                "minimum_energy_RNA": -21.5,
                "minimum_temperature_RNA": 0.8,
            },
        ]
        with tempfile.TemporaryDirectory() as directory_RNA:
            path_RNA = Path(directory_RNA) / "transition.trafl"
            returned_RNA = write_transition_trafl_RNA(results_RNA, path_RNA)
            lines_RNA = path_RNA.read_text().splitlines()

        self.assertEqual(returned_RNA, path_RNA)
        self.assertEqual(len(lines_RNA), 4)
        self.assertEqual(lines_RNA[0].split()[:2], ["1", "3"])
        self.assertEqual(lines_RNA[2].split()[:2], ["2", "4"])
        self.assertEqual(lines_RNA[1].split(), ["0", "0", "0", "1", "1", "1"])
        self.assertEqual(lines_RNA[3].split(), ["2", "2", "2", "3", "3", "3"])


if __name__ == "__main__":
    unittest.main()
