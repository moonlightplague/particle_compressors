import unittest

import numpy as np

from experiments.velocity_id_relationship import (
    velocity_cuberoot_sum_cubes,
)


class VelocityIdRelationshipTests(unittest.TestCase):
    def test_requested_signed_cube_root_formula(self) -> None:
        vx = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        vy = np.array([2.0, 1.0, -4.0], dtype=np.float32)
        vz = np.array([2.0, 1.0, 1.0], dtype=np.float32)

        actual = velocity_cuberoot_sum_cubes(vx, vy, vz)
        expected = np.cbrt(
            vx.astype(np.float64) ** 3
            + vy.astype(np.float64) ** 3
            + vz.astype(np.float64) ** 3
        ).astype(np.float32)

        np.testing.assert_array_equal(actual, expected)
        self.assertLess(float(actual[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
