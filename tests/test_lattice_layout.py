"""Tests for the reversible periodic ID-lattice transform."""

import unittest

import numpy as np

from src.lattice_layout import (
    IDENTITY_TRANSFORM,
    POSITION_RESIDUAL_TRANSFORM,
    LatticeLayoutUnavailable,
    infer_complete_lattice_layout,
    infer_dense_lattice_layout,
    lattice_layout_from_metadata,
)


class DenseLatticeLayoutTests(unittest.TestCase):
    @staticmethod
    def _periodic_slab(drop_stride=None):
        side = 50
        high, middle, low = np.meshgrid(
            np.array([48, 49, 0, 1], dtype=np.int64),
            np.arange(side, dtype=np.int64),
            np.arange(side, dtype=np.int64),
            indexing="ij",
        )
        high = high.reshape(-1)
        middle = middle.reshape(-1)
        low = low.reshape(-1)
        ids = ((high * side + middle) * side + low).astype(np.uint64)
        if drop_stride is not None:
            cell_hash = (
                high * 73_856_093
                + middle * 19_349_663
                + low * 83_492_791
            )
            keep = cell_hash % drop_stride != 0
            high, middle, low, ids = (
                values[keep] for values in (high, middle, low, ids)
            )
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        high, middle, low = high[order], middle[order], low[order]
        positions = {
            "x": ((low + 0.125) / side).astype(np.float32),
            "y": ((high + 0.25) / side).astype(np.float32),
            "z": ((middle + 0.375) / side).astype(np.float32),
        }
        return side, ids, positions, (high, middle, low)

    def test_infers_periodic_geometry_and_position_axis_mapping(self):
        side, ids, positions, _ = self._periodic_slab()
        layout = infer_dense_lattice_layout(ids, positions, side, 0.8)

        self.assertEqual(layout.id_base, 0)
        self.assertEqual(layout.starts, (48, 0, 0))
        self.assertEqual(layout.shape, (4, 50, 50))
        self.assertEqual(layout.position_digit_axes, (2, 0, 1))
        self.assertEqual(layout.dense_count, 10_000)
        self.assertEqual(layout.occupancy, 1.0)

    def test_position_residual_and_sparse_dense_fill_are_reversible(self):
        side, ids, positions, coordinates = self._periodic_slab(101)
        layout = infer_dense_lattice_layout(ids, positions, side, 0.8)
        self.assertGreater(layout.missing_indices.size, 0)

        dense, transform_error, wrap_offsets = layout.encode_field(
            positions["x"],
            "x",
            position_residual=True,
        )
        self.assertTrue(np.all(np.isfinite(dense)))
        self.assertLessEqual(transform_error, np.finfo(np.float32).eps)
        restored = layout.decode_field(
            dense,
            "x",
            POSITION_RESIDUAL_TRANSFORM,
            np.dtype("float32"),
            wrap_offsets,
        )
        np.testing.assert_allclose(restored, positions["x"], atol=1e-7, rtol=0)

        velocity = (
            coordinates[0] * 0.5
            + coordinates[1] * 0.25
            - coordinates[2]
        ).astype(np.float32)
        dense_velocity, _, velocity_wrap = layout.encode_field(
            velocity,
            "vx",
            position_residual=False,
        )
        self.assertIsNone(velocity_wrap)
        decoded_velocity = layout.decode_field(
            dense_velocity,
            "vx",
            IDENTITY_TRANSFORM,
            np.dtype("float32"),
        )
        np.testing.assert_array_equal(decoded_velocity, velocity)

        restored_layout = lattice_layout_from_metadata(
            ids,
            layout.manifest_metadata(),
        )
        restored_again = restored_layout.decode_field(
            dense_velocity,
            "vx",
            IDENTITY_TRANSFORM,
            np.dtype("float32"),
        )
        np.testing.assert_array_equal(restored_again, velocity)

    def test_native_position_order_is_preserved_by_velocity_lattice(self):
        side, ids, positions, coordinates = self._periodic_slab(101)
        native_order = np.random.default_rng(42).permutation(ids.size)
        ordered_ids = ids[native_order]
        ordered_positions = {
            logical: values[native_order]
            for logical, values in positions.items()
        }
        velocity = (
            coordinates[0] * 0.5
            + coordinates[1] * 0.25
            - coordinates[2]
        ).astype(np.float32)[native_order]

        layout = infer_dense_lattice_layout(
            ordered_ids,
            ordered_positions,
            side,
            0.8,
        )
        dense, _, _ = layout.encode_field(
            velocity,
            "vx",
            position_residual=False,
        )
        restored = lattice_layout_from_metadata(
            ordered_ids,
            layout.manifest_metadata(),
        ).decode_field(
            dense,
            "vx",
            IDENTITY_TRANSFORM,
            np.dtype("float32"),
        )

        np.testing.assert_array_equal(restored, velocity)

    def test_rejects_a_periodic_box_below_the_occupancy_threshold(self):
        side, ids, positions, _ = self._periodic_slab(2)
        with self.assertRaisesRegex(
            LatticeLayoutUnavailable,
            "occupancy",
        ):
            infer_dense_lattice_layout(ids, positions, side, 0.8)

    def test_complete_lattice_uses_implicit_geometry(self):
        side = 12
        ids = np.arange(side**3, dtype=np.uint64)
        high = ids // (side * side)
        middle = (ids // side) % side
        low = ids % side
        positions = {
            "x": ((low + 0.125) / side).astype(np.float32),
            "y": ((high + 0.25) / side).astype(np.float32),
            "z": ((middle + 0.375) / side).astype(np.float32),
        }

        layout = infer_complete_lattice_layout(
            ids,
            positions,
            side,
            0,
            side**3 - 1,
        )

        self.assertTrue(layout.implicit_full_lattice)
        self.assertEqual(layout.shape, (side, side, side))
        self.assertEqual(layout.position_digit_axes, (2, 0, 1))
        self.assertEqual(layout.dense_indices.size, 0)
        dense, _, wraps = layout.encode_field(
            positions["x"],
            "x",
            position_residual=True,
        )
        restored = layout.decode_field(
            dense,
            "x",
            POSITION_RESIDUAL_TRANSFORM,
            np.dtype("float32"),
            wraps,
        )
        np.testing.assert_allclose(
            restored,
            positions["x"],
            atol=1e-7,
            rtol=0,
        )

        from_metadata = lattice_layout_from_metadata(
            None,
            layout.manifest_metadata(),
        )
        self.assertTrue(from_metadata.implicit_full_lattice)


if __name__ == "__main__":
    unittest.main()
