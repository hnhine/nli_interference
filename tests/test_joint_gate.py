from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.joint_gate_data import (  # noqa: E402
    JOINT_GATE_CELLS,
    generate_joint_gate_rows,
)
from interference_suite.joint_gate_intervention import (  # noqa: E402
    constrained_patch,
    sequential_patch,
)


class JointGateGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_joint_gate_rows(n_base_events=2, seed=11)

    def test_six_balanced_cells_and_two_rho_source_regimes(self):
        expected_per_cell = 2 * 3 * 2 * 2
        self.assertEqual(
            Counter(row["cell_type"] for row in self.rows),
            Counter({cell.name: expected_per_cell for cell in JOINT_GATE_CELLS}),
        )
        self.assertEqual(Counter(int(row["rho_source_m"]) for row in self.rows), {0: 72, 1: 72})

    def test_open_cells_require_opposite_rho_and_true_synergy(self):
        open_rows = [row for row in self.rows if row["cell_family"] == "open_synergy"]
        self.assertTrue(open_rows)
        for row in open_rows:
            self.assertEqual(int(row["rho_donor"]), -int(row["rho_base"]))
            self.assertNotEqual(row["expected_m_only"], row["expected_joint"])
            self.assertNotEqual(row["expected_rho_only"], row["expected_joint"])
            if int(row["rho_source_m"]) == 0:
                self.assertEqual(int(row["strict_assembly"]), 1)

    def test_same_value_donors_hold_the_target_variables(self):
        for row in self.rows:
            self.assertEqual(int(row["m_same_source_m"]), int(row["m_base"]))
            self.assertEqual(int(row["rho_same_source_rho"]), int(row["rho_base"]))
            self.assertEqual(row["expected_same_value"], row["expected_none"])


class JointGateProjectionTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - repo runtime includes torch
            self.skipTest(str(exc))
        self.torch = torch

    def test_constrained_patch_sets_overlapping_coordinates(self):
        torch = self.torch
        u_m = torch.tensor([[1.0], [0.0], [0.0]])
        u_rho = torch.tensor([[1 / math.sqrt(2)], [1 / math.sqrt(2)], [0.0]])
        h = torch.tensor([[0.2, -0.4, 0.7], [-0.5, 0.3, -0.1]])
        z_m = torch.tensor([[1.2], [-0.7]])
        z_rho = torch.tensor([[-0.4], [0.9]])
        patched, diagnostics = constrained_patch(torch, h, [u_m, u_rho], [z_m, z_rho])
        torch.testing.assert_close(patched @ u_m, z_m, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(patched @ u_rho, z_rho, atol=1e-5, rtol=1e-5)
        self.assertLess(diagnostics["coordinate_residual_max"], 1e-4)

    def test_orthogonal_case_reduces_to_sum_of_independent_projections(self):
        torch = self.torch
        u_m = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        u_rho = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
        h = torch.tensor([[0.2, -0.4, 0.7, 1.1]])
        z_m = torch.tensor([[1.2]])
        z_rho = torch.tensor([[-0.9]])
        patched, _ = constrained_patch(torch, h, [u_m, u_rho], [z_m, z_rho])
        expected = h + (z_m - h @ u_m) @ u_m.T + (z_rho - h @ u_rho) @ u_rho.T
        torch.testing.assert_close(patched, expected, atol=1e-6, rtol=1e-6)

    def test_naive_sequential_patch_is_order_dependent_under_overlap(self):
        torch = self.torch
        u_m = torch.tensor([[1.0], [0.0]])
        u_rho = torch.tensor([[0.8], [0.6]])
        h = torch.zeros((1, 2))
        z_m = torch.tensor([[1.0]])
        z_rho = torch.tensor([[-1.0]])
        m_then_rho, _ = sequential_patch(torch, h, [u_m, u_rho], [z_m, z_rho])
        rho_then_m, _ = sequential_patch(torch, h, [u_rho, u_m], [z_rho, z_m])
        self.assertFalse(torch.allclose(m_then_rho, rho_then_m))


if __name__ == "__main__":
    unittest.main()
