from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.joint_gate_rescue import (  # noqa: E402
    RESCUE_CONDITIONS,
    condition_by_name,
    expected_label,
    state_after_condition,
)


class JointGateRescueSemanticsTests(unittest.TestCase):
    def test_condition_names_are_unique(self):
        names = [condition.name for condition in RESCUE_CONDITIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_single_knockout_and_matched_rescue_restore_base_state(self):
        for m_base in (0, 1):
            for rho_base in (-1, 1):
                self.assertEqual(
                    state_after_condition(condition_by_name("claim_m_flip"), m_base, rho_base),
                    (1 - m_base, rho_base),
                )
                self.assertEqual(
                    state_after_condition(
                        condition_by_name("claim_m_flip_answer_m_restore"), m_base, rho_base
                    ),
                    (m_base, rho_base),
                )
                self.assertEqual(
                    state_after_condition(condition_by_name("claim_rho_flip"), m_base, rho_base),
                    (m_base, -rho_base),
                )
                self.assertEqual(
                    state_after_condition(
                        condition_by_name("claim_rho_flip_answer_rho_restore"), m_base, rho_base
                    ),
                    (m_base, rho_base),
                )

    def test_selective_rescue_has_gate_specific_predictions(self):
        for rho_base in (-1, 1):
            restore_m = condition_by_name("claim_both_flip_answer_m_restore")
            restore_rho = condition_by_name("claim_both_flip_answer_rho_restore")
            restore_both = condition_by_name("claim_both_flip_answer_both_restore")

            self.assertEqual(state_after_condition(restore_m, 0, rho_base), (0, -rho_base))
            self.assertEqual(expected_label(restore_m, 0, rho_base), "U")
            self.assertEqual(state_after_condition(restore_rho, 0, rho_base), (1, rho_base))

            self.assertEqual(state_after_condition(restore_m, 1, rho_base), (1, -rho_base))
            self.assertEqual(state_after_condition(restore_rho, 1, rho_base), (0, rho_base))
            self.assertEqual(state_after_condition(restore_both, 1, rho_base), (1, rho_base))

    def test_random_restore_retains_double_corruption_target(self):
        condition = condition_by_name("claim_both_flip_answer_random_restore")
        for m_base in (0, 1):
            for rho_base in (-1, 1):
                self.assertEqual(
                    state_after_condition(condition, m_base, rho_base),
                    (1 - m_base, -rho_base),
                )


if __name__ == "__main__":
    unittest.main()
