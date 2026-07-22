from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import generate_das_pairs, high_level_label  # noqa: E402
from interference_suite.das_pyvene import resolved_train_control_types, validate_control_proportions  # noqa: E402
from run_das_relay_map import RHO_DEFAULT_CONTROL_PROPORTIONS, resolve_control_proportions  # noqa: E402


class RhoGenerationTests(unittest.TestCase):
    def test_rho_emits_six_balanced_conditions_at_claim_final(self):
        n_base_events = 4
        rows = generate_das_pairs(
            n_base_events=n_base_events,
            seed=0,
            targets=["rho"],
        )
        expected_per_control = n_base_events * 3 * 2 * 2
        self.assertEqual(
            Counter(row["control_type"] for row in rows),
            Counter({
                "flip_pi": expected_per_control,
                "flip_pc": expected_per_control,
                "hold_both": expected_per_control,
                "source_m0": expected_per_control,
                "gate_m0": expected_per_control,
                "label_copy_trap": expected_per_control,
            }),
        )
        self.assertEqual({row["target_var"] for row in rows}, {"rho"})
        self.assertEqual({row["run_family"] for row in rows}, {"das_atomic_rho"})
        self.assertEqual({row["base_site"] for row in rows}, {"claim_final"})
        self.assertEqual({row["source_site"] for row in rows}, {"claim_final"})

    def test_each_condition_obeys_the_pre_gate_rho_rule(self):
        rows = generate_das_pairs(n_base_events=3, seed=1, targets=["rho"])
        for row in rows:
            rho_base = int(row["p_i_base"]) * int(row["p_c_base"])
            rho_src = int(row["p_i_src"]) * int(row["p_c_src"])
            self.assertEqual(row["rho_base"], rho_base)
            self.assertEqual(row["rho_src"], rho_src)

            control = row["control_type"]
            if control in {"flip_pi", "flip_pc", "source_m0"}:
                self.assertEqual(rho_src, -rho_base)
                self.assertEqual(
                    row["target_label"],
                    high_level_label(row["m_base"], row["p_i_src"], row["p_c_src"]),
                )
            elif control == "hold_both":
                self.assertEqual(rho_src, rho_base)
                self.assertNotEqual(row["p_i_src"], row["p_i_base"])
                self.assertNotEqual(row["p_c_src"], row["p_c_base"])
                self.assertEqual(row["target_label"], row["base_label"])
            else:
                self.assertEqual(row["m_base"], 0)
                self.assertEqual(row["target_label"], "U")

    def test_off_diagonal_controls_separate_rho_from_source_label(self):
        rows = generate_das_pairs(n_base_events=2, seed=2, targets=["rho"])
        source_m0 = [row for row in rows if row["control_type"] == "source_m0"]
        traps = [row for row in rows if row["control_type"] == "label_copy_trap"]
        self.assertTrue(source_m0)
        self.assertTrue(traps)

        for row in source_m0:
            self.assertEqual((row["m_base"], row["m_src"]), (1, 0))
            self.assertEqual(row["source_label"], "U")
            self.assertIn(row["target_label"], {"T", "F"})
            self.assertNotEqual(row["target_label"], row["source_label"])

        for row in traps:
            self.assertEqual((row["m_base"], row["m_src"]), (0, 1))
            self.assertIn(row["source_label"], {"T", "F"})
            self.assertEqual(row["target_label"], "U")
            self.assertNotEqual(row["target_label"], row["source_label"])

    def test_default_training_uses_all_six_controls_at_20_20_20_20_10_10(self):
        rows = generate_das_pairs(n_base_events=2, seed=3, targets=["rho"])
        controls = resolved_train_control_types("rho", ["auto"])
        self.assertIn("source_m0", controls)
        self.assertEqual(
            controls,
            ["flip_pi", "flip_pc", "hold_both", "source_m0", "gate_m0", "label_copy_trap"],
        )

        proportions = resolve_control_proportions(
            rows=rows,
            target_var="rho",
            train_control_types=["auto"],
            values=None,
        )
        normalized = validate_control_proportions(rows, proportions)
        active = sum(normalized[name] for name in ("flip_pi", "flip_pc", "hold_both", "source_m0"))
        inactive = sum(normalized[name] for name in ("gate_m0", "label_copy_trap"))
        self.assertAlmostEqual(active, 0.8)
        self.assertAlmostEqual(inactive, 0.2)
        self.assertEqual(set(proportions), set(RHO_DEFAULT_CONTROL_PROPORTIONS))
        self.assertEqual(normalized["flip_pi"], 0.2)
        self.assertEqual(normalized["flip_pc"], 0.2)
        self.assertEqual(normalized["hold_both"], 0.2)
        self.assertEqual(normalized["source_m0"], 0.2)
        self.assertEqual(normalized["gate_m0"], 0.1)
        self.assertEqual(normalized["label_copy_trap"], 0.1)

    def test_legacy_default_generation_does_not_add_rho(self):
        rows = generate_das_pairs(n_base_events=1, seed=4)
        self.assertNotIn("rho", {row["target_var"] for row in rows})


if __name__ == "__main__":
    unittest.main()
