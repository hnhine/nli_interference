from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import generate_das_pairs  # noqa: E402


class MV4GenerationTests(unittest.TestCase):
    def test_explicit_legacy_is_identical_to_default(self):
        default_rows = generate_das_pairs(n_base_events=4, seed=7, targets=["m"])
        legacy_rows = generate_das_pairs(n_base_events=4, seed=7, targets=["m"], m_variant="legacy")
        self.assertEqual(default_rows, legacy_rows)

    def test_same_match_label_copy_trap(self):
        n_base_events = 4
        rows = generate_das_pairs(
            n_base_events=n_base_events,
            seed=0,
            targets=["m"],
            m_variant="v4",
        )
        traps = [row for row in rows if row["control_type"] == "label_copy_trap_same_m1"]
        self.assertEqual(len(traps), n_base_events * 3 * 2 * 2)
        self.assertEqual({row["m_variant"] for row in rows}, {"v4"})
        self.assertEqual({row["run_family"] for row in rows}, {"das_atomic_m_v4"})
        for row in traps:
            self.assertEqual((row["m_base"], row["m_src"]), (1, 1))
            self.assertEqual(row["p_i_src"], -row["p_i_base"])
            self.assertEqual(row["p_c_src"], row["p_c_base"])
            self.assertNotEqual(row["source_label"], row["base_label"])
            self.assertEqual(row["target_label"], row["base_label"])
            self.assertEqual(row["m_label_copy_trap_type"], "same_m1")

    def test_validation_and_test_are_balanced_across_controls(self):
        rows = generate_das_pairs(
            n_base_events=10,
            seed=0,
            targets=["m"],
            m_variant="v4",
        )
        controls = {
            "nomatch_to_match",
            "match_to_nomatch",
            "label_copy_trap",
            "label_copy_trap_same_m1",
        }
        for split in ("val", "test"):
            counts = Counter(
                row["control_type"]
                for row in rows
                if row["split"] == split
            )
            self.assertEqual(set(counts), controls)
            self.assertEqual(len(set(counts.values())), 1)

    def test_v4_requires_m_only(self):
        with self.assertRaisesRegex(ValueError, "requires --targets m only"):
            generate_das_pairs(n_base_events=2, targets=["pi", "m"], m_variant="v4")


if __name__ == "__main__":
    unittest.main()
