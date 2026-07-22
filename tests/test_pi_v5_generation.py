from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import generate_das_pairs, high_level_label  # noqa: E402


class PiV5GenerationTests(unittest.TestCase):
    def test_v5_adds_balanced_raw_pi_identity_probes(self):
        n_base_events = 4
        rows = generate_das_pairs(
            n_base_events=n_base_events,
            seed=0,
            targets=["pi"],
            pi_variant="v5",
        )
        expected_per_control = n_base_events * 3 * 2 * 2
        self.assertEqual(
            Counter(row["control_type"] for row in rows),
            Counter({
                "main": expected_per_control,
                "active_source_m0": expected_per_control,
                "probe_flip_both": expected_per_control,
                "probe_flip_pc": expected_per_control,
                "gate_m0": expected_per_control,
                "label_copy_trap": expected_per_control,
                "distractor": expected_per_control,
            }),
        )
        self.assertEqual({row["pi_variant"] for row in rows}, {"v5"})
        self.assertEqual({row["run_family"] for row in rows}, {"das_atomic_pi_v5"})

    def test_v5_probes_separate_raw_pi_from_rel(self):
        rows = generate_das_pairs(n_base_events=3, seed=1, targets=["pi"], pi_variant="v5")
        probes = [row for row in rows if row["control_type"].startswith("probe_")]
        self.assertTrue(probes)
        for row in probes:
            self.assertEqual(
                row["target_label"],
                high_level_label(row["m_base"], row["p_i_src"], row["p_c_base"]),
            )
            rel_base = row["p_i_base"] * row["p_c_base"]
            rel_source = row["p_i_src"] * row["p_c_src"]
            if row["control_type"] == "probe_flip_both":
                self.assertNotEqual(row["p_i_base"], row["p_i_src"])
                self.assertEqual(rel_base, rel_source)
                self.assertNotEqual(row["target_label"], row["base_label"])
            else:
                self.assertEqual(row["p_i_base"], row["p_i_src"])
                self.assertNotEqual(rel_base, rel_source)
                self.assertEqual(row["target_label"], row["base_label"])


if __name__ == "__main__":
    unittest.main()
