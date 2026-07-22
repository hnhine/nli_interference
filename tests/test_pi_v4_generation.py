from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import generate_das_pairs, high_level_label  # noqa: E402
from interference_suite.das_pyvene import summarize_scored  # noqa: E402


class PiV4GenerationTests(unittest.TestCase):
    def test_explicit_legacy_is_identical_to_default(self):
        default_rows = generate_das_pairs(n_base_events=4, seed=7, targets=["pi"])
        legacy_rows = generate_das_pairs(n_base_events=4, seed=7, targets=["pi"], pi_variant="legacy")
        self.assertEqual(default_rows, legacy_rows)

    def test_v4_has_complete_source_gate_coverage(self):
        n_base_events = 4
        rows = generate_das_pairs(
            n_base_events=n_base_events,
            seed=0,
            targets=["pi"],
            pi_variant="v4",
        )
        expected_per_control = n_base_events * 3 * 2 * 2
        self.assertEqual(
            Counter(row["control_type"] for row in rows),
            Counter({
                "main": expected_per_control,
                "active_source_m0": expected_per_control,
                "gate_m0": expected_per_control,
                "label_copy_trap": expected_per_control,
                "distractor": expected_per_control,
            }),
        )
        self.assertEqual({row["pi_variant"] for row in rows}, {"v4"})
        self.assertEqual({row["run_family"] for row in rows}, {"das_atomic_pi_v4"})

        causal_rows = [row for row in rows if row["control_type"] != "distractor"]
        self.assertEqual(
            {(row["m_base"], row["m_src"]) for row in causal_rows},
            {(1, 1), (1, 0), (0, 0), (0, 1)},
        )
        for row in causal_rows:
            self.assertEqual(
                row["target_label"],
                high_level_label(row["m_base"], row["p_i_src"], row["p_c_base"]),
            )

    def test_v4_locality_rows_keep_base_output(self):
        rows = generate_das_pairs(n_base_events=2, seed=1, targets=["pi"], pi_variant="v4")
        locality_rows = [row for row in rows if row["pi_regime"] == "locality"]
        self.assertTrue(locality_rows)
        for row in locality_rows:
            self.assertEqual(row["control_type"], "distractor")
            self.assertEqual(row["target_label"], row["base_label"])

    def test_regime_macro_weights_active_inactive_and_locality_equally(self):
        controls = [
            ("main", "active", 1.0),
            ("active_source_m0", "active", 0.0),
            ("gate_m0", "inactive", 1.0),
            ("label_copy_trap", "inactive", 1.0),
            ("distractor", "locality", 0.0),
        ]
        rows = [
            {
                "control_type": control,
                "pi_regime": regime,
                "is_correct": correct,
                "pred_label": "T",
                "R": 0.0,
                "U_gap": 0.0,
                "global_top_in_TFU": 1.0,
            }
            for control, regime, correct in controls
        ]
        summary = summarize_scored(rows)
        self.assertAlmostEqual(summary["IIA"], 0.6)
        self.assertAlmostEqual(summary["by_pi_regime"]["active"]["IIA"], 0.5)
        self.assertAlmostEqual(summary["by_pi_regime"]["inactive"]["IIA"], 1.0)
        self.assertAlmostEqual(summary["by_pi_regime"]["locality"]["IIA"], 0.0)
        self.assertAlmostEqual(summary["pi_regime_macro_IIA"], 0.5)

    def test_v4_requires_pi_only(self):
        with self.assertRaisesRegex(ValueError, "requires --targets pi only"):
            generate_das_pairs(n_base_events=2, targets=["pc", "pi"], pi_variant="v4")


if __name__ == "__main__":
    unittest.main()
