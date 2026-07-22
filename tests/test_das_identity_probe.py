from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import generate_das_pairs  # noqa: E402
from run_das_identity_probe_sweep import annotate_hypotheses, summarize_identity  # noqa: E402


class DasIdentityProbeTests(unittest.TestCase):
    def setUp(self):
        rows = generate_das_pairs(n_base_events=3, seed=0, targets=["pi"], pi_variant="v5")
        self.rows = [annotate_hypotheses(row) for row in rows]
        mapping = {"main": "flip_pi", "probe_flip_both": "flip_both", "probe_flip_pc": "flip_pc"}
        for row in self.rows:
            row["identity_condition"] = mapping.get(row["control_type"], row["control_type"])

    def test_pi_probe_hypothesis_patterns(self):
        by = {
            name: next(row for row in self.rows if row["identity_condition"] == name)
            for name in ("flip_pi", "flip_pc", "flip_both")
        }
        self.assertEqual(by["flip_pi"]["pred_H_pi"], by["flip_pi"]["pred_H_rel"])
        self.assertEqual(by["flip_pi"]["pred_H_pc"], by["flip_pi"]["base_label"])
        self.assertEqual(by["flip_pc"]["pred_H_pc"], by["flip_pc"]["pred_H_rel"])
        self.assertEqual(by["flip_pc"]["pred_H_pi"], by["flip_pc"]["base_label"])
        self.assertEqual(by["flip_both"]["pred_H_pi"], by["flip_both"]["pred_H_pc"])
        self.assertEqual(by["flip_both"]["pred_H_rel"], by["flip_both"]["base_label"])

    def test_each_hypothesis_is_recoverable(self):
        selected = [
            row for row in self.rows
            if row["identity_condition"] in {
                "flip_pi", "flip_pc", "flip_both", "gate_m0", "label_copy_trap"
            }
        ]
        for hypothesis in ("pi", "pc", "rel"):
            scored = [
                {**row, "pred_label": row[f"pred_H_{hypothesis}"], "global_top_in_TFU": 1}
                for row in selected
            ]
            summary = summarize_identity(scored, "pi")
            self.assertEqual(summary["identity_winner"], hypothesis)
            self.assertEqual(summary["macro"][hypothesis], 1.0)


if __name__ == "__main__":
    unittest.main()
