from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.generation import generate_exp3, sample_base_events  # noqa: E402
from interference_suite.metrics import summarize_exp3  # noqa: E402


class Exp3MinimalPairTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        claim = sample_base_events(1, rng)[0]
        self.rows = generate_exp3("base_0000", claim, rng)

    @staticmethod
    def source_identity(row, idx):
        return (
            row[f"source{idx}_subject"],
            row[f"source{idx}_verb_base"],
            row[f"source{idx}_object"],
        )

    @staticmethod
    def polarities(row):
        return tuple(row[f"source{idx}_polarity"] for idx in (1, 2, 3))

    def test_factorial_design_has_exact_target_and_distractor_pairs(self):
        self.assertEqual(len(self.rows), 18)
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 18)
        self.assertEqual({row["exp3_design"] for row in self.rows}, {"minimal_pairs_v2"})

        for match_idx in (1, 2, 3):
            cell = [row for row in self.rows if row["match_idx"] == match_idx]
            self.assertEqual(len(cell), 6)
            identities = {
                tuple(self.source_identity(row, idx) for idx in (1, 2, 3))
                for row in cell
            }
            self.assertEqual(len(identities), 1)

            configs = {row["exp3_distractor_config"] for row in cell}
            distractor_positions = {idx for idx in (1, 2, 3) if idx != match_idx}
            self.assertEqual(configs, {"anchor", *(f"flip_d{idx}" for idx in distractor_positions)})

            for config in configs:
                pair = [row for row in cell if row["exp3_distractor_config"] == config]
                self.assertEqual(len(pair), 2)
                positive = next(row for row in pair if row["match_polarity"] == "positive")
                negative = next(row for row in pair if row["match_polarity"] == "negative")
                changed = {
                    idx
                    for idx, (left, right) in enumerate(
                        zip(self.polarities(positive), self.polarities(negative)), start=1
                    )
                    if left != right
                }
                self.assertEqual(changed, {match_idx})

            for target_polarity in ("positive", "negative"):
                target_rows = [row for row in cell if row["match_polarity"] == target_polarity]
                anchor = next(row for row in target_rows if row["exp3_distractor_config"] == "anchor")
                for flipped in (row for row in target_rows if row["exp3_distractor_config"] != "anchor"):
                    changed = {
                        idx
                        for idx, (left, right) in enumerate(
                            zip(self.polarities(anchor), self.polarities(flipped)), start=1
                        )
                        if left != right
                    }
                    self.assertEqual(changed, {int(flipped["exp3_flipped_distractor_idx"])})

    def test_summary_reports_target_and_distractor_effects(self):
        scored = []
        for row in self.rows:
            out = dict(row)
            target_sign = 1 if row["match_polarity"] == "positive" else -1
            distractor_signs = [
                1 if row[f"source{idx}_polarity"] == "positive" else -1
                for idx in (1, 2, 3)
                if idx != row["match_idx"]
            ]
            out["R"] = 10.0 * target_sign + 0.2 * sum(distractor_signs)
            out["U_gap"] = -1.0
            out["pred_label"] = "T" if target_sign > 0 else "F"
            out["is_correct"] = 1
            scored.append(out)

        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize_exp3(pd.DataFrame(scored), Path(tmp))
            pair_csv = Path(tmp) / "exp3_intervention_pairs.csv"
            self.assertTrue(pair_csv.exists())

        minimal = summary["minimal_pairs"]
        self.assertEqual(minimal["target_flip"]["n_pairs"], 9)
        self.assertAlmostEqual(minimal["target_flip"]["mean_abs_delta_R"], 20.0)
        self.assertEqual(minimal["distractor_flip"]["n_pairs"], 12)
        self.assertAlmostEqual(minimal["distractor_flip"]["mean_abs_delta_R"], 0.4)
        self.assertAlmostEqual(minimal["target_to_distractor_mean_abs_effect_ratio"], 50.0)
        self.assertEqual(minimal["distractor_flip"]["prediction_invariance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
