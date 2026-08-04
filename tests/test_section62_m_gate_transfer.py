from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from run_section62_m_gate_transfer import (  # noqa: E402
    CELL_TARGET,
    RANDOM_SEEDS,
    build_gate_transfer_design,
    summarize_gate_transfer,
)


class GateTransferDesignTests(unittest.TestCase):
    def test_full_state_filtered_pairing_is_same_cell_one_to_one(self):
        rows = []
        base = {}
        for cell in ("++", "-+", "+-", "--"):
            for index in range(3):
                sample_id = f"s:{cell}:{index}"
                rows.append(
                    {
                        "corpus": "MNLI",
                        "analysis_stratum": "square_valid",
                        "triple_id": f"square_{cell}_{index}",
                        "sample_id": sample_id,
                        "cell": cell,
                        "expected_label": CELL_TARGET[cell],
                        "rho_base": 1 if cell in {"++", "--"} else -1,
                    }
                )
                base[sample_id] = {
                    "pred_label": "U" if index == 2 else CELL_TARGET[cell]
                }
            for index in range(2):
                sample_id = f"u:{cell}:{index}"
                rows.append(
                    {
                        "corpus": "MNLI",
                        "analysis_stratum": "all_u",
                        "triple_id": f"all_u_{cell}_{index}",
                        "sample_id": sample_id,
                        "cell": cell,
                        "expected_label": "U",
                        "rho_base": 1 if cell in {"++", "--"} else -1,
                    }
                )
                base[sample_id] = {"pred_label": "U"}

        transfer, same, manifest = build_gate_transfer_design(rows, base)

        self.assertEqual(manifest["n_pairs"], 8)
        self.assertEqual(len(transfer), 16)
        self.assertEqual(len(same), 16)
        self.assertFalse(manifest["length_matching"])
        self.assertEqual(
            [row["sample_id"] for row in transfer],
            [row["sample_id"] for row in same],
        )
        self.assertEqual(len({row["sample_id"] for row in transfer}), 16)
        for row in transfer:
            self.assertEqual(row["cell"], row["portability_donor_cell"])
            if row["gate_direction"] == "match_to_no_overlap":
                self.assertNotEqual(row["gate_base_prediction"], "U")
                self.assertEqual(row["gate_target_label"], "U")
            else:
                self.assertEqual(row["gate_base_prediction"], "U")
                self.assertEqual(row["gate_target_label"], CELL_TARGET[row["cell"]])

    def test_summary_keeps_controls_and_random_seeds_separate(self):
        rows = []
        for cell in ("++", "-+", "+-", "--"):
            for direction in ("match_to_no_overlap", "no_overlap_to_match"):
                target = "U" if direction == "match_to_no_overlap" else CELL_TARGET[cell]
                for condition, success in (
                    ("m_cross_stratum", 1),
                    ("rho_cross_stratum", 0),
                ):
                    rows.append(self.record(condition, direction, cell, target, success))
                for seed in RANDOM_SEEDS:
                    rows.append(
                        self.record(
                            "rand_cross_stratum",
                            direction,
                            cell,
                            target,
                            0,
                            random_seed=seed,
                        )
                    )
            rows.append(self.record("m_same_stratum", "same_match", cell, "T", 1))
            rows.append(self.record("m_same_stratum", "same_no_overlap", cell, "U", 1))

        summary = summarize_gate_transfer(rows)

        self.assertEqual(len(summary["confirmatory"]), 2)
        self.assertEqual(summary["headline"]["target_IIA_direction_min"], 1.0)
        self.assertEqual(summary["headline"]["NEx_IIA_vs_rho_direction_min"], 1.0)
        self.assertEqual(summary["headline"]["NEx_IIA_vs_rand_direction_min"], 1.0)
        random_groups = [
            row for row in summary["by_group"]
            if row["condition"] == "rand_cross_stratum"
        ]
        self.assertEqual(len(random_groups), 24)
        self.assertEqual({row["random_seed"] for row in random_groups}, {0, 1, 2})

    @staticmethod
    def record(condition, direction, cell, target, success, random_seed=""):
        prediction = target if success else ("T" if target == "U" else "U")
        return {
            "condition": condition,
            "gate_direction": direction,
            "cell": cell,
            "random_seed": random_seed,
            "pred_label": prediction,
            "is_correct": success,
            "base_target_correct": 0,
            "gate_success": success,
            "is_tf_correct": (success if target in {"T", "F"} else ""),
            "target_TF_margin": (1.0 if target in {"T", "F"} else ""),
            "U_gap": 1.0 if prediction == "U" else -1.0,
            "U_gap_delta": 1.0 if prediction == "U" else -1.0,
        }


if __name__ == "__main__":
    unittest.main()
