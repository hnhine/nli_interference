from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section61_data import RHO_CONTROLS  # noqa: E402
from interference_suite.section61_metrics import (  # noqa: E402
    behavioral_summary,
    das_summary,
    main_table_rows,
)


class Section61MetricTests(unittest.TestCase):
    def test_behavioral_gate_uses_worst_cell_not_overall_average(self):
        rows = []
        for cell, correct in {"++": 1, "+-": 1, "-+": 1, "--": 0}.items():
            for repeat in range(10):
                is_correct = correct if cell == "--" else 1
                rows.append(
                    {
                        "form": "failed_to",
                        "cell": cell,
                        "is_correct": is_correct,
                        "R": 1.0 if cell in {"++", "--"} else -1.0,
                        "expected_R_sign": 1 if cell in {"++", "--"} else -1,
                        "rho": 1 if cell in {"++", "--"} else -1,
                        "U_gap": -1.0,
                        "pred_label": "T" if is_correct else "F",
                    }
                )
        summary = behavioral_summary(rows, model_name="model", gate_threshold=0.9)
        form = summary["by_form"][0]
        self.assertEqual(form["accuracy"], 0.75)
        self.assertEqual(form["min_cell_accuracy"], 0.0)
        self.assertFalse(form["eligible"])

    def test_rho_confirmatory_score_is_minimum_over_six_controls(self):
        rows = []
        for index, control in enumerate(RHO_CONTROLS):
            rows.append(
                {
                    "control_type": control,
                    "is_counterfactual_correct": int(index != 3),
                    "R": 1.0,
                    "U_gap": -1.0,
                    "section61_experiment": "E2",
                    "target_var": "rho",
                    "form": "never",
                    "direction": "",
                    "base_form": "never",
                    "source_form": "never",
                    "base_site": "claim_final",
                }
            )
        summary = das_summary(
            rows,
            model_name="Qwen/Qwen3-8B",
            rotation={"layer": 18, "site": "claim_final", "rank": 16},
        )
        self.assertTrue(summary["rho_full_audit_complete"])
        self.assertEqual(summary["rho_full_audit_min_IIA"], 0.0)
        self.assertGreater(summary["IIA"], 0.0)

    def test_main_table_worst_uses_only_gate_eligible_forms_and_flags_coverage(self):
        behavior = [
            {
                "model_name": "Qwen/Qwen3-8B",
                "by_form": [
                    {"form": "did_not", "min_cell_accuracy": 1.0, "eligible": True},
                    {"form": "didnt", "min_cell_accuracy": 0.8, "eligible": False},
                    {"form": "never", "min_cell_accuracy": 0.95, "eligible": True},
                ],
            }
        ]
        das = [
            {
                "model_name": "Qwen/Qwen3-8B",
                "section61_experiment": "E2",
                "target_var": "rho",
                "form": form,
                "reported_site": "claim_final",
                "rho_full_audit_min_IIA": score,
            }
            for form, score in (("did_not", 0.99), ("didnt", 0.1), ("never", 0.91))
        ]
        table = main_table_rows(behavior, das)
        worst = table[-1]
        self.assertEqual(worst["qwen_within"], 0.91)
        self.assertEqual(worst["qwen_n_eligible"], 2)
        self.assertFalse(worst["qwen_eligible"])


if __name__ == "__main__":
    unittest.main()
