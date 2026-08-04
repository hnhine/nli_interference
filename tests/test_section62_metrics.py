from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section62_data import FULL_CELL_ORDER  # noqa: E402
from interference_suite.section62_metrics import (  # noqa: E402
    behavioral_gate_summary,
)


SQUARE_LABELS = {"++": "T", "-+": "F", "+-": "F", "--": "T"}


def behavioral_row(
    *,
    triple_id: str,
    stratum: str,
    cell: str,
    pred_label: str | None = None,
    include_global_top: bool = True,
) -> dict[str, object]:
    expected_label = SQUARE_LABELS[cell] if stratum == "square_valid" else "U"
    prediction = expected_label if pred_label is None else pred_label
    if expected_label == "T":
        r_value = 2.0 if prediction == "T" else -2.0
    elif expected_label == "F":
        r_value = -2.0 if prediction == "F" else 2.0
    else:
        r_value = 0.25
    row: dict[str, object] = {
        "corpus": "MNLI",
        "triple_id": triple_id,
        "analysis_stratum": stratum,
        "cell": cell,
        "expected_label": expected_label,
        "pred_label": prediction,
        "is_correct": int(prediction == expected_label),
        "R": r_value,
        "U_gap": 1.0 if expected_label == "U" else -1.0,
        "gold_margin_3": 1.5 if prediction == expected_label else -1.5,
        "join_override_used": 0,
    }
    if stratum == "square_valid":
        predicted_tf = "T" if r_value >= 0 else "F"
        row["is_tf_correct"] = int(predicted_tf == expected_label)
        row["M_TF"] = r_value if expected_label == "T" else -r_value
    if include_global_top:
        row["global_top_in_TFU"] = 1
    return row


def square_rows(
    triple_id: str,
    *,
    stratum: str = "square_valid",
    wrong_cell: str | None = None,
    include_global_top: bool = True,
) -> list[dict[str, object]]:
    rows = []
    for cell in FULL_CELL_ORDER:
        wrong_prediction = None
        if cell == wrong_cell:
            expected = SQUARE_LABELS[cell]
            wrong_prediction = "F" if expected == "T" else "T"
        rows.append(
            behavioral_row(
                triple_id=triple_id,
                stratum=stratum,
                cell=cell,
                pred_label=wrong_prediction,
                include_global_top=include_global_top,
            )
        )
    return rows


class Section62MetricTests(unittest.TestCase):
    def test_square_gate_uses_worst_cell_and_keeps_secondary_scope_separate(self):
        rows = []
        for index in range(10):
            rows.extend(
                square_rows(
                    f"mnli_{index:04d}",
                    wrong_cell="--" if index < 2 else None,
                )
            )

        summary = behavioral_gate_summary(
            rows,
            model_name="model",
            gate_threshold=0.90,
            min_baseline_correct_per_cell=8,
        )
        gate = summary["square_gate"][0]

        self.assertAlmostEqual(gate["accuracy"], 0.95)
        self.assertAlmostEqual(gate["min_cell_accuracy"], 0.80)
        self.assertAlmostEqual(gate["min_cell_tf_accuracy"], 0.80)
        self.assertAlmostEqual(gate["whole_square_accuracy"], 0.80)
        self.assertFalse(gate["strict_gate_eligible"])
        self.assertTrue(gate["secondary_baseline_correct_eligible"])
        self.assertEqual(gate["interpretation_scope"], "baseline_correct_secondary_only")
        self.assertEqual(
            [record["cell"] for record in summary["by_cell"]],
            list(FULL_CELL_ORDER),
        )

    def test_all_u_gate_is_separate_and_has_no_vacuous_tf_success(self):
        rows = square_rows(
            "mnli_0100",
            stratum="all_u",
            include_global_top=False,
        )
        summary = behavioral_gate_summary(
            rows,
            model_name="model",
            min_baseline_correct_per_cell=1,
        )

        self.assertEqual(summary["square_gate"], [])
        self.assertEqual(len(summary["all_u_gate"]), 1)
        gate = summary["all_u_gate"][0]
        self.assertTrue(gate["strict_gate_eligible"])
        self.assertEqual(gate["min_cell_accuracy"], 1.0)
        self.assertIsNone(gate["min_cell_tf_accuracy"])
        self.assertIsNone(summary["by_square"][0]["all_tf_correct"])
        self.assertTrue(
            all(record["tf_accuracy"] is None for record in summary["by_cell"])
        )
        self.assertTrue(
            all(
                record["global_top_in_TFU_rate"] is None
                for record in summary["by_cell"]
            )
        )

    def test_rejects_duplicate_or_incomplete_square_cells(self):
        rows = square_rows("mnli_0200")
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate cells"):
            behavioral_gate_summary(rows, model_name="model")

        with self.assertRaisesRegex(ValueError, "expected="):
            behavioral_gate_summary(
                square_rows("mnli_0201")[:-1],
                model_name="model",
            )

    def test_rejects_inconsistent_signature_and_correctness_fields(self):
        wrong_signature = square_rows("mnli_0300")
        wrong_signature[0]["expected_label"] = "F"
        wrong_signature[0]["pred_label"] = "F"
        with self.assertRaisesRegex(ValueError, "requires expected_label='T'"):
            behavioral_gate_summary(wrong_signature, model_name="model")

        wrong_correctness = square_rows("mnli_0301")
        wrong_correctness[0]["is_correct"] = 0
        with self.assertRaisesRegex(ValueError, "but pred_label='T'"):
            behavioral_gate_summary(wrong_correctness, model_name="model")

        missing_tf_metric = square_rows("mnli_0302")
        del missing_tf_metric[0]["is_tf_correct"]
        with self.assertRaisesRegex(ValueError, "is_tf_correct"):
            behavioral_gate_summary(missing_tf_metric, model_name="model")


if __name__ == "__main__":
    unittest.main()
