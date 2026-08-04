from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_pyvene import (  # noqa: E402
    filter_train_rows,
    resolved_train_control_types,
    validate_control_proportions,
)
from run_das_relay_map import (  # noqa: E402
    RHO_DEFAULT_CONTROL_PROPORTIONS,
    RHO_V2_DEFAULT_CONTROL_PROPORTIONS,
    cell_record,
    resolve_control_proportions,
)


LEGACY_CONTROLS = tuple(RHO_DEFAULT_CONTROL_PROPORTIONS)
V2_CONTROLS = tuple(RHO_V2_DEFAULT_CONTROL_PROPORTIONS)


def rows_for(controls: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"target_var": "rho", "split": "train", "control_type": control}
        for control in controls
    ]


class RhoV2TrainingTests(unittest.TestCase):
    def test_v2_default_mix_is_detected_before_legacy(self):
        rows = rows_for(V2_CONTROLS)
        proportions = resolve_control_proportions(
            rows=rows,
            target_var="rho",
            train_control_types=["auto"],
            values=None,
        )
        self.assertEqual(proportions, RHO_V2_DEFAULT_CONTROL_PROPORTIONS)
        self.assertEqual(validate_control_proportions(rows, proportions), {
            "flip_pi": 0.16,
            "flip_pc": 0.16,
            "hold_both": 0.16,
            "source_m0": 0.16,
            "source_m0_same": 0.16,
            "gate_m0": 0.10,
            "label_copy_trap": 0.10,
        })

    def test_legacy_rho_default_is_unchanged(self):
        rows = rows_for(LEGACY_CONTROLS)
        proportions = resolve_control_proportions(
            rows=rows,
            target_var="rho",
            train_control_types=["auto"],
            values=None,
        )
        self.assertEqual(proportions, RHO_DEFAULT_CONTROL_PROPORTIONS)
        self.assertNotIn("source_m0_same", proportions)

    def test_auto_training_filter_includes_v2_control_only_when_available(self):
        v2_rows = rows_for(V2_CONTROLS)
        self.assertIn(
            "source_m0_same",
            resolved_train_control_types(
                "rho",
                ["auto"],
                available_controls=set(V2_CONTROLS),
            ),
        )
        self.assertEqual(
            {row["control_type"] for row in filter_train_rows(v2_rows, "rho", ["auto"])},
            set(V2_CONTROLS),
        )

        legacy_rows = rows_for(LEGACY_CONTROLS)
        self.assertNotIn(
            "source_m0_same",
            resolved_train_control_types(
                "rho",
                ["auto"],
                available_controls=set(LEGACY_CONTROLS),
            ),
        )
        self.assertEqual(
            {row["control_type"] for row in filter_train_rows(legacy_rows, "rho", ["auto"])},
            set(LEGACY_CONTROLS),
        )

    def test_relay_metrics_require_new_control_only_for_v2(self):
        legacy_by_control = {
            name: {"IIA": 0.9, "n": 10}
            for name in LEGACY_CONTROLS
        }
        legacy = cell_record(
            12,
            "claim_final",
            {"test": {"by_control": legacy_by_control}},
        )
        self.assertIsNone(legacy["source_m0_same_IIA"])
        self.assertAlmostEqual(legacy["rho_identification_min_IIA"], 0.9)
        self.assertAlmostEqual(legacy["rho_full_audit_min_IIA"], 0.9)

        v2_by_control = dict(legacy_by_control)
        v2_by_control["source_m0_same"] = {"IIA": 0.2, "n": 10}
        v2 = cell_record(
            12,
            "claim_final",
            {"test": {"by_control": v2_by_control}},
        )
        self.assertAlmostEqual(v2["source_m0_same_IIA"], 0.2)
        self.assertAlmostEqual(v2["rho_active_IIA"], 0.76)
        self.assertAlmostEqual(v2["rho_identification_min_IIA"], 0.2)
        self.assertAlmostEqual(v2["rho_full_audit_min_IIA"], 0.2)


if __name__ == "__main__":
    unittest.main()
