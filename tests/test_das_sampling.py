from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_pyvene import (  # noqa: E402
    sample_training_batch,
    validate_control_proportions,
)
from run_das_relay_map import (  # noqa: E402
    PI_V4_DEFAULT_CONTROL_PROPORTIONS,
    PI_V5_DEFAULT_CONTROL_PROPORTIONS,
    parse_control_proportions,
    resolve_control_proportions,
    resolve_training_steps,
)


class StratifiedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"control_type": control, "row": index}
            for control in ("main", "distractor", "gate_m0", "label_copy_trap")
            for index in range(20)
        ]

    def test_exact_half_main_batch(self):
        proportions = validate_control_proportions(
            self.rows,
            {
                "main": 0.5,
                "distractor": 1 / 6,
                "gate_m0": 1 / 6,
                "label_copy_trap": 1 / 6,
            },
        )
        batch = sample_training_batch(
            self.rows,
            30,
            random.Random(0),
            control_proportions=proportions,
        )
        self.assertEqual(Counter(row["control_type"] for row in batch), {
            "main": 15,
            "distractor": 5,
            "gate_m0": 5,
            "label_copy_trap": 5,
        })

    def test_cli_parser_and_normalization(self):
        parsed = parse_control_proportions(["main=3", "gate_m0=1"])
        normalized = validate_control_proportions(self.rows, parsed)
        self.assertEqual(normalized, {"main": 0.75, "gate_m0": 0.25})

    def test_pi_v4_defaults_to_80_10_10_regime_mix(self):
        rows = [
            {
                "target_var": "pi",
                "pi_variant": "v4",
                "control_type": control,
            }
            for control in PI_V4_DEFAULT_CONTROL_PROPORTIONS
        ]
        proportions = resolve_control_proportions(
            rows=rows,
            target_var="pi",
            train_control_types=["auto"],
            values=None,
        )
        normalized = validate_control_proportions(rows, proportions)
        self.assertEqual(normalized, {
            "main": 0.4,
            "active_source_m0": 0.4,
            "gate_m0": 0.05,
            "label_copy_trap": 0.05,
            "distractor": 0.1,
        })

    def test_pi_v4_default_can_be_overridden(self):
        proportions = resolve_control_proportions(
            rows=[],
            target_var="pi",
            train_control_types=["all"],
            values=["main=3", "gate_m0=1"],
        )
        self.assertEqual(proportions, {"main": 3.0, "gate_m0": 1.0})

    def test_pi_v5_keeps_probe_weights_at_ten_percent(self):
        rows = [
            {"target_var": "pi", "pi_variant": "v5", "control_type": control}
            for control in PI_V5_DEFAULT_CONTROL_PROPORTIONS
        ]
        proportions = resolve_control_proportions(
            rows=rows,
            target_var="pi",
            train_control_types=["all"],
            values=None,
        )
        normalized = validate_control_proportions(rows, proportions)
        self.assertEqual(normalized, {
            "main": 0.30,
            "active_source_m0": 0.30,
            "probe_flip_both": 0.10,
            "probe_flip_pc": 0.10,
            "gate_m0": 0.05,
            "label_copy_trap": 0.05,
            "distractor": 0.10,
        })

    def test_legacy_pi_keeps_existing_default(self):
        rows = [{"target_var": "pi", "pi_variant": "legacy", "control_type": "main"}]
        self.assertIsNone(resolve_control_proportions(
            rows=rows,
            target_var="pi",
            train_control_types=["auto"],
            values=None,
        ))

    def test_missing_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent from the training pool"):
            validate_control_proportions(self.rows, {"not_a_control": 1.0})

    def test_epoch_equivalent_resolves_after_train_filtering(self):
        rows = [
            {"target_var": "pi", "split": "train", "control_type": control}
            for control in ("main", "distractor", "gate_m0", "label_copy_trap")
            for _ in range(15)
        ]
        self.assertEqual(
            resolve_training_steps(
                rows=rows,
                target_var="pi",
                train_control_types=["all"],
                include_relaxed=False,
                batch_size=16,
                steps=None,
                epochs=1,
            ),
            4,
        )

    def test_steps_default_is_preserved(self):
        self.assertEqual(
            resolve_training_steps(
                rows=[],
                target_var="pi",
                train_control_types=["all"],
                include_relaxed=False,
                batch_size=16,
                steps=None,
                epochs=None,
            ),
            500,
        )


if __name__ == "__main__":
    unittest.main()
