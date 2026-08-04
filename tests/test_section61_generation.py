from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.base import Event, VERBS  # noqa: E402
from interference_suite.das_data import generate_das_pairs  # noqa: E402
from interference_suite.negation_forms import (  # noqa: E402
    TIER1_FORM_KEYS,
    render_double_negation_family,
    render_polarity,
)
from interference_suite.section61_data import (  # noqa: E402
    RHO_CONTROLS,
    anchor_mismatches,
    derive_m_form,
    derive_rho_cross,
    derive_rho_mixed,
    derive_rho_within,
    generate_behavioral_rows,
)


class NegationFormTests(unittest.TestCase):
    def setUp(self):
        self.event = Event("Jack", VERBS[0], "PlaceA")

    def test_tier1_inventory_and_exact_negative_strings(self):
        self.assertEqual(len(TIER1_FORM_KEYS), 6)
        expected = {
            "did_not": "Jack did not visit PlaceA.",
            "didnt": "Jack didn't visit PlaceA.",
            "never": "Jack never visited PlaceA.",
            "did_not_ever": "Jack did not ever visit PlaceA.",
            "failed_to": "Jack failed to visit PlaceA.",
            "not_the_case": "It is not the case that Jack visited PlaceA.",
        }
        self.assertEqual(
            {form: render_polarity(self.event, -1, form) for form in TIER1_FORM_KEYS},
            expected,
        )

    def test_positive_is_identical_across_tier1_forms(self):
        values = {render_polarity(self.event, 1, form) for form in TIER1_FORM_KEYS}
        self.assertEqual(values, {"Jack visited PlaceA."})

    def test_double_negation_family_has_positive_and_negative_members(self):
        self.assertEqual(
            render_double_negation_family(self.event, 1),
            "Jack did not fail to visit PlaceA.",
        )
        self.assertEqual(
            render_double_negation_family(self.event, -1),
            "Jack failed to visit PlaceA.",
        )


class Section61GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = generate_das_pairs(n_base_events=4, seed=0, targets=["rho"])
        cls.rows = [
            row
            for row in rows
            if row["base_event_id"] == "base_0000"
            and row["matched_idx"] == 0
        ]

    def test_behavioral_crosses_forms_cells_and_events(self):
        events = [("a", self.event("Ava")), ("b", self.event("Mia"))]
        rows = generate_behavioral_rows(events)
        self.assertEqual(len(rows), 2 * 6 * 4)
        self.assertEqual({row["cell"] for row in rows}, {"++", "+-", "-+", "--"})
        self.assertEqual({row["n_assumptions"] for row in rows}, {1})

    def test_canonical_within_is_byte_identical(self):
        derived = derive_rho_within(self.rows, "did_not")
        self.assertEqual(anchor_mismatches(self.rows, derived), 0)

    def test_rho_within_preserves_all_controls_and_labels(self):
        derived = derive_rho_within(self.rows, "never")
        self.assertEqual(
            {row["control_type"] for row in derived},
            set(RHO_CONTROLS),
        )
        for before, after in zip(self.rows, derived):
            self.assertEqual(after["target_label"], before["target_label"])
            self.assertEqual(after["rho_base"], before["rho_base"])
            self.assertEqual(after["rho_src"], before["rho_src"])

    def test_cross_uses_new_base_and_canonical_source(self):
        source = next(
            row for row in self.rows
            if int(row["p_i_base"]) == -1 or int(row["p_c_base"]) == -1
        )
        row = derive_rho_cross([source], "never")[0]
        self.assertEqual((row["base_form"], row["source_form"]), ("never", "did_not"))
        self.assertIn(" never ", row["base_prompt"])
        self.assertNotIn(" never ", row["source_prompt"])

    def test_mixed_form_emits_both_named_directions(self):
        first = derive_rho_mixed(self.rows[:1], "never", "did_not")[0]
        second = derive_rho_mixed(self.rows[:1], "did_not", "never")[0]
        self.assertNotEqual(first["direction"], second["direction"])

    def test_m_control_emits_matched_canonical_and_never_arms(self):
        source = next(
            row for row in self.rows
            if int(row["p_i_base"]) == -1 or int(row["p_c_base"]) == -1
        )
        canonical = derive_m_form([source], "did_not")[0]
        never = derive_m_form([source], "never")[0]

        self.assertEqual(anchor_mismatches([source], [canonical]), 0)
        self.assertEqual(canonical["form"], "did_not")
        self.assertEqual(never["form"], "never")
        self.assertIn(" never ", never["base_prompt"])
        for derived in (canonical, never):
            self.assertEqual(derived["section61_experiment"], "E5")
            self.assertEqual(derived["target_label"], source["target_label"])
            self.assertEqual(derived["m_base"], source["m_base"])
            self.assertEqual(derived["m_src"], source["m_src"])
            self.assertEqual(derived["control_type"], source["control_type"])

    @staticmethod
    def event(subject: str) -> Event:
        return Event(subject, VERBS[0], "PlaceA")


if __name__ == "__main__":
    unittest.main()
