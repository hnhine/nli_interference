from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.das_data import (  # noqa: E402
    M_INDEPENDENT_VERBS,
    allowed_independent_mismatch_verbs,
    generate_das_pairs,
)
from run_das_ablation import (  # noqa: E402
    build_donor_indices,
    resolve_condition_hold_columns,
    resolve_donor_event_policy,
    resolve_donor_pool,
)
from run_das_match_transfer import select_rows, source_as_base, validate_reverse_pairs  # noqa: E402


class MDonorTests(unittest.TestCase):
    def synthetic_rows(self):
        rows = []
        for event in ("e0", "e1", "e2"):
            for control, m in (("match_to_nomatch", "1"), ("nomatch_to_match", "0"), ("label_copy_trap", "0")):
                rows.append({
                    "sample_id": f"{event}_{control}",
                    "base_event_id": event,
                    "control_type": control,
                    "m_base": m,
                    "p_i_base": "1",
                    "p_c_base": "-1",
                    "mismatch_type": "object",
                    "matched_idx": "0",
                })
        return rows

    def test_auto_m_configuration(self):
        self.assertEqual(resolve_condition_hold_columns("m", "auto"), [
            "p_i_base", "p_c_base", "mismatch_type", "matched_idx"
        ])
        self.assertEqual(resolve_donor_pool("m", "auto"), "m_main")
        self.assertEqual(resolve_donor_event_policy("m", "auto"), "require")

    def test_m_cross_control_donor_holds_and_excludes_trap(self):
        rows = self.synthetic_rows()
        indices = build_donor_indices(
            rows,
            "m_base",
            0,
            condition_hold=["p_i_base", "p_c_base", "mismatch_type", "matched_idx"],
            donor_pool="m_main",
            donor_event_policy="require",
        )
        for index, donor_index in enumerate(indices["opposite"]):
            donor = rows[donor_index]
            row = rows[index]
            self.assertNotEqual(donor["m_base"], row["m_base"])
            self.assertIn(donor["control_type"], ("match_to_nomatch", "nomatch_to_match"))
            self.assertNotEqual(donor["base_event_id"], row["base_event_id"])
            for column in ("p_i_base", "p_c_base", "mismatch_type", "matched_idx"):
                self.assertEqual(donor[column], row[column])

    def test_reverse_pairs_and_source_promotion(self):
        rows = generate_das_pairs(n_base_events=4, seed=0, targets=["m"])
        validate_reverse_pairs(rows)
        selected = select_rows(rows, "all", 0, include_relaxed=False)
        source = source_as_base(selected[0])
        self.assertEqual(source["base_prompt"], selected[0]["source_prompt"])
        self.assertEqual(source["base_label"], selected[0]["source_label"])
        self.assertEqual(source["m_base"], selected[0]["m_src"])

    def test_independent_v1_uses_cross_group_verb_pairs(self):
        physical = {"visit", "reach", "explore", "enter"}
        discourse = {"like", "mention", "recommend", "describe"}
        original_objects = {"open", "create", "clean", "collect", "close"}
        control_objects = {"inspect", "paint", "move"}
        allowed_group_pairs = {
            (frozenset(physical), frozenset(discourse)),
            (frozenset(original_objects), frozenset(control_objects)),
        }
        for claim_verb in M_INDEPENDENT_VERBS:
            candidates = allowed_independent_mismatch_verbs(claim_verb)
            self.assertTrue(candidates)
            for mismatch_verb in candidates:
                self.assertEqual(claim_verb.arg_type, mismatch_verb.arg_type)
                pair_groups = next(
                    (groups for groups in allowed_group_pairs if claim_verb.base in groups[0] | groups[1]),
                    None,
                )
                self.assertIsNotNone(pair_groups)
                left, right = pair_groups
                self.assertTrue(
                    (claim_verb.base in left and mismatch_verb.base in right)
                    or (claim_verb.base in right and mismatch_verb.base in left)
                )

        rows = generate_das_pairs(
            n_base_events=20,
            seed=0,
            targets=["m"],
            m_verb_policy="independent_v1",
        )
        self.assertTrue(rows)
        self.assertEqual({row["m_verb_policy"] for row in rows}, {"independent_v1"})


if __name__ == "__main__":
    unittest.main()
