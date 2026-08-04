from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section62_ablation import (  # noqa: E402
    AnalysisMembership,
    FrozenSiteProfile,
)
from run_section62_ablation import (  # noqa: E402
    _selected_profiles,
    attach_base_metrics,
    build_opposite_rho_rows,
    build_runtime_rows,
    donor_coordinates_for_rows,
    enrich_intervention_records,
    random_accuracy_spread,
    stable_generator_seed,
    summarize_opposite_rho,
    verify_recomputed_base,
)


def metrics(
    *,
    prediction: str = "T",
    is_correct: int = 1,
    m_tf: float | str = 2.0,
    g_u: float = -1.0,
) -> dict[str, object]:
    return {
        "logit_T": 3.0,
        "logit_F": 1.0,
        "logit_U": 0.0,
        "R": 2.0,
        "M_TF": m_tf,
        "G_U": g_u,
        "gold_margin_3": 1.5,
        "U_gap": -3.0,
        "pred_label": prediction,
        "pred_tf_label": "T",
        "is_correct": is_correct,
        "is_tf_correct": 1,
        "global_top_token_id": 51,
        "global_top_token": "T",
        "global_top_in_TFU": 1,
    }


def profile() -> FrozenSiteProfile:
    return FrozenSiteProfile(
        model_name="example/model",
        model_key="example",
        reported_site="claim_final",
        layer=3,
        rank=2,
        hidden_size=4,
        rho_rotation_dir="rho",
        m_rotation_dir="m",
        rho_metadata_site="claim_final",
        m_metadata_site="claim_final",
    )


class SeedAndMembershipTests(unittest.TestCase):
    def test_rotation_override_is_single_site_and_rank_matched(self):
        selected = _selected_profiles(
            "Qwen/Qwen3-8B",
            ["claim_final"],
            rho_rotation_dir="rho/seed1/L18_claim_final",
            m_rotation_dir="m/seed1/L18_claim_final",
            rotation_rank=64,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].layer, 18)
        self.assertEqual(selected[0].rank, 64)
        self.assertEqual(selected[0].rho_metadata_site, "claim_final")
        self.assertEqual(selected[0].m_metadata_site, "claim_final")

        with self.assertRaisesRegex(ValueError, "exactly one"):
            _selected_profiles(
                "Qwen/Qwen3-8B",
                ["claim_final", "answer_token"],
                rho_rotation_dir="rho",
                m_rotation_dir="m",
                rotation_rank=64,
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            _selected_profiles(
                "Qwen/Qwen3-8B",
                ["claim_final"],
                rho_rotation_dir="rho",
                rotation_rank=64,
            )

    def test_opposite_rho_rows_are_same_triple_and_retargeted(self):
        labels = {"++": "T", "-+": "F", "+-": "F", "--": "T"}
        rhos = {"++": 1, "-+": -1, "+-": -1, "--": 1}
        rows = [
            {
                "corpus": "MNLI",
                "analysis_stratum": "square_valid",
                "triple_id": "x",
                "cell": cell,
                "sample_id": f"x:{cell}",
                "expected_label": labels[cell],
                "rho_base": rhos[cell],
            }
            for cell in ("++", "-+", "+-", "--")
        ]

        retargeted = build_opposite_rho_rows(rows)
        by_cell = {row["cell"]: row for row in retargeted}
        self.assertEqual(by_cell["++"]["portability_donor_cell"], "-+")
        self.assertEqual(by_cell["--"]["portability_donor_cell"], "+-")
        self.assertEqual(by_cell["-+"]["portability_donor_cell"], "++")
        self.assertEqual(by_cell["+-"]["portability_donor_cell"], "--")
        for row in retargeted:
            self.assertEqual(row["rho_base"], -row["opposite_base_rho"])
            self.assertNotEqual(
                row["expected_label"], row["opposite_base_expected_label"]
            )

        scored = []
        for row in retargeted:
            target = row["expected_label"]
            scored.append(
                {
                    **row,
                    "condition": "rho_opposite_cross_cell",
                    "model_name": "example/model",
                    "site_alias": "claim_final",
                    "layer": 3,
                    "pred_label": target,
                    "is_correct": 1,
                    "is_tf_correct": 1,
                    "M_TF": 2.0,
                    "R": float(row["rho_base"]) * 2.0,
                    "base_is_correct": 1,
                }
            )
        summary = summarize_opposite_rho(scored)
        self.assertEqual(summary["overall"]["IIA"], 1.0)
        self.assertEqual(summary["whole_square"]["whole_square_IIA"], 1.0)
        self.assertEqual(len(summary["by_cell"]), 4)

    def test_generator_seed_is_stable_and_namespaced(self):
        first = stable_generator_seed(
            model_name="example/model",
            site="claim_final",
            layer=3,
            logical_seed=0,
        )
        self.assertEqual(
            first,
            stable_generator_seed(
                model_name="example/model",
                site="claim_final",
                layer=3,
                logical_seed=0,
            ),
        )
        self.assertNotEqual(
            first,
            stable_generator_seed(
                model_name="example/model",
                site="answer_token",
                layer=3,
                logical_seed=0,
            ),
        )

    def test_runtime_rows_preserve_r1_scores_under_prefix(self):
        r1 = {
            "sample_id": "a",
            "condition": "base",
            "site_alias": "",
            "layer": "",
            "subspace_var": "",
            "random_seed": "",
            **metrics(),
        }
        member = {
            "sample_id": "a",
            "row_eligible": 1,
            "pair_eligible": 1,
            "baseline_pred_label": "T",
        }
        membership = AnalysisMembership(rows=(member,), summary={})

        rows = build_runtime_rows([r1], membership)

        self.assertEqual(rows[0]["r1_pred_label"], "T")
        self.assertEqual(rows[0]["r1_is_correct"], 1)
        self.assertNotIn("pred_label", rows[0])
        self.assertEqual(rows[0]["row_eligible"], 1)
        self.assertEqual(rows[0]["r1_condition"], "base")


class BaselineAndRecordTests(unittest.TestCase):
    def row(self, *, stratum: str = "square_valid") -> dict[str, object]:
        return {
            "sample_id": "a",
            "analysis_stratum": stratum,
            "portability_donor_sample_id": "b",
            "r1_pred_label": "T" if stratum == "square_valid" else "U",
            "r1_logit_T": 3.0,
            "r1_logit_F": 1.0,
            "r1_logit_U": 0.0,
        }

    def test_recomputed_base_requires_prediction_reproduction(self):
        row = self.row()
        audit = verify_recomputed_base([row], [metrics()])
        self.assertEqual(audit["n_prediction_mismatches"], 0)
        self.assertEqual(audit["max_abs_logit_drift"]["T"], 0.0)

        bad = metrics(prediction="F", is_correct=0)
        with self.assertRaisesRegex(RuntimeError, "baseline disagrees"):
            verify_recomputed_base([row], [bad])

    def test_attaches_base_and_condition_metrics_without_overwriting_r1(self):
        row = self.row()
        base_records, by_id = attach_base_metrics([row], [metrics()])
        self.assertEqual(base_records[0]["condition"], "base")
        self.assertEqual(base_records[0]["base_pred_label"], "T")

        changed = metrics(prediction="F", is_correct=0, m_tf=-0.5)
        records = enrich_intervention_records(
            [row],
            [changed],
            base_metrics_by_id=by_id,
            profile=profile(),
            condition="rho_zero",
            subspace_var="rho",
            intervention_kind="ordinary_zero_projection",
            basis_sha256="abc",
        )
        record = records[0]
        self.assertEqual(record["base_is_correct"], 1)
        self.assertEqual(record["is_correct"], 0)
        self.assertEqual(record["accuracy_drop"], 1)
        self.assertEqual(record["M_TF_drop"], 2.5)
        self.assertEqual(record["G_U_drop"], "")
        self.assertEqual(record["coordinate_source_sample_id"], "")

    def test_all_u_uses_g_u_delta_and_portability_names_donor(self):
        row = self.row(stratum="all_u")
        base = metrics(prediction="U", m_tf="", g_u=2.0)
        base["pred_tf_label"] = "T"
        base["is_tf_correct"] = ""
        base_records, by_id = attach_base_metrics([row], [base])
        self.assertEqual(base_records[0]["base_G_U"], 2.0)
        changed = dict(base)
        changed["G_U"] = 0.5
        records = enrich_intervention_records(
            [row],
            [changed],
            base_metrics_by_id=by_id,
            profile=profile(),
            condition="rho_same_cross_cell",
            subspace_var="rho",
            intervention_kind="ordinary_coordinate_replacement",
            basis_sha256="abc",
        )
        self.assertEqual(records[0]["M_TF_drop"], "")
        self.assertEqual(records[0]["G_U_drop"], 1.5)
        self.assertEqual(records[0]["coordinate_source_sample_id"], "b")


class DonorAndSpreadTests(unittest.TestCase):
    def test_reorders_coordinates_by_reciprocal_donor_id(self):
        rows = [
            {"sample_id": "a", "portability_donor_sample_id": "b"},
            {"sample_id": "b", "portability_donor_sample_id": "a"},
        ]
        coordinates = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        donors = donor_coordinates_for_rows(
            torch=torch,
            rows=rows,
            coordinates=coordinates,
        )

        self.assertTrue(
            torch.equal(
                donors,
                torch.tensor([[3.0, 4.0], [1.0, 2.0]]),
            )
        )

    def test_random_spread_is_computed_per_site_stratum_and_cell(self):
        rows = []
        for seed, values in ((0, (1, 1)), (1, (1, 0)), (2, (0, 0))):
            for index, value in enumerate(values):
                rows.append(
                    {
                        "sample_id": f"{seed}-{index}",
                        "condition": "rand_zero",
                        "site_alias": "claim_final",
                        "analysis_stratum": "square_valid",
                        "cell": "++",
                        "random_seed": seed,
                        "is_correct": value,
                    }
                )

        maximum, detail = random_accuracy_spread(rows)

        self.assertEqual(maximum, 1.0)
        self.assertEqual(len(detail), 1)
        self.assertEqual(detail[0]["accuracy_by_seed"], [1.0, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
