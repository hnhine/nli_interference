from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section62_intervention import (  # noqa: E402
    basis_overlap_metrics,
    effective_site,
    load_rotation_basis,
    next_token_metrics,
)


class DecodeTokenizer:
    def decode(self, token_ids):
        return f"token-{token_ids[0]}"


def rotation_metadata(**updates):
    metadata = {
        "target_var": "rho",
        "model_name": "test/model",
        "layer": 2,
        "rank": 2,
        "component": "block_output",
        "site": "row",
    }
    metadata.update(updates)
    return metadata


def rotation_expectation(**updates):
    expected = {
        "target_var": "rho",
        "model_name": "test/model",
        "layer": 2,
        "rank": 2,
        "component": "block_output",
        "allowed_metadata_sites": ["row", "claim_final"],
    }
    expected.update(updates)
    return expected


class Section62InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover - project runtime includes both
            raise unittest.SkipTest(str(exc)) from exc
        cls.np = np
        cls.torch = torch

    def write_rotation(self, root: Path, matrix, *, metadata=None) -> None:
        resolved_metadata = (
            rotation_metadata() if metadata is None else metadata
        )
        (root / "rotation_weight_metadata.json").write_text(
            json.dumps(resolved_metadata),
            encoding="utf-8",
        )
        self.np.save(
            root / "rotation_weight.npy",
            self.np.asarray(matrix, dtype=self.np.float32),
        )

    def test_effective_site_maps_row_metadata_to_each_rows_base_site(self):
        self.assertEqual(
            effective_site("row", {"base_site": "claim_final"}),
            "claim_final",
        )
        self.assertEqual(
            effective_site("row", {"base_site": "answer_token"}),
            "answer_token",
        )
        self.assertEqual(effective_site("claim_final", {}), "claim_final")

        with self.assertRaisesRegex(ValueError, "requires base_site"):
            effective_site("row", {})
        with self.assertRaisesRegex(ValueError, "requires base_site"):
            effective_site("row", {"base_site": ""})

    def test_next_token_metrics_match_r_mtf_and_logsumexp_u_gap(self):
        torch = self.torch
        rows = [
            {
                "expected_label": "F",
                "rho_base": -1,
                "analysis_stratum": "square_valid",
            },
            {
                "expected_label": "U",
                "rho_base": 1,
                "analysis_stratum": "all_u",
            },
        ]
        # Token ids 0/1/2 are T/F/U. Row 0 deliberately has an unrelated
        # vocabulary token as its global argmax while its TFU prediction is F.
        next_logits = torch.tensor(
            [
                [1.0, 3.0, 0.5, 5.0],
                [1.0, 0.0, 2.0, -1.0],
            ],
            dtype=torch.float32,
        )

        metrics = next_token_metrics(
            torch=torch,
            next_logits=next_logits,
            rows=rows,
            label_token_ids={"T": 0, "F": 1, "U": 2},
            tokenizer=DecodeTokenizer(),
        )

        self.assertEqual(metrics[0]["R"], -2.0)
        self.assertEqual(metrics[0]["M_TF"], 2.0)
        self.assertAlmostEqual(
            metrics[0]["G_U"],
            float(0.5 - torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0)),
            places=6,
        )
        self.assertEqual(metrics[0]["pred_label"], "F")
        self.assertEqual(metrics[0]["is_correct"], 1)
        self.assertEqual(metrics[0]["global_top_token_id"], 3)
        self.assertEqual(metrics[0]["global_top_token"], "token-3")
        self.assertEqual(metrics[0]["global_top_in_TFU"], 0)

        self.assertEqual(metrics[1]["R"], 1.0)
        self.assertEqual(metrics[1]["M_TF"], "")
        self.assertAlmostEqual(
            metrics[1]["G_U"],
            float(2.0 - torch.logsumexp(torch.tensor([1.0, 0.0]), dim=0)),
            places=6,
        )
        self.assertEqual(metrics[1]["pred_label"], "U")
        self.assertEqual(metrics[1]["is_tf_correct"], "")
        self.assertEqual(metrics[1]["global_top_in_TFU"], 1)

    def test_load_rotation_basis_strictly_validates_and_qr_orthonormalizes(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_rotation(
                root,
                [
                    [2.0, 0.0],
                    [0.0, 3.0],
                    [0.0, 0.0],
                ],
            )

            basis, provenance = load_rotation_basis(
                rotation_dir=root,
                expected=rotation_expectation(),
                hidden_size=3,
                torch=torch,
                device=torch.device("cpu"),
            )

            self.assertEqual(tuple(basis.shape), (3, 2))
            torch.testing.assert_close(
                basis.T @ basis,
                torch.eye(2),
                atol=1e-6,
                rtol=1e-6,
            )
            self.assertEqual(provenance["shape"], [3, 2])
            self.assertEqual(provenance["dtype"], "float32")
            self.assertEqual(provenance["matrix_rank"], 2)
            self.assertGreater(provenance["raw_gram_max_abs_error"], 0.0)
            self.assertLess(provenance["qr_gram_max_abs_error"], 1e-6)
            self.assertEqual(len(provenance["metadata_sha256"]), 64)
            self.assertEqual(len(provenance["weight_sha256"]), 64)
            self.assertEqual(provenance["metadata"]["site"], "row")

    def test_load_rotation_basis_rejects_metadata_and_shape_mismatches(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_rank = [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
            self.write_rotation(
                root,
                full_rank,
                metadata=rotation_metadata(target_var="m"),
            )
            with self.assertRaisesRegex(ValueError, "metadata target_var"):
                load_rotation_basis(
                    rotation_dir=root,
                    expected=rotation_expectation(),
                    hidden_size=3,
                    torch=torch,
                    device=torch.device("cpu"),
                )

            self.write_rotation(root, [[1.0, 0.0], [0.0, 1.0]])
            with self.assertRaisesRegex(ValueError, "rotation shape"):
                load_rotation_basis(
                    rotation_dir=root,
                    expected=rotation_expectation(),
                    hidden_size=3,
                    torch=torch,
                    device=torch.device("cpu"),
                )

            self.write_rotation(
                root,
                [
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            )
            with self.assertRaisesRegex(ValueError, "rotation rank=1"):
                load_rotation_basis(
                    rotation_dir=root,
                    expected=rotation_expectation(),
                    hidden_size=3,
                    torch=torch,
                    device=torch.device("cpu"),
                )

    def test_basis_overlap_reports_identical_and_orthogonal_subspaces(self):
        torch = self.torch
        rho = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        identical = basis_overlap_metrics(torch, rho, rho.clone())
        self.assertEqual(identical["hidden_size"], 4)
        self.assertEqual(identical["rank"], 2)
        self.assertAlmostEqual(identical["normalized_frobenius_overlap"], 1.0)
        self.assertAlmostEqual(
            identical["random_baseline_rank_over_hidden"],
            0.5,
        )
        self.assertAlmostEqual(identical["overlap_to_random_ratio"], 2.0)
        self.assertAlmostEqual(identical["max_principal_cosine"], 1.0)
        self.assertEqual(identical["n_principal_cosine_gt_0_7"], 2)

        orthogonal = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        disjoint = basis_overlap_metrics(torch, rho, orthogonal)
        self.assertAlmostEqual(disjoint["normalized_frobenius_overlap"], 0.0)
        self.assertAlmostEqual(disjoint["overlap_to_random_ratio"], 0.0)
        self.assertAlmostEqual(disjoint["max_principal_cosine"], 0.0)
        self.assertEqual(disjoint["n_principal_cosine_gt_0_7"], 0)

        with self.assertRaisesRegex(ValueError, "identical shape"):
            basis_overlap_metrics(torch, rho, orthogonal[:, :1])


if __name__ == "__main__":
    unittest.main()
