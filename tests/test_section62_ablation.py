from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.io_utils import write_rows_csv  # noqa: E402
from interference_suite.section62_ablation import (  # noqa: E402
    EXPECTED_R1_INPUT_ROWS,
    R1_RUN_TYPE,
    R1_SCHEMA_VERSION,
    R2_ANALYSIS_SCOPE,
    FrozenSiteProfile,
    Section62AblationContractError,
    build_analysis_membership,
    frozen_profiles_for_model,
    load_r1_handoff,
    preflight_frozen_profile,
    preflight_rotation_metadata,
)
from interference_suite.section62_data import (  # noqa: E402
    FULL_CELL_ORDER,
    RHO_SAME_DONOR,
    sha256_file,
)


LABELS = {
    "square_valid": {"++": "T", "-+": "F", "+-": "F", "--": "T"},
    "all_u": {cell: "U" for cell in FULL_CELL_ORDER},
}


def scored_square(
    *,
    model_name: str,
    stratum: str,
    triple_id: str,
    correct_cells: set[str],
    site: str = "claim_final",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in FULL_CELL_ORDER:
        expected = LABELS[stratum][cell]
        if cell in correct_cells:
            prediction = expected
        else:
            prediction = next(label for label in ("T", "F", "U") if label != expected)
        claim = f"Claim for {triple_id} {cell}."
        prompt = (
            f"Assumption: Premise for {triple_id} {cell}.\n\n"
            f"Claim: {claim}\n\n"
            "Choose exactly one:\nT = must be true\nF = must be false\n"
            "U = cannot be determined\n\nAnswer:\n"
        )
        claim_start = prompt.index(claim)
        answer_start = prompt.rindex("Answer:")
        rows.append(
            {
                "sample_id": f"{triple_id}_{cell}",
                "model_name": model_name,
                "condition": "base",
                "corpus": "MNLI",
                "triple_id": triple_id,
                "prompt": prompt,
                "analysis_stratum": stratum,
                "cell": cell,
                "expected_label": expected,
                "target_label": expected,
                "target_var": "rho",
                "base_label": expected,
                "base_prompt": prompt,
                "base_claim": claim,
                "base_site": site,
                "base_claim_span_start": claim_start,
                "base_claim_span_end": claim_start + len(claim),
                "base_answer_span_start": answer_start,
                "base_answer_span_end": answer_start + len("Answer:"),
                "pred_label": prediction,
                "is_correct": int(cell in correct_cells),
                "rho_base": 1 if cell in {"++", "--"} else -1,
                "portability_group": (
                    "negation_count_parity"
                    if cell in {"++", "--"}
                    else "negation_position"
                ),
                "join_override_used": 0,
            }
        )
    by_cell = {str(row["cell"]): row for row in rows}
    for row in rows:
        donor = by_cell[RHO_SAME_DONOR[str(row["cell"])]]
        row["portability_donor_cell"] = donor["cell"]
        row["portability_donor_sample_id"] = donor["sample_id"]
        row["source_label"] = donor["base_label"]
        row["source_prompt"] = donor["base_prompt"]
        row["source_claim"] = donor["base_claim"]
        row["source_site"] = donor["base_site"]
        row["source_claim_span_start"] = donor["base_claim_span_start"]
        row["source_claim_span_end"] = donor["base_claim_span_end"]
        row["source_answer_span_start"] = donor["base_answer_span_start"]
        row["source_answer_span_end"] = donor["base_answer_span_end"]
        row["rho_src"] = donor["rho_base"]
    return rows


def make_r1_fixture(
    root: Path,
    *,
    model_name: str,
    full_chain: bool = False,
) -> tuple[Path, Path]:
    rows = scored_square(
        model_name=model_name,
        stratum="square_valid",
        triple_id="mnli_square",
        correct_cells=set(FULL_CELL_ORDER),
    )
    rows.extend(
        scored_square(
            model_name=model_name,
            stratum="all_u",
            triple_id="mnli_all_u",
            correct_cells=set(FULL_CELL_ORDER),
        )
    )
    scored_path = root / "scored.csv"
    write_rows_csv(rows, scored_path)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["corpus"]),
                str(row["analysis_stratum"]),
                str(row["cell"]),
            )
        ].append(row)
    by_cell = [
        {
            "model_name": model_name,
            "corpus": corpus,
            "analysis_stratum": stratum,
            "cell": cell,
            "n": len(values),
            "n_correct": sum(int(row["is_correct"]) for row in values),
        }
        for (corpus, stratum, cell), values in grouped.items()
    ]
    by_stratum = [
        {
            "model_name": model_name,
            "corpus": "MNLI",
            "analysis_stratum": stratum,
            "strict_gate_eligible": False,
            "secondary_baseline_correct_eligible": True,
            "interpretation_scope": R2_ANALYSIS_SCOPE,
        }
        for stratum in ("square_valid", "all_u")
    ]
    summary = {
        "schema_version": R1_SCHEMA_VERSION,
        "run_type": R1_RUN_TYPE,
        "model_name": model_name,
        "n_rows": len(rows),
        "run": {"model_name": model_name},
        "artifacts": {
            "scored": {
                "path": scored_path.name,
                "rows": len(rows),
                "sha256": sha256_file(scored_path),
                "bytes": scored_path.stat().st_size,
            }
        },
        "by_cell": by_cell,
        "by_stratum": by_stratum,
    }
    if full_chain:
        r0_root = root / "r0"
        r0_root.mkdir()
        inputs: dict[str, dict[str, object]] = {}
        matched: dict[str, dict[str, object]] = {}
        outputs: dict[str, dict[str, object]] = {}
        input_roles = {
            "square_samples": "square_valid",
            "all_u_samples": "all_u",
        }
        for input_name, role in input_roles.items():
            path = r0_root / f"{role}.csv"
            n_rows = EXPECTED_R1_INPUT_ROWS[role]
            write_rows_csv(
                [{"fixture_row": index} for index in range(n_rows)],
                path,
            )
            relative_path = path.relative_to(root).as_posix()
            record = {
                "path": relative_path,
                "rows": n_rows,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            inputs[input_name] = dict(record)
            artifact_name = f"mnli_{role}"
            matched[role] = {
                **record,
                "artifact_name": artifact_name,
            }
            outputs[artifact_name] = {
                **record,
                "path": path.name,
            }
        r0_manifest = {
            "schema_version": 1,
            "outputs": outputs,
        }
        r0_manifest_path = r0_root / "manifest.json"
        r0_manifest_path.write_text(
            json.dumps(r0_manifest),
            encoding="utf-8",
        )
        summary["inputs"] = inputs
        summary["r0_manifest"] = {
            "path": r0_manifest_path.relative_to(root).as_posix(),
            "schema_version": 1,
            "sha256": sha256_file(r0_manifest_path),
            "bytes": r0_manifest_path.stat().st_size,
            "matched_artifacts": matched,
        }
    summary_path = root / "behavioral_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, scored_path


def write_rotation(
    root: Path,
    *,
    target_var: str,
    model_name: str,
    layer: int,
    rank: int,
    hidden_size: int,
    site: str,
    fill: float,
) -> Path:
    root.mkdir(parents=True)
    npy_path = root / "rotation_weight.npy"
    np.save(
        npy_path,
        np.full((hidden_size, rank), fill, dtype=np.float32),
    )
    pt_path = root / "rotation_weight.pt"
    pt_path.write_bytes(f"{target_var}-{fill}".encode("ascii"))
    metadata = {
        "target_var": target_var,
        "model_name": model_name,
        "layer": layer,
        "rank": rank,
        "component": "block_output",
        "site": site,
        "weights": {
            "weight": {
                "shape": [hidden_size, rank],
                "npy_path": str(npy_path),
                "pt_path": str(pt_path),
            }
        },
    }
    (root / "rotation_weight_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return root


class R1HandoffTests(unittest.TestCase):
    def test_loads_hash_verified_secondary_handoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, scored_path = make_r1_fixture(
                root,
                model_name="example/model",
            )

            handoff = load_r1_handoff(
                summary_path,
                expected_model_name="example/model",
            )

            self.assertEqual(handoff.model_name, "example/model")
            self.assertEqual(handoff.scored_path, scored_path)
            self.assertEqual(len(handoff.rows), 8)
            self.assertEqual(
                handoff.provenance["analysis_scope"],
                R2_ANALYSIS_SCOPE,
            )
            self.assertEqual(
                handoff.provenance["scored"]["sha256"],
                sha256_file(scored_path),
            )

    def test_rejects_tampered_or_substituted_scored_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, scored_path = make_r1_fixture(
                root,
                model_name="example/model",
            )
            substitute = root / "substitute.csv"
            substitute.write_text(scored_path.read_text(), encoding="utf-8")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "not the artifact pinned",
            ):
                load_r1_handoff(summary_path, scored_path=substitute)

            with scored_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "SHA256 mismatch",
            ):
                load_r1_handoff(summary_path)

    def test_rejects_wrong_model_or_nonsecondary_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, _ = make_r1_fixture(
                root,
                model_name="example/model",
            )
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "model mismatch",
            ):
                load_r1_handoff(
                    summary_path,
                    expected_model_name="different/model",
                )

            summary = json.loads(summary_path.read_text())
            summary["by_stratum"][0]["secondary_baseline_correct_eligible"] = False
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "lacks enough baseline-correct",
            ):
                load_r1_handoff(summary_path)

    def test_verifies_full_r0_input_chain_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, _ = make_r1_fixture(
                root,
                model_name="example/model",
                full_chain=True,
            )
            handoff = load_r1_handoff(
                summary_path,
                verify_r0_manifest=True,
                repo_root=root,
            )
            self.assertEqual(
                handoff.provenance["inputs"]["square_valid"]["rows"],
                620,
            )
            square_path = root / "r0" / "square_valid.csv"
            with square_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "R1 input square_samples SHA256 mismatch",
            ):
                load_r1_handoff(
                    summary_path,
                    verify_r0_manifest=True,
                    repo_root=root,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, _ = make_r1_fixture(
                root,
                model_name="example/model",
                full_chain=True,
            )
            summary = json.loads(summary_path.read_text())
            summary["r0_manifest"]["matched_artifacts"]["all_u"][
                "sha256"
            ] = "0" * 64
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "does not match R0 role all_u",
            ):
                load_r1_handoff(
                    summary_path,
                    verify_r0_manifest=True,
                    repo_root=root,
                )

    def test_rejects_unexpected_r1_input_row_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path, _ = make_r1_fixture(
                root,
                model_name="example/model",
                full_chain=True,
            )
            square_path = root / "r0" / "square_valid.csv"
            write_rows_csv(
                [{"fixture_row": index} for index in range(619)],
                square_path,
            )
            summary = json.loads(summary_path.read_text())
            replacement = {
                "rows": 619,
                "sha256": sha256_file(square_path),
                "bytes": square_path.stat().st_size,
            }
            summary["inputs"]["square_samples"].update(replacement)
            summary["r0_manifest"]["matched_artifacts"]["square_valid"].update(
                replacement
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "has 619 rows, expected 620",
            ):
                load_r1_handoff(
                    summary_path,
                    verify_r0_manifest=True,
                    repo_root=root,
                )


class AnalysisMembershipTests(unittest.TestCase):
    def rows(self) -> list[dict[str, object]]:
        model_name = "example/model"
        rows = scored_square(
            model_name=model_name,
            stratum="square_valid",
            triple_id="square_all_correct",
            correct_cells=set(FULL_CELL_ORDER),
        )
        rows.extend(
            scored_square(
                model_name=model_name,
                stratum="square_valid",
                triple_id="square_partial",
                correct_cells={"++", "-+", "+-"},
            )
        )
        rows.extend(
            scored_square(
                model_name=model_name,
                stratum="all_u",
                triple_id="all_u_all_correct",
                correct_cells=set(FULL_CELL_ORDER),
            )
        )
        rows.extend(
            scored_square(
                model_name=model_name,
                stratum="all_u",
                triple_id="all_u_partial",
                correct_cells={"-+", "--"},
            )
        )
        return rows

    def test_freezes_row_pair_and_whole_square_membership(self):
        membership = build_analysis_membership(
            self.rows(),
            min_baseline_correct_per_cell=1,
        )

        self.assertEqual(membership.summary["n_rows"], 16)
        self.assertEqual(membership.summary["n_row_eligible"], 13)
        strata = {
            row["analysis_stratum"]: row
            for row in membership.summary["by_stratum"]
        }
        self.assertEqual(
            strata["square_valid"]["n_whole_square_eligible"],
            1,
        )
        self.assertEqual(strata["all_u"]["n_whole_square_eligible"], 1)

        pairs = {
            (row["analysis_stratum"], row["portability_group"]): row
            for row in membership.summary["by_pair_group"]
        }
        self.assertEqual(
            pairs[("square_valid", "negation_count_parity")][
                "n_pair_eligible"
            ],
            1,
        )
        self.assertEqual(
            pairs[("square_valid", "negation_position")][
                "n_pair_eligible"
            ],
            2,
        )
        self.assertEqual(
            pairs[("all_u", "negation_count_parity")]["n_pair_eligible"],
            1,
        )
        self.assertEqual(
            pairs[("all_u", "negation_position")]["n_pair_eligible"],
            1,
        )

        by_id = {row["sample_id"]: row for row in membership.rows}
        eligible_base_bad_donor = by_id["square_partial_++"]
        self.assertEqual(eligible_base_bad_donor["row_eligible"], 1)
        self.assertEqual(eligible_base_bad_donor["pair_eligible"], 0)
        self.assertEqual(
            eligible_base_bad_donor["pair_exclusion_reason"],
            "donor_baseline_incorrect",
        )
        self.assertEqual(
            eligible_base_bad_donor["whole_square_eligible"],
            0,
        )

    def test_enforces_worst_cell_secondary_count(self):
        with self.assertRaisesRegex(
            Section62AblationContractError,
            "worst cell",
        ):
            build_analysis_membership(
                self.rows(),
                min_baseline_correct_per_cell=2,
            )

    def test_full_square_includes_every_main_row_and_excludes_all_u(self):
        membership = build_analysis_membership(
            self.rows(),
            min_baseline_correct_per_cell=2,
            analysis_population="full-square",
        )

        self.assertEqual(
            membership.summary["analysis_scope"],
            "full_square_mtf_primary",
        )
        self.assertEqual(membership.summary["n_row_eligible"], 8)
        by_id = {row["sample_id"]: row for row in membership.rows}
        self.assertEqual(by_id["square_partial_--"]["row_eligible"], 1)
        self.assertEqual(by_id["square_partial_--"]["pair_eligible"], 1)
        self.assertEqual(
            by_id["square_partial_--"]["whole_square_eligible"],
            1,
        )
        self.assertEqual(by_id["all_u_all_correct_++"]["row_eligible"], 0)
        self.assertEqual(
            by_id["all_u_all_correct_++"]["row_exclusion_reason"],
            "not_main_square",
        )

    def test_rejects_unknown_analysis_population(self):
        with self.assertRaisesRegex(
            Section62AblationContractError,
            "analysis_population",
        ):
            build_analysis_membership(
                self.rows(),
                min_baseline_correct_per_cell=1,
                analysis_population="unknown",
            )

    def test_rejects_broken_donor_or_behavioral_label(self):
        rows = self.rows()
        rows[0]["portability_donor_sample_id"] = rows[1]["sample_id"]
        with self.assertRaisesRegex(
            Section62AblationContractError,
            "donor",
        ):
            build_analysis_membership(
                rows,
                min_baseline_correct_per_cell=1,
            )

        rows = self.rows()
        rows[0]["is_correct"] = 0
        with self.assertRaisesRegex(
            Section62AblationContractError,
            "inconsistent is_correct",
        ):
            build_analysis_membership(
                rows,
                min_baseline_correct_per_cell=1,
            )

    def test_accepts_answer_token_site_contract(self):
        model_name = "example/model"
        rows = scored_square(
            model_name=model_name,
            stratum="square_valid",
            triple_id="square_answer",
            correct_cells=set(FULL_CELL_ORDER),
            site="answer_token",
        )
        rows.extend(
            scored_square(
                model_name=model_name,
                stratum="all_u",
                triple_id="all_u_answer",
                correct_cells=set(FULL_CELL_ORDER),
                site="answer_token",
            )
        )
        membership = build_analysis_membership(
            rows,
            min_baseline_correct_per_cell=1,
        )
        self.assertEqual(membership.summary["n_row_eligible"], 8)

    def test_rejects_prompt_span_site_and_materialization_drift(self):
        rows = self.rows()
        rows[0]["target_var"] = "m"
        with self.assertRaisesRegex(Section62AblationContractError, "target_var"):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["base_claim_span_start"] = (
            int(rows[0]["base_claim_span_start"]) + 1
        )
        with self.assertRaisesRegex(
            Section62AblationContractError, "claim span does not recover"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["base_answer_span_start"] = (
            int(rows[0]["base_answer_span_start"]) - 1
        )
        with self.assertRaisesRegex(
            Section62AblationContractError, "answer span does not recover"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["base_site"] = "row"
        with self.assertRaisesRegex(
            Section62AblationContractError, "expected one of"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        source_prompt = str(rows[0]["source_prompt"])
        rows[0]["source_prompt"] = "X" + source_prompt[1:]
        with self.assertRaisesRegex(
            Section62AblationContractError, "source_prompt does not match"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["source_label"] = "U"
        with self.assertRaisesRegex(
            Section62AblationContractError, "source_label does not match"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["source_site"] = "answer_token"
        with self.assertRaisesRegex(
            Section62AblationContractError, "source_site does not match"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["source_answer_span_end"] = (
            int(rows[0]["source_answer_span_end"]) - 1
        )
        with self.assertRaisesRegex(
            Section62AblationContractError, "answer span does not recover"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)

        rows = self.rows()
        rows[0]["rho_src"] = -float(rows[0]["rho_src"])
        with self.assertRaisesRegex(
            Section62AblationContractError, "rho_src does not match"
        ):
            build_analysis_membership(rows, min_baseline_correct_per_cell=1)


class RotationPreflightTests(unittest.TestCase):
    def test_frozen_profiles_pin_both_sites(self):
        phi = frozen_profiles_for_model("microsoft/Phi-4-mini-instruct")
        qwen = frozen_profiles_for_model("Qwen/Qwen3-8B")

        self.assertEqual(
            [(profile.reported_site, profile.layer, profile.rank) for profile in phi],
            [("claim_final", 12, 64), ("answer_token", 16, 64)],
        )
        self.assertEqual(
            [(profile.reported_site, profile.layer, profile.rank) for profile in qwen],
            [("claim_final", 18, 16), ("answer_token", 32, 16)],
        )
        self.assertEqual(phi[0].rho_metadata_site, "row")

    def test_preflights_profile_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_rotation(
                root / "rho",
                target_var="rho",
                model_name="example/model",
                layer=3,
                rank=2,
                hidden_size=4,
                site="row",
                fill=1.0,
            )
            write_rotation(
                root / "m",
                target_var="m",
                model_name="example/model",
                layer=3,
                rank=2,
                hidden_size=4,
                site="claim_final",
                fill=2.0,
            )
            profile = FrozenSiteProfile(
                model_name="example/model",
                model_key="example",
                reported_site="claim_final",
                layer=3,
                rank=2,
                hidden_size=4,
                rho_rotation_dir="rho",
                m_rotation_dir="m",
                rho_metadata_site="row",
                m_metadata_site="claim_final",
            )

            result = preflight_frozen_profile(profile, repo_root=root)

            self.assertEqual(result["rho_resolved_site"], "claim_final")
            self.assertEqual(result["rho"]["shape"], [4, 2])
            self.assertEqual(result["rho"]["dtype"], "float32")
            self.assertNotEqual(
                result["rho"]["rotation_weight_npy"]["sha256"],
                result["m"]["rotation_weight_npy"]["sha256"],
            )

    def test_rejects_explicit_frozen_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            rotation = write_rotation(
                Path(temp_dir) / "rho",
                target_var="rho",
                model_name="example/model",
                layer=3,
                rank=2,
                hidden_size=4,
                site="claim_final",
                fill=1.0,
            )
            metadata_path = rotation / "rotation_weight_metadata.json"
            npy_path = rotation / "rotation_weight.npy"
            pt_path = rotation / "rotation_weight.pt"
            expected_sha256 = {
                "rotation_weight_metadata.json": sha256_file(metadata_path),
                "rotation_weight.npy": "0" * 64,
                "rotation_weight.pt": sha256_file(pt_path),
            }
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "frozen SHA256 mismatch",
            ):
                preflight_rotation_metadata(
                    rotation,
                    expected_target_var="rho",
                    expected_model_name="example/model",
                    expected_layer=3,
                    expected_rank=2,
                    expected_hidden_size=4,
                    allowed_metadata_sites=("claim_final",),
                    expected_sha256=expected_sha256,
                )
            self.assertTrue(npy_path.is_file())

    def test_rejects_metadata_or_actual_array_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rotation = write_rotation(
                root / "rho",
                target_var="rho",
                model_name="example/model",
                layer=3,
                rank=2,
                hidden_size=4,
                site="claim_final",
                fill=1.0,
            )
            metadata_path = rotation / "rotation_weight_metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["weights"]["weight"]["shape"] = [4, 1]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "metadata shape",
            ):
                preflight_rotation_metadata(
                    rotation,
                    expected_target_var="rho",
                    expected_model_name="example/model",
                    expected_layer=3,
                    expected_rank=2,
                    expected_hidden_size=4,
                    allowed_metadata_sites=("claim_final",),
                )

            metadata["weights"]["weight"]["shape"] = [4, 2]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            np.save(
                rotation / "rotation_weight.npy",
                np.zeros((4, 1), dtype=np.float32),
            )
            with self.assertRaisesRegex(
                Section62AblationContractError,
                "actual shape",
            ):
                preflight_rotation_metadata(
                    rotation,
                    expected_target_var="rho",
                    expected_model_name="example/model",
                    expected_layer=3,
                    expected_rank=2,
                    expected_hidden_size=4,
                    allowed_metadata_sites=("claim_final",),
                )


if __name__ == "__main__":
    unittest.main()
