from __future__ import annotations

import sys
import csv
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.io_utils import write_rows_csv  # noqa: E402
from interference_suite.section62_data import (  # noqa: E402
    FULL_CELL_ORDER,
    RHO_SAME_DONOR,
    sha256_file,
)
from run_section62 import validate_behavioral_rows, validate_r0_manifest  # noqa: E402


SQUARE_LABELS = {"++": "T", "-+": "F", "+-": "F", "--": "T"}


def input_rows(
    stratum: str,
    *,
    corpus: str = "MNLI",
    triple_id: str | None = None,
) -> list[dict[str, object]]:
    resolved_triple_id = triple_id or (
        "mnli_square" if stratum == "square_valid" else "mnli_all_u"
    )
    signature = "(E,C,C,E)" if stratum == "square_valid" else "(N,N,N,N)"
    rows: list[dict[str, object]] = []
    for cell in FULL_CELL_ORDER:
        rho = 1 if cell in {"++", "--"} else -1
        claim = f"claim {cell}"
        prompt = f"Premise: premise {cell}\nClaim: {claim}\nLabel:"
        claim_start = prompt.index(claim)
        rows.append(
            {
                "sample_id": f"{resolved_triple_id}_{cell}",
                "corpus": corpus,
                "triple_id": resolved_triple_id,
                "analysis_stratum": stratum,
                "cell": cell,
                "expected_label": (
                    SQUARE_LABELS[cell] if stratum == "square_valid" else "U"
                ),
                "target_label": (
                    SQUARE_LABELS[cell] if stratum == "square_valid" else "U"
                ),
                "base_label": (
                    SQUARE_LABELS[cell] if stratum == "square_valid" else "U"
                ),
                "target_var": "rho",
                "prompt": prompt,
                "base_prompt": prompt,
                "base_claim": claim,
                "base_site": "claim_final",
                "base_claim_span_start": claim_start,
                "base_claim_span_end": claim_start + len(claim),
                "full_signature": signature,
                "rho_base": rho,
            }
        )
    by_cell = {str(row["cell"]): row for row in rows}
    for row in rows:
        donor = by_cell[RHO_SAME_DONOR[str(row["cell"])]]
        row.update(
            {
                "source_prompt": donor["base_prompt"],
                "source_label": donor["base_label"],
                "source_claim": donor["base_claim"],
                "source_site": donor["base_site"],
                "source_claim_span_start": donor["base_claim_span_start"],
                "source_claim_span_end": donor["base_claim_span_end"],
                "rho_src": donor["rho_base"],
                "portability_donor_cell": donor["cell"],
                "portability_donor_sample_id": donor["sample_id"],
                "rho_same_donor_cell": donor["cell"],
                "rho_same_donor_sample_id": donor["sample_id"],
                "portability_group": (
                    "negation_count_parity"
                    if row["cell"] in {"++", "--"}
                    else "negation_position"
                ),
            }
        )

    return rows


class Section62BehavioralInputTests(unittest.TestCase):
    def test_accepts_complete_square_and_all_u_inputs(self):
        rows = validate_behavioral_rows(
            input_rows("square_valid"),
            input_rows("all_u"),
        )

        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row["analysis_stratum"] for row in rows[:4]],
            ["square_valid"] * 4,
        )
        self.assertEqual(
            [row["analysis_stratum"] for row in rows[4:]],
            ["all_u"] * 4,
        )

    def test_rejects_wrong_label_signature_or_rho_before_model_load(self):
        square = input_rows("square_valid")
        square[0]["expected_label"] = "F"
        with self.assertRaisesRegex(ValueError, "expected_label"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        square[0]["full_signature"] = "(N,N,N,N)"
        with self.assertRaisesRegex(ValueError, "full_signature"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        square[0]["rho_base"] = -1
        with self.assertRaisesRegex(ValueError, "rho_base"):
            validate_behavioral_rows(square, input_rows("all_u"))

    def test_rejects_mismatched_corpora_and_incomplete_squares(self):
        with self.assertRaisesRegex(ValueError, "same corpora"):
            validate_behavioral_rows(
                input_rows("square_valid"),
                input_rows("all_u", corpus="SNLI", triple_id="snli_all_u"),
            )

        with self.assertRaisesRegex(ValueError, "has cells"):
            validate_behavioral_rows(
                input_rows("square_valid")[:-1],
                input_rows("all_u"),
            )

    def test_rejects_broken_labels_spans_and_donor_materialization(self):
        square = input_rows("square_valid")
        square[0]["base_label"] = "F"
        with self.assertRaisesRegex(ValueError, "base_label == target_label"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        del square[0]["base_claim_span_end"]
        with self.assertRaisesRegex(ValueError, "base_claim_span_end"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        square[0]["source_label"] = "F"
        with self.assertRaisesRegex(ValueError, "source_label does not match"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        square[0]["portability_donor_sample_id"] = square[1]["sample_id"]
        with self.assertRaisesRegex(ValueError, "wrong donor cell"):
            validate_behavioral_rows(square, input_rows("all_u"))

        square = input_rows("square_valid")
        square[0]["portability_group"] = "negation_position"
        with self.assertRaisesRegex(ValueError, "portability_group"):
            validate_behavioral_rows(square, input_rows("all_u"))


    def test_rejects_duplicate_sample_ids_across_inputs(self):
        square = input_rows("square_valid")
        all_u = input_rows("all_u")
        all_u[0]["sample_id"] = square[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate behavioral sample_id"):
            validate_behavioral_rows(square, all_u)

    def test_r0_manifest_pins_input_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            square_path = root / "mnli_square_valid.csv"
            all_u_path = root / "mnli_all_u.csv"
            square_rows = [{"sample_id": "square"}]
            all_u_rows = [{"sample_id": "all_u"}]
            write_rows_csv(square_rows, square_path)
            write_rows_csv(all_u_rows, all_u_path)
            manifest = {
                "schema_version": 1,
                "outputs": {
                    "mnli_square_valid": {
                        "path": square_path.name,
                        "rows": 1,
                        "sha256": sha256_file(square_path),
                        "bytes": square_path.stat().st_size,
                    },
                    "mnli_all_u": {
                        "path": all_u_path.name,
                        "rows": 1,
                        "sha256": sha256_file(all_u_path),
                        "bytes": all_u_path.stat().st_size,
                    },
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            provenance = validate_r0_manifest(
                manifest_path,
                square_path=square_path,
                square_rows=square_rows,
                all_u_path=all_u_path,
                all_u_rows=all_u_rows,
            )
            self.assertEqual(
                provenance["matched_artifacts"]["square_valid"][
                    "artifact_name"
                ],
                "mnli_square_valid",
            )

            manifest["outputs"]["mnli_square_valid"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_r0_manifest(
                    manifest_path,
                    square_path=square_path,
                    square_rows=square_rows,
                    all_u_path=all_u_path,
                    all_u_rows=all_u_rows,
                )

    def test_empty_filtered_csv_retains_parent_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.csv"
            write_rows_csv(
                [],
                path,
                schema_rows=[{"sample_id": "x", "condition": "base"}],
            )
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["sample_id", "condition"])
                self.assertEqual(list(reader), [])


if __name__ == "__main__":
    unittest.main()
