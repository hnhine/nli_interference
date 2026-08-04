from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section62_data import (  # noqa: E402
    Section62DataError,
    SourceJoinError,
    StructuralValidationError,
    prepare_corpus,
    read_tsv,
    text_pair_sha256,
)


def write_tsv(
    path: Path,
    rows: list[dict[str, str]],
    *,
    encoding: str = "utf-8",
) -> None:
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def transformed_rows(
    *,
    start: int,
    premise_positive: str,
    premise_negative: str,
    hypothesis_positive: str,
    hypothesis_negative: str,
    labels: tuple[str, str, str],
) -> list[dict[str, str]]:
    return [
        {
            "index": str(start),
            "Text": premise_negative,
            "Hypothesis": hypothesis_positive,
            "gold_label": labels[0],
        },
        {
            "index": str(start + 1),
            "Text": premise_positive,
            "Hypothesis": hypothesis_negative,
            "gold_label": labels[1],
        },
        {
            "index": str(start + 2),
            "Text": premise_negative,
            "Hypothesis": hypothesis_negative,
            "gold_label": labels[2],
        },
    ]


class Section62DataTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def prepare(
        self,
        negation_rows: list[dict[str, str]],
        source_rows: list[dict[str, str]],
        **kwargs,
    ):
        negation = self.root / "MNLI.txt"
        source = self.root / "train.tsv"
        write_tsv(negation, negation_rows, encoding="cp1252")
        write_tsv(source, source_rows)
        return prepare_corpus(
            corpus="MNLI",
            negation_path=negation,
            source_paths={"train": source},
            expected_triples=len(negation_rows) // 3,
            **kwargs,
        )

    def test_raw_tsv_treats_ascii_quotes_as_text_and_keeps_shape_strict(self):
        raw = self.root / "raw.tsv"
        raw.write_text(
            'gold_label\tpremise\thypothesis\n'
            'entailment\t"Quoted premise.\tPlain hypothesis.\n',
            encoding="utf-8",
        )
        fieldnames, rows = read_tsv(raw)
        self.assertEqual(fieldnames, ["gold_label", "premise", "hypothesis"])
        self.assertEqual(rows[0]["premise"], '"Quoted premise.')
        self.assertEqual(rows[0]["hypothesis"], "Plain hypothesis.")

        malformed = self.root / "extra.tsv"
        malformed.write_text(
            "gold_label\tpremise\thypothesis\n"
            "entailment\tP.\tH.\tnonempty-extra\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(Section62DataError, "extra TSV field"):
            read_tsv(malformed)

    def test_explicit_source_override_is_hash_verified_and_audited(self):
        negation_rows = transformed_rows(
            start=0,
            premise_positive="Edited premise.",
            premise_negative="Not edited premise.",
            hypothesis_positive="Corrected hypothesis.",
            hypothesis_negative="Not corrected hypothesis.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        source_rows = [
            {
                "pairID": "canonical-e",
                "sentence1": "Edited premise.",
                "sentence2": "Corected hypothesis.",
                "gold_label": "entailment",
            }
        ]
        override = {
            "source_split": "train",
            "pair_id": "canonical-e",
            "anchor_sides": ["premise"],
            "clean_pair_sha256": text_pair_sha256(
                "Edited premise.", "Corrected hypothesis."
            ),
            "source_pair_sha256": text_pair_sha256(
                "Edited premise.", "Corected hypothesis."
            ),
            "expected_gold_label": "entailment",
            "reason": "fixture correction",
        }
        prepared = self.prepare(
            negation_rows,
            source_rows,
            source_join_overrides={"mnli_0000": override},
            require_complete_joins=True,
        )
        triple = prepared.triples[0]
        self.assertEqual(triple["join_status"], "explicit_source_override")
        self.assertEqual(triple["gold_original"], "E")
        self.assertEqual(triple["join_override_pair_id"], "canonical-e")
        self.assertEqual(len(prepared.square_rows), 4)
        self.assertTrue(
            all(row["join_override_used"] == 1 for row in prepared.square_rows)
        )
        counts = prepared.manifest["counts"]
        self.assertEqual(counts["source_join_overrides_requested"], 1)
        self.assertEqual(counts["source_join_overrides_applied"], 1)
        self.assertEqual(counts["square_valid_without_overrides_triples"], 0)

        bad_hash = {**override, "clean_pair_sha256": "0" * 64}
        with self.assertRaisesRegex(SourceJoinError, "clean-pair hash mismatch"):
            self.prepare(
                negation_rows,
                source_rows,
                source_join_overrides={"mnli_0000": bad_hash},
            )

        exact_source = [
            {
                **source_rows[0],
                "sentence2": "Corrected hypothesis.",
            }
        ]
        with self.assertRaisesRegex(SourceJoinError, "stale source override"):
            self.prepare(
                negation_rows,
                exact_source,
                source_join_overrides={"mnli_0000": override},
            )

        with self.assertRaisesRegex(SourceJoinError, "were not applied"):
            self.prepare(
                negation_rows,
                source_rows,
                source_join_overrides={"mnli_0001": override},
            )

    def test_reconstructs_ecce_square_and_portability_groups(self):
        negation_rows = transformed_rows(
            start=0,
            premise_positive="His knees were “bent”.",
            premise_negative="His knees were not “bent”.",
            hypothesis_positive="He bent his legs.",
            hypothesis_negative="He did not bend his legs.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        source_rows = [
            {
                "index": "source-1",
                "sentence1": "His knees were “bent”.",
                "sentence2": "He bent his legs.",
                "gold_label": "entailment",
            }
        ]
        prepared = self.prepare(negation_rows, source_rows)

        self.assertEqual(len(prepared.square_rows), 4)
        self.assertEqual(
            {row["cell"]: row["base_label"] for row in prepared.square_rows},
            {"++": "T", "-+": "F", "+-": "F", "--": "T"},
        )
        self.assertEqual(
            {row["cell"]: row["rho_same_donor_cell"] for row in prepared.square_rows},
            {"++": "--", "--": "++", "-+": "+-", "+-": "-+"},
        )
        self.assertEqual(
            {
                row["cell"]: row["portability_group"]
                for row in prepared.square_rows
            },
            {
                "++": "negation_count_parity",
                "--": "negation_count_parity",
                "-+": "negation_position",
                "+-": "negation_position",
            },
        )
        self.assertEqual(
            {row["cell"]: row["sample_id"] for row in prepared.square_rows},
            {
                "++": "section62_mnli_0000_pp",
                "-+": "section62_mnli_0000_np",
                "+-": "section62_mnli_0000_pn",
                "--": "section62_mnli_0000_nn",
            },
        )
        self.assertEqual(
            {
                row["cell"]: (row["base_source_row_index"], row["source_row_index"])
                for row in prepared.square_rows
            },
            {"++": ("", 2), "-+": (0, 1), "+-": (1, 0), "--": (2, "")},
        )
        by_sample_id = {row["sample_id"]: row for row in prepared.square_rows}
        for row in prepared.square_rows:
            start = int(row["base_claim_span_start"])
            end = int(row["base_claim_span_end"])
            self.assertEqual(row["base_prompt"][start:end], row["base_claim"])
            self.assertEqual(row["rho_src"], row["rho_base"])
            self.assertEqual(row["target_label"], row["base_label"])
            source_start = int(row["source_claim_span_start"])
            source_end = int(row["source_claim_span_end"])
            self.assertEqual(row["source_prompt"][source_start:source_end], row["source_claim"])
            self.assertNotIn("m_base", row)
            donor = by_sample_id[row["portability_donor_sample_id"]]
            self.assertEqual(donor["portability_donor_sample_id"], row["sample_id"])

    def test_manifest_records_signatures_chain_and_final_retention(self):
        first = transformed_rows(
            start=0,
            premise_positive="P one.",
            premise_negative="Not P one.",
            hypothesis_positive="H one.",
            hypothesis_negative="Not H one.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        second = transformed_rows(
            start=3,
            premise_positive="P two.",
            premise_negative="Not P two.",
            hypothesis_positive="H two.",
            hypothesis_negative="Not H two.",
            labels=("neutral", "neutral", "neutral"),
        )
        sources = [
            {
                "index": "a",
                "sentence1": "P one.",
                "sentence2": "H one.",
                "gold_label": "entailment",
            },
            {
                "index": "b",
                "sentence1": "P two.",
                "sentence2": "H two.",
                "gold_label": "neutral",
            },
        ]
        prepared = self.prepare(first + second, sources)
        counts = prepared.manifest["counts"]
        self.assertEqual(prepared.manifest["input"]["negation_encoding"], "cp1252")

        self.assertEqual(counts["raw_triples"], 2)
        self.assertEqual(counts["candidate_cce_triples"], 1)
        self.assertEqual(counts["square_valid_triples"], 1)
        self.assertEqual(counts["transformed_all_u_triples"], 1)
        self.assertEqual(counts["all_u_square_triples"], 1)
        self.assertEqual(counts["all_u_rows"], 4)
        self.assertEqual(
            {row["analysis_stratum"] for row in prepared.all_u_rows},
            {"all_u"},
        )
        self.assertTrue(all(row["full_signature"] == "(N,N,N,N)" for row in prepared.all_u_rows))
        self.assertEqual(
            prepared.manifest["transformed_signature_counts"],
            {"(C,C,E)": 1, "(N,N,N)": 1},
        )
        self.assertEqual(
            prepared.manifest["square_valid_retention_rate_of_raw"],
            0.5,
        )
        self.assertIsNone(
            prepared.manifest["retention_chain"]["behavioral_pass"]
        )

        second_only = [{**row, "index": str(index)} for index, row in enumerate(second)]
        second_ent_source = [{**sources[1], "gold_label": "entailment"}]
        not_all_u = self.prepare(second_only, second_ent_source)
        self.assertEqual(len(not_all_u.all_u_rows), 0)

    def test_structural_mismatch_fails_loud_without_quarantine(self):
        rows = transformed_rows(
            start=0,
            premise_positive="P.",
            premise_negative="Not P.",
            hypothesis_positive="H.",
            hypothesis_negative="Not H.",
            labels=("neutral", "neutral", "neutral"),
        )
        rows[2]["Text"] = "Different negative premise."
        source_rows = [
            {
                "index": "a",
                "sentence1": "P.",
                "sentence2": "H.",
                "gold_label": "neutral",
            }
        ]
        bad_index_rows = [dict(row) for row in rows]
        bad_index_rows[1]["index"] = "9"
        with self.assertRaisesRegex(StructuralValidationError, "exactly 0..N-1"):
            self.prepare(bad_index_rows, source_rows)
        with self.assertRaisesRegex(
            StructuralValidationError, r"mnli_0000.*P\(-\+\) != P\(--\)"
        ):
            self.prepare(rows, source_rows)

    def test_explicit_quarantine_skips_only_named_invalid_group(self):
        invalid = transformed_rows(
            start=0,
            premise_positive="Bad P.",
            premise_negative="Not bad P.",
            hypothesis_positive="Bad H.",
            hypothesis_negative="Not bad H.",
            labels=("neutral", "neutral", "neutral"),
        )
        invalid[2]["Hypothesis"] = "Mismatched H."
        valid = transformed_rows(
            start=3,
            premise_positive="Good P.",
            premise_negative="Not good P.",
            hypothesis_positive="Good H.",
            hypothesis_negative="Not good H.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        source_rows = [
            {
                "index": "good",
                "sentence1": "Good P.",
                "sentence2": "Good H.",
                "gold_label": "entailment",
            }
        ]
        prepared = self.prepare(
            invalid + valid,
            source_rows,
            quarantine={"mnli_0000": "known upstream malformed triple"},
        )

        self.assertEqual(prepared.manifest["counts"]["quarantined_triples"], 1)
        self.assertEqual(prepared.manifest["counts"]["square_valid_triples"], 1)
        self.assertEqual(
            prepared.manifest["quarantined"][0]["quarantine_reason"],
            "known upstream malformed triple",
        )

    def test_conflicting_duplicate_source_labels_fail(self):
        rows = transformed_rows(
            start=0,
            premise_positive="P.",
            premise_negative="Not P.",
            hypothesis_positive="H.",
            hypothesis_negative="Not H.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        source_rows = [
            {
                "index": "a",
                "sentence1": "P.",
                "sentence2": "H.",
                "gold_label": "entailment",
            },
            {
                "index": "b",
                "sentence1": "P.",
                "sentence2": "H.",
                "gold_label": "neutral",
            },
        ]
        same_label_rows = [
            source_rows[0],
            {**source_rows[0], "index": "copy"},
        ]
        same_label = self.prepare(rows, same_label_rows)
        self.assertEqual(same_label.triples[0]["join_match_count"], 2)
        unlabeled = {
            **source_rows[0], "index": "unlabeled", "gold_label": "-"
        }
        with self.assertRaisesRegex(SourceJoinError, "unlabeled source provenance"):
            self.prepare(rows, [source_rows[0], unlabeled])
        with self.assertRaisesRegex(SourceJoinError, "conflicting \\+\\+ labels"):
            self.prepare(rows, source_rows)

    def test_normalized_join_is_logged_and_unresolved_is_not_valid(self):
        first = transformed_rows(
            start=0,
            premise_positive="P   one.",
            premise_negative="Not P one.",
            hypothesis_positive="H one.",
            hypothesis_negative="Not H one.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        second = transformed_rows(
            start=3,
            premise_positive="Missing P.",
            premise_negative="Not missing P.",
            hypothesis_positive="Missing H.",
            hypothesis_negative="Not missing H.",
            labels=("contradiction", "contradiction", "entailment"),
        )
        source_rows = [
            {
                "index": "a",
                "sentence1": "P one.",
                "sentence2": "H one.",
                "gold_label": "entailment",
            }
        ]
        exact_only = self.prepare(first + second, source_rows)
        exact_by_id = {row["triple_id"]: row for row in exact_only.triples}
        self.assertEqual(exact_by_id["mnli_0000"]["join_status"], "unresolved")
        self.assertEqual(exact_only.manifest["counts"]["square_valid_triples"], 0)
        self.assertFalse(
            exact_only.manifest["join_policy"]["normalized_fallback_enabled"]
        )
        with self.assertRaisesRegex(SourceJoinError, "were unresolved"):
            self.prepare(
                first + second, source_rows, require_complete_joins=True
            )
        prepared = self.prepare(
            first + second, source_rows, allow_normalized_fallback=True
        )
        by_id = {row["triple_id"]: row for row in prepared.triples}

        self.assertEqual(by_id["mnli_0000"]["join_status"], "normalized_unique")
        self.assertEqual(by_id["mnli_0001"]["join_status"], "unresolved")
        self.assertEqual(prepared.manifest["counts"]["candidate_cce_triples"], 2)
        self.assertEqual(prepared.manifest["counts"]["square_valid_triples"], 1)


if __name__ == "__main__":
    unittest.main()
