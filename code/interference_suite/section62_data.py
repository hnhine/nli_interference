"""Data preparation for Section 6.2 naturalistic polarity squares.

The Hossain et al. negation benchmarks contain three consecutive variants of
each original NLI pair:

``-+`` (negated premise), ``+-`` (negated hypothesis), and ``--`` (both
negated).  This module validates that structure, reconstructs the missing
``++`` pair, joins its gold label from the original corpus, and marks complete
human-annotated Boolean squares with the signature ``(E,C,C,E)``.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .das_spans import add_span_columns, build_prompt_with_spans

NEGATED_CELL_ORDER = ("-+", "+-", "--")
FULL_CELL_ORDER = ("++", "-+", "+-", "--")
CELL_VALUES = {
    "++": (1, 1),
    "-+": (-1, 1),
    "+-": (1, -1),
    "--": (-1, -1),
}
CELL_NEGATION_LOCATION = {
    "++": "none",
    "-+": "premise",
    "+-": "hypothesis",
    "--": "both",
}
RHO_SAME_DONOR = {
    "++": "--",
    "--": "++",
    "-+": "+-",
    "+-": "-+",
}
CELL_ID_TAG = {"++": "pp", "-+": "np", "+-": "pn", "--": "nn"}
CELL_SOURCE_OFFSET = {"-+": 0, "+-": 1, "--": 2}
GOLD_TO_TFU = {"E": "T", "C": "F", "N": "U"}
UPSTREAM_REPOSITORY = "https://github.com/mosharafhossain/negation-and-nli"
UPSTREAM_REVISION = "e7848fb621c32ec43511a5bfc406fc9208baf4f8"
UPSTREAM_GIT_BLOBS = {
    "MNLI": "b441baa207b202379d0cd23421ae492a98c233b3",
    "SNLI": "93270275cdee2ad9c670c90331df791416f8aa22",
}
UPSTREAM_TEXT_ENCODINGS = {
    "MNLI": "cp1252",
    "SNLI": "utf-8",
}
JOIN_NORMALIZATION_VERSION = "nfc_whitespace_v1"
SOURCE_JOIN_OVERRIDE_FIELDS = frozenset(
    {
        "source_split",
        "pair_id",
        "anchor_sides",
        "clean_pair_sha256",
        "source_pair_sha256",
        "expected_gold_label",
        "reason",
    }
)
SOURCE_PAIR_HASH_VERSION = "utf8_nul_pair_v1"


class Section62DataError(ValueError):
    """Base class for deterministic Section 6.2 data failures."""


class StructuralValidationError(Section62DataError):
    """Raised when a negation triple violates the upstream construction."""


class SourceJoinError(Section62DataError):
    """Raised when reconstructed ``++`` pairs have ambiguous source labels."""


@dataclass(frozen=True)
class SourceRecord:
    corpus: str
    split: str
    path: str
    line_number: int
    source_id: str
    premise: str
    hypothesis: str
    gold: str


@dataclass
class SourceIndex:
    exact: dict[tuple[str, str], list[SourceRecord]]
    normalized: dict[tuple[str, str], list[SourceRecord]]
    invalid_exact: dict[tuple[str, str], list[SourceRecord]]
    invalid_normalized: dict[tuple[str, str], list[SourceRecord]]
    by_provenance: dict[tuple[str, str], list[SourceRecord]]
    manifest: dict[str, Any]


@dataclass
class PreparedCorpus:
    corpus: str
    triples: list[dict[str, Any]]
    all_rows: list[dict[str, Any]]
    square_rows: list[dict[str, Any]]
    all_u_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def text_pair_sha256(premise: str, hypothesis: str) -> str:
    payload = (
        "premise\0" + premise + "\0hypothesis\0" + hypothesis
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {source.stat().st_size}\0".encode("ascii"))
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_gold_label(value: Any, *, allow_missing: bool = False) -> str:
    raw = str(value if value is not None else "").strip().lower()
    if allow_missing and raw in {"", "-"}:
        return ""
    aliases = {
        "e": "E",
        "entailment": "E",
        "c": "C",
        "contradiction": "C",
        "n": "N",
        "neutral": "N",
    }
    if raw not in aliases:
        raise Section62DataError(f"Unknown three-way NLI label: {value!r}")
    return aliases[raw]


def normalize_whitespace(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def read_tsv(
    path: str | Path,
    *,
    encoding: str = "utf-8",
) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    with source.open(newline="", encoding=encoding) as handle:
        # The upstream NLI releases are raw TSV, not RFC-style quoted CSV.
        # Natural-language fields contain unmatched ASCII quotes; treating those
        # quotes as delimiters can swallow tabs/newlines and silently shift rows.
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        if not reader.fieldnames:
            raise Section62DataError(f"{source}: missing TSV header")
        fieldnames = [str(name) for name in reader.fieldnames]
        if len(set(fieldnames)) != len(fieldnames):
            raise Section62DataError(
                f"{source}: duplicate TSV header names: {fieldnames}"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row:
                raise Section62DataError(
                    f"{source}:{reader.line_num}: extra TSV field(s): {row[None]}"
                )
            missing = [name for name in fieldnames if row.get(name) is None]
            if missing:
                raise Section62DataError(
                    f"{source}:{reader.line_num}: missing TSV field(s): {missing}"
                )
            clean = {name: str(row[name]) for name in fieldnames}
            clean["__line_number__"] = str(reader.line_num)
            rows.append(clean)
    return fieldnames, rows


def resolve_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    *,
    path: str | Path,
    role: str,
    required: bool = True,
) -> str | None:
    by_folded = {name.strip().casefold(): name for name in fieldnames}
    for alias in aliases:
        if alias.casefold() in by_folded:
            return by_folded[alias.casefold()]
    if not required:
        return None
    raise Section62DataError(
        f"{path}: cannot resolve {role} column; expected one of {list(aliases)}, "
        f"found {list(fieldnames)}"
    )


def read_negation_rows(path: str | Path, corpus: str) -> list[dict[str, Any]]:
    encoding = UPSTREAM_TEXT_ENCODINGS.get(corpus.upper(), "utf-8")
    fieldnames, raw_rows = read_tsv(path, encoding=encoding)
    expected_header = ["index", "Text", "Hypothesis", "gold_label"]
    if fieldnames != expected_header:
        raise Section62DataError(
            f"{path}: expected exact Hossain header {expected_header}, found {fieldnames}"
        )
    premise_col = resolve_column(
        fieldnames, ("Text", "sentence1", "premise"), path=path, role="premise"
    )
    hypothesis_col = resolve_column(
        fieldnames, ("Hypothesis", "sentence2"), path=path, role="hypothesis"
    )
    label_col = resolve_column(
        fieldnames, ("gold_label", "label"), path=path, role="gold label"
    )
    index_col = resolve_column(
        fieldnames, ("index", "id", "pairID", "pair_id"), path=path, role="index"
    )
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(raw_rows):
        rows.append(
            {
                "corpus": corpus.upper(),
                "position": position,
                "line_number": int(row["__line_number__"]),
                "raw_index": row[index_col].strip(),
                "premise": row[premise_col],
                "hypothesis": row[hypothesis_col],
                "gold": normalize_gold_label(row[label_col]),
            }
        )
    return rows


def _signature(labels: Iterable[str]) -> str:
    return "(" + ",".join(labels) + ")"


def _quarantine_key(
    triple_id: str,
    quarantine: Mapping[str, str],
) -> str | None:
    return triple_id if triple_id in quarantine else None


def group_negation_triples(
    rows: Sequence[dict[str, Any]],
    corpus: str,
    *,
    expected_triples: int | None = 500,
    quarantine: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group and strictly validate consecutive ``-+,+-,--`` rows.

    Invalid groups must be named in an explicit quarantine mapping.  Stale
    quarantine entries are errors so an upstream data change cannot silently
    alter the analysis set.
    """

    bad_indices: list[str] = []
    seen_indices: set[int] = set()
    for position, row in enumerate(rows):
        try:
            raw_index = int(str(row["raw_index"]))
        except ValueError:
            bad_indices.append(
                f"position {position}: non-integer index {row['raw_index']!r}"
            )
            continue
        if raw_index in seen_indices:
            bad_indices.append(f"position {position}: duplicate index {raw_index}")
        if raw_index != position:
            bad_indices.append(
                f"position {position}: expected index {position}, found {raw_index}"
            )
        seen_indices.add(raw_index)
    if bad_indices:
        preview = "; ".join(bad_indices[:10])
        suffix = " ..." if len(bad_indices) > 10 else ""
        raise StructuralValidationError(
            f"{corpus}: generated row indices must be exactly 0..N-1: "
            f"{preview}{suffix}"
        )
    corpus = corpus.upper()
    quarantine = dict(quarantine or {})
    if len(rows) % 3:
        raise StructuralValidationError(
            f"{corpus}: expected a multiple of three rows, found {len(rows)}"
        )
    raw_triples = len(rows) // 3
    if expected_triples is not None and raw_triples != expected_triples:
        raise StructuralValidationError(
            f"{corpus}: expected {expected_triples} triples, found {raw_triples}"
        )

    triples: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    invalid: list[str] = []
    used_quarantine: set[str] = set()
    for group_index in range(raw_triples):
        neg_p, neg_h, neg_both = rows[group_index * 3 : group_index * 3 + 3]
        triple_id = f"{corpus.lower()}_{group_index:04d}"
        raw_indices = (
            str(neg_p["raw_index"]),
            str(neg_h["raw_index"]),
            str(neg_both["raw_index"]),
        )
        issues: list[str] = []
        if neg_p["premise"] != neg_both["premise"]:
            issues.append("P(-+) != P(--)")
        if neg_h["hypothesis"] != neg_both["hypothesis"]:
            issues.append("H(+-) != H(--)")
        if neg_p["premise"] == neg_h["premise"]:
            issues.append("negative and affirmative premises are identical")
        if neg_p["hypothesis"] == neg_h["hypothesis"]:
            issues.append("affirmative and negative hypotheses are identical")

        if issues:
            key = _quarantine_key(triple_id, quarantine)
            detail = {
                "triple_id": triple_id,
                "group_index": group_index,
                "raw_indices": list(raw_indices),
                "lines": [
                    neg_p["line_number"],
                    neg_h["line_number"],
                    neg_both["line_number"],
                ],
                "issues": issues,
            }
            if key is None:
                invalid.append(
                    f"{triple_id} lines={detail['lines']}: " + "; ".join(issues)
                )
            else:
                used_quarantine.add(key)
                detail["quarantine_reason"] = quarantine[key]
                quarantined.append(detail)
            continue

        gold_by_cell = {
            "-+": neg_p["gold"],
            "+-": neg_h["gold"],
            "--": neg_both["gold"],
        }
        triples.append(
            {
                "corpus": corpus,
                "triple_id": triple_id,
                "group_index": group_index,
                "raw_indices": "|".join(raw_indices),
                "raw_lines": "|".join(
                    str(row["line_number"]) for row in (neg_p, neg_h, neg_both)
                ),
                "premise_positive": neg_h["premise"],
                "premise_negative": neg_p["premise"],
                "hypothesis_positive": neg_p["hypothesis"],
                "hypothesis_negative": neg_h["hypothesis"],
                "gold_neg_premise": gold_by_cell["-+"],
                "gold_neg_hypothesis": gold_by_cell["+-"],
                "gold_both_negative": gold_by_cell["--"],
                "transformed_signature": _signature(
                    gold_by_cell[cell] for cell in NEGATED_CELL_ORDER
                ),
            }
        )

    if invalid:
        joined = "\n  ".join(invalid)
        raise StructuralValidationError(
            f"{corpus}: {len(invalid)} invalid triple(s) not quarantined:\n  {joined}"
        )
    unused = sorted(set(quarantine) - used_quarantine)
    if unused:
        raise StructuralValidationError(
            f"{corpus}: quarantine entries did not match invalid triples: {unused}"
        )
    return triples, quarantined


def load_source_index(
    corpus: str,
    source_paths: Mapping[str, str | Path],
) -> SourceIndex:
    corpus = corpus.upper()
    exact: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    normalized: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    invalid_exact: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    invalid_normalized: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    by_provenance: dict[tuple[str, str], list[SourceRecord]] = defaultdict(list)
    file_manifests: list[dict[str, Any]] = []
    for split, raw_path in sorted(source_paths.items()):
        path = Path(raw_path)
        fieldnames, raw_rows = read_tsv(path)
        premise_col = resolve_column(
            fieldnames,
            ("sentence1", "premise", "Text"),
            path=path,
            role="premise",
        )
        hypothesis_col = resolve_column(
            fieldnames,
            ("sentence2", "hypothesis", "Hypothesis"),
            path=path,
            role="hypothesis",
        )
        label_col = resolve_column(
            fieldnames,
            ("gold_label", "label"),
            path=path,
            role="gold label",
        )
        id_col = resolve_column(
            fieldnames,
            ("pairID", "pair_id", "index", "captionID", "id"),
            path=path,
            role="source id",
            required=False,
        )
        kept = 0
        unlabeled = 0
        for row in raw_rows:
            premise = row[premise_col]
            hypothesis = row[hypothesis_col]
            gold = normalize_gold_label(row[label_col], allow_missing=True)
            record = SourceRecord(
                corpus=corpus,
                split=str(split),
                path=str(path),
                line_number=int(row["__line_number__"]),
                source_id=row[id_col].strip() if id_col else "",
                premise=premise,
                hypothesis=hypothesis,
                gold=gold,
            )
            by_provenance[(record.split, record.source_id)].append(record)
            exact_key = (premise, hypothesis)
            normalized_key = (
                normalize_whitespace(premise),
                normalize_whitespace(hypothesis),
            )
            if gold:
                exact[exact_key].append(record)
                normalized[normalized_key].append(record)
                kept += 1
            else:
                invalid_exact[exact_key].append(record)
                invalid_normalized[normalized_key].append(record)
                unlabeled += 1
        file_manifests.append(
            {
                "split": split,
                "path": str(path),
                "encoding": "utf-8",
                "header": fieldnames,
                "resolved_columns": {
                    "premise": premise_col,
                    "hypothesis": hypothesis_col,
                    "gold_label": label_col,
                    "source_id": id_col,
                },
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(raw_rows),
                "labeled_rows": kept,
                "unlabeled_rows": unlabeled,
            }
        )
    if not source_paths:
        raise Section62DataError(f"{corpus}: at least one original source file is required")
    return SourceIndex(
        exact=dict(exact),
        normalized=dict(normalized),
        invalid_exact=dict(invalid_exact),
        invalid_normalized=dict(invalid_normalized),
        by_provenance=dict(by_provenance),
        manifest={"files": file_manifests},
    )


def _resolve_source_candidates(
    candidates: Sequence[SourceRecord],
    *,
    triple_id: str,
    match_kind: str,
    require_one_raw_pair: bool = False,
) -> dict[str, Any]:
    raw_pairs = {(record.premise, record.hypothesis) for record in candidates}
    if require_one_raw_pair and len(raw_pairs) != 1:
        details = [
            f"{record.split}:{record.source_id or record.line_number}"
            for record in candidates
        ]
        raise SourceJoinError(
            f"{triple_id}: {match_kind} fallback matched multiple distinct raw "
            f"source pairs: {details}"
        )
    labels = sorted({record.gold for record in candidates})
    if len(labels) != 1:
        details = [
            f"{record.split}:{record.source_id or record.line_number}={record.gold}"
            for record in candidates
        ]
        raise SourceJoinError(
            f"{triple_id}: conflicting ++ labels under {match_kind} match: {details}"
        )
    unique = len(candidates) == 1
    ordered = sorted(
        candidates,
        key=lambda record: (
            record.split,
            record.path,
            record.line_number,
            record.source_id,
        ),
    )
    return {
        "gold_original": labels[0],
        "join_status": f"{match_kind}_{'unique' if unique else 'duplicate_same_label'}",
        "join_match_count": len(ordered),
        "source_splits": "|".join(sorted({record.split for record in ordered})),
        "source_paths": "|".join(sorted({record.path for record in ordered})),
        "source_ids": "|".join(
            record.source_id or f"line:{record.line_number}" for record in ordered
        ),
        "source_locations": "|".join(
            f"{record.path}:{record.line_number}" for record in ordered
        ),
    }


def _resolve_source_override(
    triple: Mapping[str, Any],
    source_index: SourceIndex,
    override: Mapping[str, Any],
) -> dict[str, Any]:
    triple_id = str(triple["triple_id"])
    actual_fields = set(override)
    if actual_fields != SOURCE_JOIN_OVERRIDE_FIELDS:
        raise SourceJoinError(
            f"{triple_id}: source override fields must be exactly "
            f"{sorted(SOURCE_JOIN_OVERRIDE_FIELDS)}; "
            f"missing={sorted(SOURCE_JOIN_OVERRIDE_FIELDS - actual_fields)}, "
            f"extra={sorted(actual_fields - SOURCE_JOIN_OVERRIDE_FIELDS)}"
        )

    split = str(override["source_split"]).strip()
    pair_id = str(override["pair_id"]).strip()
    if not split or not pair_id:
        raise SourceJoinError(
            f"{triple_id}: source override split and pair_id must be nonempty"
        )
    candidates = source_index.by_provenance.get((split, pair_id), [])
    if len(candidates) != 1:
        raise SourceJoinError(
            f"{triple_id}: source override {split}:{pair_id} matched "
            f"{len(candidates)} canonical source rows; expected exactly one"
        )
    record = candidates[0]
    if not record.gold:
        raise SourceJoinError(
            f"{triple_id}: source override {split}:{pair_id} is unlabeled"
        )
    expected_gold = normalize_gold_label(override["expected_gold_label"])
    if record.gold != expected_gold:
        raise SourceJoinError(
            f"{triple_id}: source override expected gold {expected_gold}, "
            f"but canonical source has {record.gold}"
        )

    clean_hash = text_pair_sha256(
        str(triple["premise_positive"]),
        str(triple["hypothesis_positive"]),
    )
    expected_clean_hash = str(override["clean_pair_sha256"]).strip().lower()
    if clean_hash != expected_clean_hash:
        raise SourceJoinError(
            f"{triple_id}: clean-pair hash mismatch for source override: "
            f"expected {expected_clean_hash}, got {clean_hash}"
        )
    source_hash = text_pair_sha256(record.premise, record.hypothesis)
    expected_source_hash = str(override["source_pair_sha256"]).strip().lower()
    if source_hash != expected_source_hash:
        raise SourceJoinError(
            f"{triple_id}: source-pair hash mismatch for {split}:{pair_id}: "
            f"expected {expected_source_hash}, got {source_hash}"
        )

    raw_anchor_sides = override["anchor_sides"]
    if not isinstance(raw_anchor_sides, list) or not raw_anchor_sides:
        raise SourceJoinError(
            f"{triple_id}: source override anchor_sides must be a nonempty list"
        )
    declared_anchor_sides = tuple(sorted(str(side) for side in raw_anchor_sides))
    if (
        len(set(declared_anchor_sides)) != len(declared_anchor_sides)
        or not set(declared_anchor_sides) <= {"premise", "hypothesis"}
    ):
        raise SourceJoinError(
            f"{triple_id}: invalid source override anchor_sides "
            f"{list(raw_anchor_sides)}"
        )
    computed_anchor_sides = tuple(
        side
        for side, clean_text, source_text in (
            ("hypothesis", str(triple["hypothesis_positive"]), record.hypothesis),
            ("premise", str(triple["premise_positive"]), record.premise),
        )
        if normalize_whitespace(clean_text) == normalize_whitespace(source_text)
    )
    if not computed_anchor_sides:
        raise SourceJoinError(
            f"{triple_id}: source override {split}:{pair_id} has no "
            "NFC-whitespace-equal anchor side"
        )
    if computed_anchor_sides != declared_anchor_sides:
        raise SourceJoinError(
            f"{triple_id}: declared anchor_sides {declared_anchor_sides} do not "
            f"match computed anchors {computed_anchor_sides}"
        )
    reason = str(override["reason"]).strip()
    if not reason:
        raise SourceJoinError(f"{triple_id}: source override reason is empty")

    result = _resolve_source_candidates(
        [record], triple_id=triple_id, match_kind="explicit_source_override"
    )
    result.update(
        {
            "join_status": "explicit_source_override",
            "join_override_used": 1,
            "join_override_source_split": split,
            "join_override_pair_id": pair_id,
            "join_override_anchor_sides": "|".join(computed_anchor_sides),
            "join_override_clean_pair_sha256": clean_hash,
            "join_override_source_pair_sha256": source_hash,
            "join_override_expected_gold": expected_gold,
            "join_override_reason": reason,
        }
    )
    return result


def join_original_pair(
    triple: Mapping[str, Any],
    source_index: SourceIndex,
    *,
    allow_normalized_fallback: bool = False,
    source_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    triple_id = str(triple["triple_id"])
    premise = str(triple["premise_positive"])
    hypothesis = str(triple["hypothesis_positive"])
    exact_key = (premise, hypothesis)
    exact_invalid = source_index.invalid_exact.get(exact_key, [])
    if exact_invalid:
        details = [
            f"{record.split}:{record.source_id or record.line_number}"
            for record in exact_invalid
        ]
        raise SourceJoinError(
            f"{triple['triple_id']}: exact ++ match includes unlabeled source "
            f"provenance: {details}"
        )
    exact = source_index.exact.get(exact_key, [])
    if exact:
        if source_override is not None:
            raise SourceJoinError(
                f"{triple_id}: stale source override; exact join now succeeds"
            )
        return _resolve_source_candidates(
            exact, triple_id=triple_id, match_kind="exact"
        )
    if not allow_normalized_fallback:
        if source_override is not None:
            return _resolve_source_override(triple, source_index, source_override)
        return {
            "gold_original": "",
            "join_status": "unresolved",
            "join_match_count": 0,
            "source_splits": "",
            "source_ids": "",
            "source_paths": "",
            "source_locations": "",
        }
    normalized_key = (
        normalize_whitespace(premise),
        normalize_whitespace(hypothesis),
    )
    normalized_invalid = source_index.invalid_normalized.get(normalized_key, [])
    if normalized_invalid:
        details = [
            f"{record.split}:{record.source_id or record.line_number}"
            for record in normalized_invalid
        ]
        raise SourceJoinError(
            f"{triple['triple_id']}: normalized ++ match includes unlabeled "
            f"source provenance: {details}"
        )
    normalized = source_index.normalized.get(normalized_key, [])
    if normalized:
        if source_override is not None:
            raise SourceJoinError(
                f"{triple_id}: stale source override; normalized join now succeeds"
            )
        return _resolve_source_candidates(
            normalized,
            triple_id=triple_id,
            match_kind="normalized",
            require_one_raw_pair=True,
        )
    if source_override is not None:
        return _resolve_source_override(triple, source_index, source_override)
    return {
        "gold_original": "",
        "join_status": "unresolved",
        "join_match_count": 0,
        "source_splits": "",
        "source_ids": "",
        "source_paths": "",
        "source_locations": "",
    }


def classify_triple(triple: Mapping[str, Any]) -> dict[str, Any]:
    transformed = str(triple["transformed_signature"])
    original = str(triple.get("gold_original", ""))
    candidate = transformed == "(C,C,E)"
    transformed_all_u = transformed == "(N,N,N)"
    full_signature = (
        _signature(
            (
                original,
                str(triple["gold_neg_premise"]),
                str(triple["gold_neg_hypothesis"]),
                str(triple["gold_both_negative"]),
            )
        )
        if original
        else ""
    )
    if full_signature == "(E,C,C,E)":
        square_class = "square_valid"
    elif full_signature == "(N,N,N,N)":
        square_class = "all_u"
    elif not original and candidate:
        square_class = "candidate_square_unjoined"
    elif not original:
        square_class = "unjoined"
    elif original == "E":
        square_class = "base_ent_non_square"
    else:
        square_class = "other_base"
    return {
        "candidate_square": int(candidate),
        "transformed_all_u": int(transformed_all_u),
        "full_signature": full_signature,
        "square_class": square_class,
        "is_square_valid": int(square_class == "square_valid"),
        "is_all_u_square": int(full_signature == "(N,N,N,N)"),
    }


def materialize_triple_rows(triple: Mapping[str, Any]) -> list[dict[str, Any]]:
    texts = {
        "++": (
            str(triple["premise_positive"]),
            str(triple["hypothesis_positive"]),
            str(triple.get("gold_original", "")),
        ),
        "-+": (
            str(triple["premise_negative"]),
            str(triple["hypothesis_positive"]),
            str(triple["gold_neg_premise"]),
        ),
        "+-": (
            str(triple["premise_positive"]),
            str(triple["hypothesis_negative"]),
            str(triple["gold_neg_hypothesis"]),
        ),
        "--": (
            str(triple["premise_negative"]),
            str(triple["hypothesis_negative"]),
            str(triple["gold_both_negative"]),
        ),
    }
    rows: list[dict[str, Any]] = []
    for cell in FULL_CELL_ORDER:
        premise, hypothesis, gold = texts[cell]
        p_i, p_c = CELL_VALUES[cell]
        rho = p_i * p_c
        cell_id = CELL_ID_TAG[cell]
        sample_id = f"section62_{str(triple['triple_id'])}_{cell_id}"
        donor_cell = RHO_SAME_DONOR[cell]
        source_premise, source_hypothesis, source_gold = texts[donor_cell]
        p_i_src, p_c_src = CELL_VALUES[donor_cell]
        rho_src = p_i_src * p_c_src
        if rho_src != rho:
            raise AssertionError("Portability donor must preserve rho")
        source_prompt = build_prompt_with_spans([source_premise], source_hypothesis)
        source_tfu = GOLD_TO_TFU.get(source_gold, "")
        raw_lines = str(triple["raw_lines"]).split("|")
        base_source_offset = CELL_SOURCE_OFFSET.get(cell)
        source_offset = CELL_SOURCE_OFFSET.get(donor_cell)
        base_source_row_index = "" if base_source_offset is None else int(triple["group_index"]) * 3 + base_source_offset
        source_row_index = "" if source_offset is None else int(triple["group_index"]) * 3 + source_offset
        base_source_line_number = "" if base_source_offset is None else raw_lines[base_source_offset]
        source_line_number = "" if source_offset is None else raw_lines[source_offset]
        prompt = build_prompt_with_spans([premise], hypothesis)
        tfu = GOLD_TO_TFU.get(gold, "")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "base_event_id": triple["triple_id"],
            "triple_id": triple["triple_id"],
            "square_id": triple["triple_id"],
            "dataset": triple["corpus"],
            "corpus": triple["corpus"],
            "experiment": "section62_naturalistic_transfer",
            "split": "test",
            "target_var": "rho",
            "control_type": cell,
            "analysis_stratum": triple["square_class"],
            "cell": cell,
            "p_i": p_i,
            "cell_id": cell_id,
            "base_source_row_index": base_source_row_index,
            "base_source_line_number": base_source_line_number,
            "source_row_index": source_row_index,
            "source_line_number": source_line_number,
            "p_c": p_c,
            "rho": rho,
            "p_i_base": p_i,
            "p_c_base": p_c,
            "rho_base": rho,
            "negation_count": int(p_i == -1) + int(p_c == -1),
            "p_i_src": p_i_src,
            "p_c_src": p_c_src,
            "rho_src": rho_src,
            "negation_location": CELL_NEGATION_LOCATION[cell],
            "premise": premise,
            "hypothesis": hypothesis,
            "assumption": premise,
            "claim": hypothesis,
            "prompt": prompt.prompt,
            "expected_label": tfu,
            "gold_label": gold,
            "gold_label_code": gold,
            "target_label": tfu,
            "base_label": tfu,
            "base_prompt": prompt.prompt,
            "base_assumption": premise,
            "base_claim": hypothesis,
            "base_site": "claim_final",
            "n_assumptions": 1,
            "expected_R_sign": rho,
            "source_label": source_tfu,
            "source_gold_label_code": source_gold,
            "source_prompt": source_prompt.prompt,
            "source_assumption": source_premise,
            "source_claim": source_hypothesis,
            "source_site": "claim_final",
            "rho_expected_label": "T" if rho == 1 else "F",
            "rho_gold_consistent": int(
                bool(tfu) and tfu == ("T" if rho == 1 else "F")
            ),
            "transformed_signature": triple["transformed_signature"],
            "gold_signature_3": triple["transformed_signature"],
            "full_signature": triple["full_signature"],
            "gold_signature_4": triple["full_signature"],
            "square_class": triple["square_class"],
            "is_square_valid": triple["is_square_valid"],
            "transformed_all_u": triple["transformed_all_u"],
            "is_all_u_square": triple["is_all_u_square"],
            "join_status": triple["join_status"],
            "join_match_count": triple["join_match_count"],
            "source_splits": triple["source_splits"],
            "join_source_ids": triple["source_ids"],
            "source_is_reconstructed": int(donor_cell == "++"),
            "join_source_locations": triple["source_locations"],
            "is_reconstructed": int(cell == "++"),
            "join_source_paths": triple["source_paths"],
            "join_override_used": int(triple.get("join_override_used", 0)),
            "join_override_source_split": triple.get("join_override_source_split", ""),
            "join_override_pair_id": triple.get("join_override_pair_id", ""),
            "join_override_anchor_sides": triple.get("join_override_anchor_sides", ""),
            "join_override_clean_pair_sha256": triple.get("join_override_clean_pair_sha256", ""),
            "join_override_source_pair_sha256": triple.get("join_override_source_pair_sha256", ""),
            "join_override_expected_gold": triple.get("join_override_expected_gold", ""),
            "join_override_reason": triple.get("join_override_reason", ""),
            "rho_same_donor_cell": donor_cell,
            "portability_donor_cell": donor_cell,
            "portability_donor_sample_id": (
                f"section62_{str(triple['triple_id'])}_{CELL_ID_TAG[donor_cell]}"
            ),
            "rho_same_donor_sample_id": (
                f"section62_{str(triple['triple_id'])}_{CELL_ID_TAG[donor_cell]}"
            ),
            "portability_group": (
                "negation_count_parity"
                if cell in {"++", "--"}
                else "negation_position"
            ),
        }
        add_span_columns(row, "base", prompt.spans)
        add_span_columns(row, "source", source_prompt.spans)
        rows.append(row)
    return rows


def _flatten_triple(triple: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "corpus",
        "triple_id",
        "group_index",
        "raw_indices",
        "raw_lines",
        "premise_positive",
        "premise_negative",
        "hypothesis_positive",
        "hypothesis_negative",
        "gold_original",
        "gold_neg_premise",
        "gold_neg_hypothesis",
        "gold_both_negative",
        "transformed_signature",
        "full_signature",
        "candidate_square",
        "transformed_all_u",
        "square_class",
        "is_square_valid",
        "is_all_u_square",
        "source_paths",
        "join_status",
        "join_match_count",
        "source_splits",
        "source_ids",
        "source_locations",
        "join_override_used",
        "join_override_source_split",
        "join_override_pair_id",
        "join_override_anchor_sides",
        "join_override_clean_pair_sha256",
        "join_override_source_pair_sha256",
        "join_override_expected_gold",
        "join_override_reason",
    )
    return {key: triple.get(key, "") for key in keys}


def prepare_corpus(
    *,
    corpus: str,
    negation_path: str | Path,
    source_paths: Mapping[str, str | Path],
    expected_triples: int | None = 500,
    quarantine: Mapping[str, str] | None = None,
    require_complete_joins: bool = False,
    allow_normalized_fallback: bool = False,
    source_join_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    verify_upstream_blob: bool = False,
) -> PreparedCorpus:
    corpus = corpus.upper()
    overrides = {
        str(triple_id): spec
        for triple_id, spec in (source_join_overrides or {}).items()
    }
    locator_owners: dict[tuple[str, str], str] = {}
    expected_id_prefix = f"{corpus.lower()}_"
    for raw_triple_id, spec in overrides.items():
        triple_id = str(raw_triple_id)
        suffix = triple_id.removeprefix(expected_id_prefix)
        if (
            not triple_id.startswith(expected_id_prefix)
            or len(suffix) != 4
            or not suffix.isdigit()
        ):
            raise SourceJoinError(
                f"{corpus}: invalid source override triple id {triple_id!r}"
            )
        if not isinstance(spec, Mapping):
            raise SourceJoinError(
                f"{triple_id}: source override must be a mapping"
            )
        actual_fields = set(spec)
        if actual_fields != SOURCE_JOIN_OVERRIDE_FIELDS:
            raise SourceJoinError(
                f"{triple_id}: source override fields must be exactly "
                f"{sorted(SOURCE_JOIN_OVERRIDE_FIELDS)}"
            )
        locator = (
            str(spec["source_split"]).strip(),
            str(spec["pair_id"]).strip(),
        )
        if locator in locator_owners:
            raise SourceJoinError(
                f"{corpus}: source override locator {locator[0]}:{locator[1]} "
                f"is reused by {locator_owners[locator]} and {triple_id}"
            )
        locator_owners[locator] = triple_id
    expected_blob = UPSTREAM_GIT_BLOBS.get(corpus, "")
    actual_blob = git_blob_sha1_file(negation_path)
    if verify_upstream_blob and expected_blob and actual_blob != expected_blob:
        raise Section62DataError(
            f"{corpus}: negation file Git blob {actual_blob} does not match "
            f"pinned upstream blob {expected_blob}"
        )
    negation_rows = read_negation_rows(negation_path, corpus)
    grouped, quarantined = group_negation_triples(
        negation_rows,
        corpus,
        expected_triples=expected_triples,
        quarantine=quarantine,
    )
    source_index = load_source_index(corpus, source_paths)
    triples: list[dict[str, Any]] = []
    for grouped_triple in grouped:
        triple = dict(grouped_triple)
        triple.update(
            join_original_pair(
                triple,
                source_index,
                allow_normalized_fallback=allow_normalized_fallback,
                source_override=overrides.get(str(triple["triple_id"])),
            )
        )
        triple.update(classify_triple(triple))
        triples.append(triple)

    applied_override_ids = sorted(
        str(triple["triple_id"])
        for triple in triples
        if triple["join_status"] == "explicit_source_override"
    )
    unused_overrides = sorted(set(overrides) - set(applied_override_ids))
    if unused_overrides:
        raise SourceJoinError(
            f"{corpus}: source join overrides were not applied: {unused_overrides}"
        )

    unresolved = [
        triple["triple_id"]
        for triple in triples
        if triple["join_status"] == "unresolved"
    ]
    if require_complete_joins and unresolved:
        preview = ", ".join(unresolved[:10])
        suffix = "..." if len(unresolved) > 10 else ""
        raise SourceJoinError(
            f"{corpus}: {len(unresolved)} reconstructed ++ pairs were unresolved: "
            f"{preview}{suffix}"
        )

    all_rows = [
        row
        for triple in triples
        for row in materialize_triple_rows(triple)
    ]
    square_rows = [row for row in all_rows if int(row["is_square_valid"]) == 1]
    all_u_rows = [
        row
        for row in all_rows
        if int(row["is_all_u_square"]) == 1
    ]

    transformed_signatures = Counter(
        str(triple["transformed_signature"]) for triple in triples
    )
    full_signatures = Counter(
        str(triple["full_signature"])
        for triple in triples
        if triple["full_signature"]
    )
    join_statuses = Counter(str(triple["join_status"]) for triple in triples)
    classes = Counter(str(triple["square_class"]) for triple in triples)
    raw_triples = len(negation_rows) // 3
    valid_triples = sum(int(triple["is_square_valid"]) for triple in triples)
    candidate_triples = sum(int(triple["candidate_square"]) for triple in triples)
    joined_triples = sum(bool(triple["gold_original"]) for triple in triples)
    joined_entailment_triples = sum(
        int(triple["gold_original"] == "E") for triple in triples
    )
    valid_without_overrides = sum(
        int(triple["is_square_valid"])
        for triple in triples
        if triple["join_status"] != "explicit_source_override"
    )
    applied_override_records = [
        {
            "triple_id": triple["triple_id"],
            "source_split": triple.get("join_override_source_split", ""),
            "pair_id": triple.get("join_override_pair_id", ""),
            "anchor_sides": triple.get("join_override_anchor_sides", ""),
            "clean_pair_sha256": triple.get("join_override_clean_pair_sha256", ""),
            "source_pair_sha256": triple.get("join_override_source_pair_sha256", ""),
            "expected_gold": triple.get("join_override_expected_gold", ""),
            "reason": triple.get("join_override_reason", ""),
        }
        for triple in triples
        if triple["join_status"] == "explicit_source_override"
    ]
    manifest = {
        "corpus": corpus,
        "input": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_revision": UPSTREAM_REVISION,
            "expected_upstream_negation_git_blob": expected_blob,
            "actual_negation_git_blob": actual_blob,
            "upstream_negation_git_blob_matches": actual_blob == expected_blob,
            "negation_path": str(negation_path),
            "negation_sha256": sha256_file(negation_path),
            "negation_bytes": Path(negation_path).stat().st_size,
            "negation_encoding": UPSTREAM_TEXT_ENCODINGS.get(corpus, "utf-8"),
            "negation_rows": len(negation_rows),
            "source": source_index.manifest,
        },
        "cell_order": list(FULL_CELL_ORDER),
        "transformed_cell_order": list(NEGATED_CELL_ORDER),
        "signature_label_order": ["E", "C", "N"],
        "join_policy": {
            "default": "exact_raw_text",
            "normalized_fallback_enabled": allow_normalized_fallback,
            "normalized_fallback_version": (
                JOIN_NORMALIZATION_VERSION if allow_normalized_fallback else None
            ),
            "duplicate_policy": "same-label accepted; conflicts fail",
            "unlabeled_match_policy": "fail",
            "explicit_override_enabled": bool(overrides),
            "explicit_override_mode": "split_pairID_hash_and_anchor",
            "explicit_override_pair_hash_version": SOURCE_PAIR_HASH_VERSION,
            "explicit_override_anchor_normalization": JOIN_NORMALIZATION_VERSION,
            "explicit_override_stale_entries_fail": True,
            "explicit_override_unused_entries_fail": True,
            "explicit_override_labels_read_from_source": True,
            "require_complete_joins": require_complete_joins,
        },
        "expected_triples": expected_triples,
        "quarantine_policy": {
            "mode": "explicit_fail_loud",
            "unused_entries_fail": True,
        },
        "counts": {
            "raw_triples": raw_triples,
            "structural_valid_triples": len(triples),
            "raw_rows": len(negation_rows),
            "total_triples": raw_triples,
            "quarantined_triples": len(quarantined),
            "candidate_cce_triples": candidate_triples,
            "square_valid_triples": valid_triples,
            "structural_invalid_quarantined_triples": len(quarantined),
            "joined_original_triples": joined_triples,
            "joined_original_entailment_triples": joined_entailment_triples,
            "unresolved_original_triples": len(unresolved),
            "source_join_overrides_requested": len(overrides),
            "source_join_overrides_applied": len(applied_override_ids),
            "source_join_overrides_unused": len(unused_overrides),
            "square_valid_without_overrides_triples": valid_without_overrides,
            "square_valid_rows": len(square_rows),
            "transformed_all_u_triples": sum(
                int(triple["transformed_all_u"]) for triple in triples
            ),
            "all_u_square_triples": sum(
                int(triple["is_all_u_square"]) for triple in triples
            ),
            "all_u_rows": len(all_u_rows),
            "all_output_rows": len(all_rows),
        },
        "retention_chain": {
            "raw_triples": raw_triples,
            "structural_valid": len(triples),
            "candidate_cce": candidate_triples,
            "candidate_cce_with_joined_pp_entailment": valid_triples,
            "square_valid": valid_triples,
            "behavioral_pass": None,
            "baseline_correct": None,
        },
        "square_valid_retention_rate_of_raw": (
            valid_triples / raw_triples if raw_triples else 0.0
        ),
        "applied_source_join_overrides": applied_override_records,
        "join_status_counts": dict(sorted(join_statuses.items())),
        "square_class_counts": dict(sorted(classes.items())),
        "transformed_signature_counts": dict(
            sorted(transformed_signatures.items())
        ),
        "full_signature_counts": dict(sorted(full_signatures.items())),
        "quarantined": quarantined,
    }
    return PreparedCorpus(
        corpus=corpus,
        triples=[_flatten_triple(triple) for triple in triples],
        all_rows=all_rows,
        square_rows=square_rows,
        all_u_rows=all_u_rows,
        manifest=manifest,
    )
