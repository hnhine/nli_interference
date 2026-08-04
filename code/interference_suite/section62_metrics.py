"""Behavioral gates for Section 6.2 naturalistic polarity squares."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import isfinite
from typing import Any

from .section62_data import FULL_CELL_ORDER


EXPECTED_LABEL_BY_STRATUM = {
    "square_valid": {
        "++": "T",
        "-+": "F",
        "+-": "F",
        "--": "T",
    },
    "all_u": {cell: "U" for cell in FULL_CELL_ORDER},
}


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def _binary(value: Any, *, field: str, row_index: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Behavioral row {row_index} has non-binary {field}={value!r}"
        ) from error
    if numeric not in {0.0, 1.0}:
        raise ValueError(
            f"Behavioral row {row_index} has non-binary {field}={value!r}"
        )
    return int(numeric)


def _finite_float(value: Any, *, field: str, row_index: int) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Behavioral row {row_index} has non-numeric {field}={value!r}"
        ) from error
    if not isfinite(numeric):
        raise ValueError(
            f"Behavioral row {row_index} has non-finite {field}={value!r}"
        )
    return numeric


def behavioral_gate_summary(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    gate_threshold: float = 0.90,
    min_baseline_correct_per_cell: int = 50,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty Section 6.2 behavioral run")
    if not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("gate_threshold must lie in [0, 1]")
    if min_baseline_correct_per_cell < 1:
        raise ValueError("min_baseline_correct_per_cell must be positive")

    required = {
        "corpus",
        "triple_id",
        "analysis_stratum",
        "cell",
        "expected_label",
        "pred_label",
        "is_correct",
        "R",
        "U_gap",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Behavioral row {index} is missing fields: {missing}")
        stratum = str(row["analysis_stratum"])
        cell = str(row["cell"])
        if stratum not in EXPECTED_LABEL_BY_STRATUM:
            raise ValueError(
                f"Behavioral row {index} has unsupported "
                f"analysis_stratum={stratum!r}"
            )
        if cell not in FULL_CELL_ORDER:
            raise ValueError(f"Behavioral row {index} has unknown cell={cell!r}")
        expected_label = str(row["expected_label"])
        required_label = EXPECTED_LABEL_BY_STRATUM[stratum][cell]
        if expected_label != required_label:
            raise ValueError(
                f"Behavioral row {index} ({stratum}, {cell}) requires "
                f"expected_label={required_label!r}, found {expected_label!r}"
            )
        pred_label = str(row["pred_label"])
        if pred_label not in {"T", "F", "U"}:
            raise ValueError(
                f"Behavioral row {index} has unknown pred_label={pred_label!r}"
            )
        is_correct = _binary(
            row["is_correct"], field="is_correct", row_index=index
        )
        expected_is_correct = int(pred_label == expected_label)
        if is_correct != expected_is_correct:
            raise ValueError(
                f"Behavioral row {index} has is_correct={is_correct}, "
                f"but pred_label={pred_label!r} and "
                f"expected_label={expected_label!r}"
            )
        r_value = _finite_float(row["R"], field="R", row_index=index)
        _finite_float(row["U_gap"], field="U_gap", row_index=index)
        if stratum == "square_valid":
            if "is_tf_correct" not in row:
                raise ValueError(
                    f"Behavioral row {index} is missing fields: ['is_tf_correct']"
                )
            is_tf_correct = _binary(
                row["is_tf_correct"], field="is_tf_correct", row_index=index
            )
            pred_tf_label = "T" if r_value >= 0.0 else "F"
            expected_tf_correct = int(pred_tf_label == expected_label)
            if is_tf_correct != expected_tf_correct:
                raise ValueError(
                    f"Behavioral row {index} has "
                    f"is_tf_correct={is_tf_correct}, but R={r_value} implies "
                    f"pred_tf_label={pred_tf_label!r}"
                )
        for optional_numeric in ("M_TF", "gold_margin_3"):
            if row.get(optional_numeric) not in (None, ""):
                _finite_float(
                    row[optional_numeric],
                    field=optional_numeric,
                    row_index=index,
                )
        if row.get("global_top_in_TFU") not in (None, ""):
            _binary(
                row["global_top_in_TFU"],
                field="global_top_in_TFU",
                row_index=index,
            )

    cell_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    triple_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        corpus = str(row["corpus"])
        stratum = str(row["analysis_stratum"])
        cell = str(row["cell"])
        cell_groups[(corpus, stratum, cell)].append(row)
        triple_groups[(corpus, stratum, str(row["triple_id"]))].append(row)

    by_cell: list[dict[str, Any]] = []
    stratum_order = {
        name: index for index, name in enumerate(EXPECTED_LABEL_BY_STRATUM)
    }
    ordered_cell_groups = sorted(
        cell_groups.items(),
        key=lambda item: (
            item[0][0],
            stratum_order[item[0][1]],
            FULL_CELL_ORDER.index(item[0][2]),
        ),
    )
    for (corpus, stratum, cell), values in ordered_cell_groups:
        m_tf_values = [
            float(row["M_TF"])
            for row in values
            if row.get("M_TF") not in (None, "")
        ]
        gold_margin_values = [
            float(row["gold_margin_3"])
            for row in values
            if row.get("gold_margin_3") not in (None, "")
        ]
        global_top_values = [
            float(row["global_top_in_TFU"])
            for row in values
            if row.get("global_top_in_TFU") not in (None, "")
        ]
        tf_rows = [
            row for row in values if str(row["expected_label"]) in {"T", "F"}
        ]
        by_cell.append(
            {
                "model_name": model_name,
                "corpus": corpus,
                "analysis_stratum": stratum,
                "cell": cell,
                "n": len(values),
                "n_correct": sum(
                    int(float(row["is_correct"])) for row in values
                ),
                "accuracy": mean(float(row["is_correct"]) for row in values),
                "tf_accuracy": mean(
                    float(row["is_tf_correct"]) for row in tf_rows
                ),
                "mean_R": mean(float(row["R"]) for row in values),
                "mean_M_TF": mean(m_tf_values),
                "mean_gold_margin_3": mean(gold_margin_values),
                "mean_U_gap": mean(float(row["U_gap"]) for row in values),
                "pred_U_rate": mean(
                    float(str(row["pred_label"]) == "U") for row in values
                ),
                "global_top_in_TFU_rate": mean(global_top_values),
            }
        )

    by_square: list[dict[str, Any]] = []
    expected_cells = set(FULL_CELL_ORDER)
    ordered_triple_groups = sorted(
        triple_groups.items(),
        key=lambda item: (
            item[0][0],
            stratum_order[item[0][1]],
            item[0][2],
        ),
    )
    for (corpus, stratum, triple_id), values in ordered_triple_groups:
        cells = [str(row["cell"]) for row in values]
        if len(cells) != len(set(cells)):
            raise ValueError(
                f"{corpus}:{triple_id}:{stratum} has duplicate cells: {cells}"
            )
        if set(cells) != expected_cells:
            raise ValueError(
                f"{corpus}:{triple_id}:{stratum} cells={sorted(cells)}, "
                f"expected={list(FULL_CELL_ORDER)}"
            )
        ordered = sorted(values, key=lambda row: FULL_CELL_ORDER.index(str(row["cell"])))
        tf_rows = [
            row for row in ordered if str(row["expected_label"]) in {"T", "F"}
        ]
        by_square.append(
            {
                "model_name": model_name,
                "corpus": corpus,
                "analysis_stratum": stratum,
                "triple_id": triple_id,
                "n_cells": len(ordered),
                "n_correct": sum(
                    int(float(row["is_correct"])) for row in ordered
                ),
                "all_cells_correct": int(
                    all(float(row["is_correct"]) == 1.0 for row in ordered)
                ),
                "all_tf_correct": (
                    int(all(float(row["is_tf_correct"]) == 1.0 for row in tf_rows))
                    if tf_rows
                    else None
                ),
                "contains_join_override": int(
                    any(int(row.get("join_override_used", 0) or 0) == 1 for row in ordered)
                ),
                "cell_predictions": "|".join(
                    f"{row['cell']}:{row['pred_label']}" for row in ordered
                ),
            }
        )

    by_stratum: list[dict[str, Any]] = []
    stratum_keys = sorted(
        {(row["corpus"], row["analysis_stratum"]) for row in by_cell},
        key=lambda item: (item[0], stratum_order[item[1]]),
    )
    for corpus, stratum in stratum_keys:
        cells = [
            row
            for row in by_cell
            if row["corpus"] == corpus and row["analysis_stratum"] == stratum
        ]
        if {row["cell"] for row in cells} != expected_cells:
            raise ValueError(
                f"{corpus}:{stratum} does not contain all four polarity cells"
            )
        squares = [
            row
            for row in by_square
            if row["corpus"] == corpus and row["analysis_stratum"] == stratum
        ]
        min_accuracy = min(float(row["accuracy"]) for row in cells)
        min_correct = min(int(row["n_correct"]) for row in cells)
        tf_accuracies = [
            float(row["tf_accuracy"])
            for row in cells
            if row["tf_accuracy"] is not None
        ]
        strict_eligible = min_accuracy >= gate_threshold
        secondary_eligible = min_correct >= min_baseline_correct_per_cell
        if strict_eligible:
            scope = "strict_gate_pass"
        elif secondary_eligible:
            scope = "baseline_correct_secondary_only"
        else:
            scope = "insufficient_baseline_correct_items"
        by_stratum.append(
            {
                "model_name": model_name,
                "corpus": corpus,
                "analysis_stratum": stratum,
                "n_rows": sum(int(row["n"]) for row in cells),
                "n_squares": len(squares),
                "accuracy": mean(float(row["is_correct"]) for row in rows if str(row["corpus"]) == corpus and str(row["analysis_stratum"]) == stratum),
                "min_cell_accuracy": min_accuracy,
                "min_cell_tf_accuracy": (
                    min(tf_accuracies) if tf_accuracies else None
                ),
                "whole_square_accuracy": mean(
                    float(row["all_cells_correct"]) for row in squares
                ),
                "min_baseline_correct_per_cell": min_correct,
                "required_baseline_correct_per_cell": min_baseline_correct_per_cell,
                "gate_threshold": gate_threshold,
                "strict_gate_eligible": strict_eligible,
                "secondary_baseline_correct_eligible": secondary_eligible,
                "interpretation_scope": scope,
                "n_squares_without_overrides": sum(
                    int(row["contains_join_override"] == 0) for row in squares
                ),
            }
        )

    return {
        "model_name": model_name,
        "gate_threshold": gate_threshold,
        "required_baseline_correct_per_cell": min_baseline_correct_per_cell,
        "n_rows": len(rows),
        "by_cell": by_cell,
        "by_square": by_square,
        "by_stratum": by_stratum,
        "square_gate": [
            row for row in by_stratum if row["analysis_stratum"] == "square_valid"
        ],
        "all_u_gate": [
            row for row in by_stratum if row["analysis_stratum"] == "all_u"
        ],
    }
