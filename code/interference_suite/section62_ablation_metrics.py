"""Pure aggregation and reporting for Section 6.2 ablation outputs."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from .section62_data import FULL_CELL_ORDER, RHO_SAME_DONOR


ZERO_CONDITIONS = ("rho_zero", "m_zero", "rand_zero")
PORTABILITY_CONDITION = "rho_same_cross_cell"
METRIC_FIELDS = ("R", "M_TF", "G_U", "gold_margin_3", "U_gap")
EXPECTED_STRATA = ("square_valid", "all_u")
PORTABILITY_GROUP_CELLS = {
    "negation_count_parity": ("++", "--"),
    "negation_position": ("-+", "+-"),
}


class Section62AblationMetricsError(ValueError):
    """Raised when intervention rows cannot support the registered summaries."""


def _site(row: Mapping[str, Any], *, context: str) -> str:
    alias = str(row.get("site_alias", "")).strip()
    ordinary = str(row.get("site", "")).strip()
    if alias and ordinary and alias != ordinary:
        raise Section62AblationMetricsError(
            f"{context} has conflicting site_alias={alias!r} and site={ordinary!r}"
        )
    value = alias or ordinary
    if not value:
        raise Section62AblationMetricsError(f"{context} has no intervention site")
    return value


def _integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise Section62AblationMetricsError(
            f"{context} must be an integer, found {value!r}"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise Section62AblationMetricsError(
            f"{context} must be an integer, found {value!r}"
        ) from error
    if not isfinite(numeric) or not numeric.is_integer():
        raise Section62AblationMetricsError(
            f"{context} must be an integer, found {value!r}"
        )
    return int(numeric)


def _seed(value: Any, *, context: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _integer(value, context=context)


def _binary(value: Any, *, context: str) -> int:
    result = _integer(value, context=context)
    if result not in {0, 1}:
        raise Section62AblationMetricsError(
            f"{context} must be binary, found {value!r}"
        )
    return result


def _optional_float(value: Any, *, context: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Section62AblationMetricsError(
            f"{context} must be numeric, found {value!r}"
        ) from error
    if not isfinite(result):
        raise Section62AblationMetricsError(
            f"{context} must be finite, found {value!r}"
        )
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Section62AblationMetricsError("Cannot average an empty sequence")
    return sum(values) / len(values)


def _identity(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    context = f"intervention row {index}"
    required = (
        "sample_id",
        "model_name",
        "layer",
        "analysis_stratum",
        "cell",
        "condition",
        "is_correct",
    )
    missing = [
        field
        for field in required
        if field not in row or str(row[field]).strip() == ""
    ]
    if missing:
        raise Section62AblationMetricsError(
            f"{context} is missing fields: {missing}"
        )
    stratum = str(row["analysis_stratum"])
    cell = str(row["cell"])
    if stratum not in EXPECTED_STRATA:
        raise Section62AblationMetricsError(
            f"{context} has unsupported stratum {stratum!r}"
        )
    if cell not in FULL_CELL_ORDER:
        raise Section62AblationMetricsError(
            f"{context} has unsupported cell {cell!r}"
        )
    condition = str(row["condition"])
    allowed = set(ZERO_CONDITIONS) | {PORTABILITY_CONDITION}
    if condition not in allowed:
        raise Section62AblationMetricsError(
            f"{context} has unsupported condition {condition!r}"
        )
    return {
        "sample_id": str(row["sample_id"]),
        "model_name": str(row["model_name"]),
        "site_alias": _site(row, context=context),
        "layer": _integer(row["layer"], context=f"{context} layer"),
        "analysis_stratum": stratum,
        "cell": cell,
        "condition": condition,
        "random_seed": _seed(
            row.get("random_seed"),
            context=f"{context} random_seed",
        ),
    }


def _validate_metric_row(row: Mapping[str, Any], *, index: int) -> None:
    context = f"intervention row {index}"
    _binary(row["is_correct"], context=f"{context} is_correct")
    for field in METRIC_FIELDS:
        value = _optional_float(row.get(field), context=f"{context} {field}")
        if field == "M_TF":
            stratum = str(row["analysis_stratum"])
            if stratum == "square_valid" and value is None:
                raise Section62AblationMetricsError(
                    f"{context} square_valid requires M_TF"
                )
            if stratum == "all_u" and value is not None:
                raise Section62AblationMetricsError(
                    f"{context} all_u requires blank M_TF"
                )
        elif value is None:
            raise Section62AblationMetricsError(
                f"{context} requires metric {field}"
            )
    for field in ("base_is_correct",) + tuple(
        f"base_{metric}" for metric in METRIC_FIELDS
    ):
        if field not in row:
            continue
        if field == "base_is_correct":
            if str(row[field]).strip() != "":
                _binary(row[field], context=f"{context} {field}")
        else:
            value = _optional_float(
                row[field],
                context=f"{context} {field}",
            )
            if field == "base_M_TF":
                stratum = str(row["analysis_stratum"])
                if stratum == "square_valid" and value is None:
                    raise Section62AblationMetricsError(
                        f"{context} square_valid requires numeric base_M_TF"
                    )
                if stratum == "all_u" and value is not None:
                    raise Section62AblationMetricsError(
                        f"{context} all_u requires blank base_M_TF"
                    )


def _variant_key(identity: Mapping[str, Any]) -> tuple[str, int | None]:
    return str(identity["condition"]), identity["random_seed"]


def _point_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        identity["model_name"],
        identity["site_alias"],
        identity["layer"],
        identity["analysis_stratum"],
        identity["cell"],
    )


def _profile_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        identity["model_name"],
        identity["site_alias"],
        identity["layer"],
    )


def validate_condition_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_random_seeds: Iterable[int] = (0, 1, 2),
    expected_strata: Iterable[str] = EXPECTED_STRATA,
) -> dict[str, Any]:
    """Validate exact zero-condition/seed coverage and optional portability rows."""

    if not rows:
        raise Section62AblationMetricsError("Intervention rows are empty")
    seeds = tuple(int(seed) for seed in expected_random_seeds)
    if not seeds or len(seeds) != len(set(seeds)):

        raise Section62AblationMetricsError(
            "expected_random_seeds must be non-empty and unique"
        )
    strata = tuple(str(stratum) for stratum in expected_strata)
    if (
        not strata
        or len(strata) != len(set(strata))
        or not set(strata) <= set(EXPECTED_STRATA)
    ):
        raise Section62AblationMetricsError(
            f"expected_strata must be a non-empty unique subset of "
            f"{EXPECTED_STRATA}, found {strata}"
        )

    identities: list[dict[str, Any]] = []
    seen_ids: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    zero_by_point: dict[
        tuple[Any, ...], dict[tuple[str, int | None], set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    portability_rows: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for index, row in enumerate(rows):
        identity = _identity(row, index=index)
        _validate_metric_row(row, index=index)
        identities.append(identity)
        condition = str(identity["condition"])
        seed = identity["random_seed"]
        if condition == "rand_zero":
            if seed not in set(seeds):
                raise Section62AblationMetricsError(
                    f"intervention row {index} has unexpected rand_zero seed {seed!r}"
                )
        elif seed is not None:
            raise Section62AblationMetricsError(
                f"intervention row {index} condition {condition!r} "
                "requires a blank random_seed"
            )

        duplicate_scope = (
            identity["model_name"],
            identity["site_alias"],
            identity["layer"],
            condition,
            seed,
        )
        sample_id = str(identity["sample_id"])
        if sample_id in seen_ids[duplicate_scope]:
            raise Section62AblationMetricsError(
                f"Duplicate sample_id {sample_id!r} within condition "
                f"{condition!r}, seed={seed!r}"
            )
        seen_ids[duplicate_scope].add(sample_id)

        if condition in ZERO_CONDITIONS:
            zero_by_point[_point_key(identity)][
                _variant_key(identity)
            ].add(sample_id)
        else:
            portability_rows.append((row, identity))

    required_variants = {
        ("rho_zero", None),
        ("m_zero", None),
        *(("rand_zero", seed) for seed in seeds),
    }
    if not zero_by_point:
        raise Section62AblationMetricsError("No zero-condition rows were supplied")
    for point, variants in zero_by_point.items():
        if set(variants) != required_variants:
            missing = sorted(
                required_variants - set(variants),
                key=lambda item: (item[0], -1 if item[1] is None else item[1]),
            )
            extra = sorted(
                set(variants) - required_variants,
                key=lambda item: (item[0], -1 if item[1] is None else item[1]),
            )
            raise Section62AblationMetricsError(
                f"Zero-condition coverage mismatch for {point}: "
                f"missing={missing}, extra={extra}"
            )
        sample_sets = list(variants.values())
        reference = sample_sets[0]
        if not reference:
            raise Section62AblationMetricsError(
                f"Zero-condition point {point} has no rows"
            )
        if any(value != reference for value in sample_sets[1:]):
            raise Section62AblationMetricsError(
                f"Zero-condition sample_id sets differ for {point}"
            )

    profile_points: dict[tuple[Any, ...], set[tuple[str, str]]] = defaultdict(set)
    for point in zero_by_point:
        profile_points[point[:3]].add((str(point[3]), str(point[4])))
    expected_points = {
        (stratum, cell)
        for stratum in strata
        for cell in FULL_CELL_ORDER
    }
    for profile, points in profile_points.items():
        if points != expected_points:
            raise Section62AblationMetricsError(
                f"Zero-condition profile {profile} has stratum/cells="
                f"{sorted(points)}, expected={sorted(expected_points)}"
            )

    _validate_portability_rows(
        portability_rows,
        expected_profiles=set(profile_points),
        expected_strata=strata,
    )
    return {
        "n_rows": len(rows),
        "n_zero_rows": sum(
            1 for identity in identities if identity["condition"] in ZERO_CONDITIONS
        ),
        "n_portability_rows": len(portability_rows),
        "expected_zero_conditions": list(ZERO_CONDITIONS),
        "expected_random_seeds": list(seeds),
        "expected_strata": list(strata),
        "n_profiles": len(profile_points),
        "profiles": [
            {
                "model_name": profile[0],
                "site_alias": profile[1],
                "layer": profile[2],
            }
            for profile in sorted(profile_points)
        ],
    }


def _validate_portability_rows(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    expected_profiles: set[tuple[Any, ...]],
    expected_strata: Sequence[str],
) -> None:
    if not rows:
        return
    by_scope: dict[
        tuple[Any, ...], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row, identity in rows:
        context = "portability row {}".format(identity.get("sample_id"))
        required = (
            "triple_id",
            "portability_group",
            "portability_donor_cell",
            "portability_donor_sample_id",
            "pair_eligible",
            "whole_square_eligible",
        )
        missing = [
            field
            for field in required
            if field not in row or str(row[field]).strip() == ""
        ]
        if missing:
            raise Section62AblationMetricsError(
                f"{context} is missing fields: {missing}"
            )
        group = str(row["portability_group"])
        cell = str(identity["cell"])
        expected_group = (
            "negation_count_parity"
            if cell in {"++", "--"}
            else "negation_position"
        )
        if group != expected_group:
            raise Section62AblationMetricsError(
                f"{context} has portability_group={group!r}, "
                f"expected={expected_group!r}"
            )
        if str(row["portability_donor_cell"]) != RHO_SAME_DONOR[cell]:
            raise Section62AblationMetricsError(
                f"{context} has the wrong portability donor cell"
            )
        if _binary(
            row["pair_eligible"], context=f"{context} pair_eligible"
        ) != 1:
            raise Section62AblationMetricsError(
                f"{context} is not pair-eligible"
            )
        _binary(
            row["whole_square_eligible"],
            context=f"{context} whole_square_eligible",
        )
        key = (
            *_profile_key(identity),
            identity["analysis_stratum"],
            str(row["triple_id"]),
            group,
        )
        if cell in by_scope[key]:
            raise Section62AblationMetricsError(
                f"Portability pair {key} has duplicate cell {cell!r}"
            )
        by_scope[key][cell] = row

    observed_scopes = {
        (key[0], key[1], key[2], key[3], key[5])
        for key in by_scope
    }
    expected_scopes = {
        (profile[0], profile[1], profile[2], stratum, group)
        for profile in expected_profiles
        for stratum in expected_strata
        for group in PORTABILITY_GROUP_CELLS
    }
    if observed_scopes != expected_scopes:
        raise Section62AblationMetricsError(
            "Portability profile/stratum/group coverage mismatch: "
            f"missing={sorted(expected_scopes - observed_scopes)}, "
            f"extra={sorted(observed_scopes - expected_scopes)}"
        )

    for key, cell_rows in by_scope.items():
        group = str(key[-1])
        expected_cells = set(PORTABILITY_GROUP_CELLS[group])
        if set(cell_rows) != expected_cells:
            raise Section62AblationMetricsError(
                f"Portability pair {key} has cells={sorted(cell_rows)}, "
                f"expected={sorted(expected_cells)}"
            )
        for cell, row in cell_rows.items():
            donor_id = str(row["portability_donor_sample_id"])
            other = cell_rows[RHO_SAME_DONOR[cell]]
            if donor_id != str(other["sample_id"]):
                sample_id = str(row["sample_id"])
                raise Section62AblationMetricsError(
                    f"Portability row {sample_id} does not point to "
                    "its within-condition reciprocal donor"
                )


def _consistent_optional_values(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    binary: bool = False,
) -> list[float] | None:
    present = [field in row and str(row.get(field, "")).strip() != "" for row in rows]
    if any(present) and not all(present):
        raise Section62AblationMetricsError(
            f"Aggregate group has partially missing {field}"
        )
    if not any(present):
        return None
    if binary:
        return [
            float(_binary(row[field], context=field))
            for row in rows
        ]
    return [
        float(_optional_float(row[field], context=field))
        for row in rows
    ]


def _aggregate_group(
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    correct = [
        float(_binary(row["is_correct"], context="is_correct"))
        for row in rows
    ]
    result: dict[str, Any] = {
        "model_name": identity["model_name"],
        "site_alias": identity["site_alias"],
        "layer": identity["layer"],
        "analysis_stratum": identity["analysis_stratum"],
        "cell": identity["cell"],
        "condition": identity["condition"],
        "random_seed": (
            "" if identity["random_seed"] is None else identity["random_seed"]
        ),
        "n": len(rows),
        "n_correct": int(sum(correct)),
        "accuracy": _mean(correct),
    }
    for field in METRIC_FIELDS:
        values = _consistent_optional_values(rows, field)
        result[f"mean_{field}"] = None if values is None else _mean(values)

    base_correct = _consistent_optional_values(
        rows,
        "base_is_correct",
        binary=True,
    )
    if base_correct is not None:
        result["base_accuracy"] = _mean(base_correct)
        result["delta_accuracy"] = result["accuracy"] - result["base_accuracy"]
    for field in METRIC_FIELDS:
        values = _consistent_optional_values(rows, f"base_{field}")
        if values is None:
            continue
        baseline_name = f"base_mean_{field}"
        delta_name = f"delta_mean_{field}"
        result[baseline_name] = _mean(values)
        condition_value = result[f"mean_{field}"]
        result[delta_name] = (
            None
            if condition_value is None
            else condition_value - result[baseline_name]
        )
    return result


def aggregate_intervention_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_random_seeds: Iterable[int] = (0, 1, 2),
    expected_strata: Iterable[str] = EXPECTED_STRATA,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate rows at the registered model/site/layer/stratum/cell grain."""

    if validate:
        validate_condition_coverage(
            rows,
            expected_random_seeds=expected_random_seeds,
            expected_strata=expected_strata,
        )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = _identity(row, index=index)
        key = (
            *_point_key(identity),
            identity["condition"],
            identity["random_seed"],
        )
        grouped[key].append(row)
        identities[key] = identity
    cell_order = {cell: index for index, cell in enumerate(FULL_CELL_ORDER)}
    keys = sorted(
        grouped,
        key=lambda key: (
            str(key[0]),
            str(key[1]),
            int(key[2]),
            EXPECTED_STRATA.index(str(key[3])),
            cell_order[str(key[4])],
            str(key[5]),
            -1 if key[6] is None else int(key[6]),
        ),
    )
    return [
        _aggregate_group(grouped[key], identities[key])
        for key in keys
    ]


def _seed_stats(
    values: Sequence[float | None],
    *,
    context: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    present = [value is not None for value in values]
    if any(present) and not all(present):
        raise Section62AblationMetricsError(
            f"Random seed aggregates have partially missing {context}"
        )
    if not any(present):
        return None, None, None, None
    numeric = [float(value) for value in values if value is not None]
    minimum = min(numeric)
    maximum = max(numeric)
    return _mean(numeric), minimum, maximum, maximum - minimum


def summarize_rand_zero(
    cell_aggregates: Sequence[Mapping[str, Any]],
    *,
    expected_random_seeds: Iterable[int] = (0, 1, 2),
    accuracy_spread_threshold: float = 0.02,
) -> dict[str, Any]:
    """Summarize seed-level random controls and make the preregistered trigger."""

    threshold = float(accuracy_spread_threshold)
    if not isfinite(threshold) or threshold < 0.0:
        raise Section62AblationMetricsError(
            "accuracy_spread_threshold must be finite and non-negative"
        )
    expected_seeds = tuple(int(seed) for seed in expected_random_seeds)
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_aggregates:
        if str(row.get("condition")) != "rand_zero":
            continue
        key = (
            str(row["model_name"]),
            str(row["site_alias"]),
            int(row["layer"]),
            str(row["analysis_stratum"]),
            str(row["cell"]),
        )
        grouped[key].append(row)
    if not grouped:
        raise Section62AblationMetricsError("No rand_zero cell aggregates found")

    by_cell: list[dict[str, Any]] = []
    trigger_cells: list[dict[str, Any]] = []
    metric_names = ("accuracy",) + tuple(
        f"mean_{field}" for field in METRIC_FIELDS
    )
    for key in sorted(grouped):
        values = grouped[key]
        seeds = tuple(sorted(_integer(row["random_seed"], context="random_seed") for row in values))
        if seeds != tuple(sorted(expected_seeds)):
            raise Section62AblationMetricsError(
                f"rand_zero seed coverage for {key} is {seeds}, "
                f"expected={tuple(sorted(expected_seeds))}"
            )
        output = {
            "model_name": key[0],
            "site_alias": key[1],
            "layer": key[2],
            "analysis_stratum": key[3],
            "cell": key[4],
            "n_seeds": len(values),
            "random_seeds": list(seeds),
        }
        by_seed = {
            _integer(row["random_seed"], context="random_seed"): row
            for row in values
        }
        for metric in metric_names:
            stats = _seed_stats(
                [by_seed[seed].get(metric) for seed in sorted(expected_seeds)],
                context=metric,
            )
            for suffix, value in zip(("mean", "min", "max", "spread"), stats):
                output[f"rand_{metric}_{suffix}"] = value
        by_cell.append(output)
        if float(output["rand_accuracy_spread"]) >= threshold:
            trigger_cells.append(
                {
                    "model_name": key[0],
                    "site_alias": key[1],
                    "layer": key[2],
                    "analysis_stratum": key[3],
                    "cell": key[4],
                    "rand_accuracy_spread": output["rand_accuracy_spread"],
                }
            )
    return {
        "accuracy_spread_threshold": threshold,
        "need_extra_seeds": bool(trigger_cells),
        "trigger_cells": trigger_cells,
        "by_cell": by_cell,
    }


def build_confirmatory_summary(
    cell_aggregates: Sequence[Mapping[str, Any]],
    rand_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build cell effects and worst-cell confirmatory headlines."""

    deterministic: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in cell_aggregates:
        condition = str(row.get("condition"))
        if condition not in {"rho_zero", "m_zero"}:
            continue
        key = (
            str(row["model_name"]),
            str(row["site_alias"]),
            int(row["layer"]),
            str(row["analysis_stratum"]),
            str(row["cell"]),
        )
        if condition in deterministic[key]:
            raise Section62AblationMetricsError(
                f"Duplicate deterministic aggregate for {key}:{condition}"
            )
        deterministic[key][condition] = row
    random_by_key = {
        (
            str(row["model_name"]),
            str(row["site_alias"]),
            int(row["layer"]),
            str(row["analysis_stratum"]),
            str(row["cell"]),
        ): row
        for row in rand_summary.get("by_cell", [])
    }
    if set(deterministic) != set(random_by_key):
        raise Section62AblationMetricsError(
            "Deterministic and random cell aggregate keys do not match"
        )

    by_cell: list[dict[str, Any]] = []
    for key in sorted(deterministic):
        conditions = deterministic[key]
        if set(conditions) != {"rho_zero", "m_zero"}:
            raise Section62AblationMetricsError(
                f"Missing deterministic condition for {key}"
            )
        rho = conditions["rho_zero"]
        m = conditions["m_zero"]
        rand = random_by_key[key]
        output: dict[str, Any] = {
            "model_name": key[0],
            "site_alias": key[1],
            "layer": key[2],
            "analysis_stratum": key[3],
            "cell": key[4],
            "rho_zero_accuracy": float(rho["accuracy"]),
            "m_zero_accuracy": float(m["accuracy"]),
            "rand_zero_accuracy_mean": float(rand["rand_accuracy_mean"]),
        }
        if key[3] == "square_valid":
            output["zero_NEx_accuracy"] = (
                float(rand["rand_accuracy_mean"]) - float(rho["accuracy"])
            )
            output["zero_NEx_margin"] = (
                float(rand["rand_mean_M_TF_mean"]) - float(rho["mean_M_TF"])
            )
            output["zero_NEx_gold_margin_3"] = (
                float(rand["rand_mean_gold_margin_3_mean"])
                - float(rho["mean_gold_margin_3"])
            )
            output["square_specificity_accuracy"] = (
                float(m["accuracy"]) - float(rho["accuracy"])
            )
            output["square_specificity_margin"] = (
                float(m["mean_M_TF"]) - float(rho["mean_M_TF"])
            )
        else:
            output["all_u_specificity_accuracy"] = (
                float(rho["accuracy"]) - float(m["accuracy"])
            )
            output["all_u_specificity_G_U"] = (
                float(rho["mean_G_U"]) - float(m["mean_G_U"])
            )
            output["all_u_specificity_gold_margin_3"] = (
                float(rho["mean_gold_margin_3"])
                - float(m["mean_gold_margin_3"])
            )
        by_cell.append(output)

    profiles = sorted(
        {
            (row["model_name"], row["site_alias"], row["layer"])
            for row in by_cell
        }
    )
    headline: list[dict[str, Any]] = []
    for model_name, site_alias, layer in profiles:
        square = [
            row
            for row in by_cell
            if row["model_name"] == model_name
            and row["site_alias"] == site_alias
            and row["layer"] == layer
            and row["analysis_stratum"] == "square_valid"
        ]
        all_u = [
            row
            for row in by_cell
            if row["model_name"] == model_name
            and row["site_alias"] == site_alias
            and row["layer"] == layer
            and row["analysis_stratum"] == "all_u"
        ]
        if len(square) != len(FULL_CELL_ORDER):
            raise Section62AblationMetricsError(
                f"Confirmatory headline for {(model_name, site_alias, layer)} "
                "requires four square_valid cells"
            )
        if all_u and len(all_u) != len(FULL_CELL_ORDER):
            raise Section62AblationMetricsError(
                f"Confirmatory headline for {(model_name, site_alias, layer)} "
                "has incomplete all_u coverage"
            )
        zero_nex_values = [float(row["zero_NEx_accuracy"]) for row in square]
        headline_row = {
            "model_name": model_name,
            "site_alias": site_alias,
            "layer": layer,
            "headline_worst_cell_min": min(zero_nex_values),
            "square_zero_NEx_accuracy_worst_cell_min": min(zero_nex_values),
            "square_zero_NEx_margin_worst_cell_min": min(
                float(row["zero_NEx_margin"]) for row in square
            ),
            "square_specificity_accuracy_worst_cell_min": min(
                float(row["square_specificity_accuracy"])
                for row in square
            ),
        }
        if all_u:
            headline_row.update(
                {
                    "all_u_specificity_accuracy_worst_cell_min": min(
                        float(row["all_u_specificity_accuracy"])
                        for row in all_u
                    ),
                    "all_u_specificity_G_U_worst_cell_min": min(
                        float(row["all_u_specificity_G_U"])
                        for row in all_u
                    ),
                    "all_u_specificity_gold_margin_3_worst_cell_min": min(
                        float(row["all_u_specificity_gold_margin_3"])
                        for row in all_u
                    ),
                }
            )
        headline.append(headline_row)
    return {"by_cell": by_cell, "headline": headline}


def _aggregate_portability_scope(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    flag = "pair_eligible" if scope == "pair_eligible" else "whole_square_eligible"
    selected = [
        row
        for row in rows
        if _binary(row[flag], context=flag) == 1
    ]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, row in enumerate(selected):
        identity = _identity(row, index=index)
        group = str(row["portability_group"])
        key = (
            *_profile_key(identity),
            identity["analysis_stratum"],
            group,
        )
        grouped[key].append(row)
        identities[key] = identity
    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        identity = identities[key]
        representative = {
            **identity,
            "cell": "all",
            "condition": PORTABILITY_CONDITION,
            "random_seed": None,
        }
        aggregate = _aggregate_group(grouped[key], representative)
        aggregate.pop("cell")
        aggregate["scope"] = scope
        aggregate["portability_group"] = key[-1]
        output.append(aggregate)
    return output


def _pair_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if _binary(row["pair_eligible"], context="pair_eligible") == 1
    ]
    pairs: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(eligible):
        identity = _identity(row, index=index)
        pair_id = str(
            row.get(
                "portability_pair_id",
                f"{row['triple_id']}:{row['portability_group']}",
            )
        )
        key = (
            *_profile_key(identity),
            identity["analysis_stratum"],
            str(row["portability_group"]),
            pair_id,
        )
        pairs[key].append(row)
    summarized: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for key, values in pairs.items():
        expected_cells = set(PORTABILITY_GROUP_CELLS[str(key[-2])])
        actual_cells = {str(row["cell"]) for row in values}
        if actual_cells != expected_cells or len(values) != 2:
            raise Section62AblationMetricsError(
                f"Eligible portability pair {key} has cells={sorted(actual_cells)}"
            )
        summarized[key[:-1]].append(
            int(all(_binary(row["is_correct"], context="is_correct") for row in values))
        )
    return [
        {
            "model_name": key[0],
            "site_alias": key[1],
            "layer": key[2],
            "analysis_stratum": key[3],
            "portability_group": key[4],
            "n_pairs": len(values),
            "n_pairs_all_correct": sum(values),
            "pair_accuracy": _mean([float(value) for value in values]),
        }
        for key, values in sorted(summarized.items())
    ]


def _whole_square_outcomes(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if _binary(
            row["whole_square_eligible"],
            context="whole_square_eligible",
        )
        == 1
    ]
    squares: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(selected):
        identity = _identity(row, index=index)
        key = (
            *_profile_key(identity),
            identity["analysis_stratum"],
            str(row["triple_id"]),
        )
        squares[key].append(row)
    summarized: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for key, values in squares.items():
        actual_cells = {str(row["cell"]) for row in values}
        if actual_cells != set(FULL_CELL_ORDER) or len(values) != len(FULL_CELL_ORDER):
            raise Section62AblationMetricsError(
                f"Whole-square portability subset {key} has "
                f"cells={sorted(actual_cells)}"
            )
        summarized[key[:-1]].append(
            int(all(_binary(row["is_correct"], context="is_correct") for row in values))
        )
    return [
        {
            "model_name": key[0],
            "site_alias": key[1],
            "layer": key[2],
            "analysis_stratum": key[3],
            "n_squares": len(values),
            "n_squares_all_correct": sum(values),
            "whole_square_accuracy": _mean(
                [float(value) for value in values]
            ),
        }
        for key, values in sorted(summarized.items())
    ]


def summarize_portability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize rho-same cross-cell portability by mechanism and strict subset."""

    portability = [
        row
        for row in rows
        if str(row.get("condition")) == PORTABILITY_CONDITION
    ]
    if not portability:
        return {
            "condition": PORTABILITY_CONDITION,
            "n_rows": 0,
            "by_group": [],
            "pair_outcomes": [],
            "whole_square_outcomes": [],
        }
    return {
        "condition": PORTABILITY_CONDITION,
        "n_rows": len(portability),
        "by_group": (
            _aggregate_portability_scope(portability, scope="pair_eligible")
            + _aggregate_portability_scope(
                portability,
                scope="whole_square_subset",
            )
        ),
        "pair_outcomes": _pair_outcomes(portability),
        "whole_square_outcomes": _whole_square_outcomes(portability),
    }


def summarize_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_random_seeds: Iterable[int] = (0, 1, 2),
    expected_strata: Iterable[str] = EXPECTED_STRATA,
    accuracy_spread_threshold: float = 0.02,
) -> dict[str, Any]:
    """Run the complete pure-data Section 6.2 ablation summary."""

    seeds = tuple(int(seed) for seed in expected_random_seeds)
    strata = tuple(str(stratum) for stratum in expected_strata)
    coverage = validate_condition_coverage(
        rows,
        expected_random_seeds=seeds,
        expected_strata=strata,
    )
    cell_aggregates = aggregate_intervention_rows(
        rows,
        expected_random_seeds=seeds,
        expected_strata=strata,
        validate=False,
    )
    random_summary = summarize_rand_zero(
        cell_aggregates,
        expected_random_seeds=seeds,
        accuracy_spread_threshold=accuracy_spread_threshold,
    )
    return {
        "coverage": coverage,
        "by_cell": cell_aggregates,
        "rand_zero": random_summary,
        "confirmatory": build_confirmatory_summary(
            cell_aggregates,
            random_summary,
        ),
        "portability": summarize_portability(rows),
    }

