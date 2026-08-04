from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interference_suite.section62_ablation_metrics import (  # noqa: E402
    PORTABILITY_CONDITION,
    Section62AblationMetricsError,
    aggregate_intervention_rows,
    summarize_ablation,
    summarize_rand_zero,
    validate_condition_coverage,
)
from interference_suite.section62_data import (  # noqa: E402
    FULL_CELL_ORDER,
    RHO_SAME_DONOR,
)


def intervention_metrics(
    *,
    condition: str,
    stratum: str,
    seed: int | None,
) -> dict[str, object]:
    if condition == "rho_zero":
        m_tf = 1.0
        gold_margin = 0.5
        r_value = 0.25
    elif condition == "m_zero":
        m_tf = 3.0
        gold_margin = 2.0
        r_value = 1.25
    elif condition == "rand_zero":
        assert seed is not None
        m_tf = 4.0 + seed
        gold_margin = 3.0 + seed
        r_value = 2.0 + seed
    else:
        m_tf = 3.5
        gold_margin = 2.5
        r_value = 1.5
    return {
        "R": r_value,
        "M_TF": m_tf if stratum == "square_valid" else "",
        "G_U": -1.0 + (0.1 * (seed or 0)),
        "gold_margin_3": gold_margin,
        "U_gap": -0.75,
    }


def condition_correct(
    *,
    condition: str,
    stratum: str,
    cell: str,
    triple_index: int,
    seed: int | None,
) -> int:
    if condition == "rho_zero":
        return int(triple_index == 0 or stratum == "all_u")
    if condition == "m_zero":
        return int(stratum == "square_valid" or triple_index == 0)
    if condition == "rand_zero":
        return int(
            not (
                stratum == "square_valid"
                and cell == "++"
                and seed == 1
                and triple_index == 1
            )
        )
    if condition == PORTABILITY_CONDITION:
        return int(
            triple_index == 0
            or cell in {"-+", "+-"}
        )
    raise AssertionError(condition)


def fixture_rows(
    *,
    random_seeds: tuple[int, ...] = (0, 1, 2),
    include_portability: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    variants = [
        ("rho_zero", None),
        ("m_zero", None),
        *((("rand_zero", seed) for seed in random_seeds)),
    ]
    for stratum in ("square_valid", "all_u"):
        for triple_index in range(2):
            triple_id = f"{stratum}_triple_{triple_index}"
            for cell in FULL_CELL_ORDER:
                sample_id = f"{triple_id}_{cell}"
                common: dict[str, object] = {
                    "sample_id": sample_id,
                    "model_name": "example/model",
                    "site_alias": "claim_final",
                    "layer": 2,
                    "analysis_stratum": stratum,
                    "cell": cell,
                    "triple_id": triple_id,
                    "base_is_correct": 1,
                    "base_R": 3.0,
                    "base_M_TF": 4.0 if stratum == "square_valid" else "",
                    "base_G_U": -2.0,
                    "base_gold_margin_3": 3.0,
                    "base_U_gap": -1.5,
                }
                for condition, seed in variants:
                    row = dict(common)
                    row.update(
                        {
                            "condition": condition,
                            "random_seed": "" if seed is None else seed,
                            "is_correct": condition_correct(
                                condition=condition,
                                stratum=stratum,
                                cell=cell,
                                triple_index=triple_index,
                                seed=seed,
                            ),
                            **intervention_metrics(
                                condition=condition,
                                stratum=stratum,
                                seed=seed,
                            ),
                        }
                    )
                    rows.append(row)

                if include_portability:
                    group = (
                        "negation_count_parity"
                        if cell in {"++", "--"}
                        else "negation_position"
                    )
                    donor_cell = RHO_SAME_DONOR[cell]
                    portability = dict(common)
                    portability.update(
                        {
                            "condition": PORTABILITY_CONDITION,
                            "random_seed": "",
                            "is_correct": condition_correct(
                                condition=PORTABILITY_CONDITION,
                                stratum=stratum,
                                cell=cell,
                                triple_index=triple_index,
                                seed=None,
                            ),
                            "portability_group": group,
                            "portability_pair_id": f"{triple_id}:{group}",
                            "portability_donor_cell": donor_cell,
                            "portability_donor_sample_id": (
                                f"{triple_id}_{donor_cell}"
                            ),
                            "pair_eligible": 1,
                            "whole_square_eligible": int(triple_index == 0),
                            **intervention_metrics(
                                condition=PORTABILITY_CONDITION,
                                stratum=stratum,
                                seed=None,
                            ),
                        }
                    )
                    rows.append(portability)
    return rows


def find_row(rows, **matches):
    found = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in matches.items())
    ]
    if len(found) != 1:
        raise AssertionError(f"Expected one row for {matches}, found {len(found)}")
    return found[0]


class AblationAggregationTests(unittest.TestCase):
    def test_full_summary_aggregates_baselines_random_seeds_and_headlines(self):
        result = summarize_ablation(
            fixture_rows(),
            expected_random_seeds=(0, 1, 2),
            accuracy_spread_threshold=0.5,
        )

        self.assertEqual(result["coverage"]["n_zero_rows"], 80)
        self.assertEqual(result["coverage"]["n_portability_rows"], 16)
        self.assertEqual(len(result["by_cell"]), 48)

        rho_square_pp = find_row(
            result["by_cell"],
            analysis_stratum="square_valid",
            cell="++",
            condition="rho_zero",
            random_seed="",
        )
        self.assertEqual(rho_square_pp["n"], 2)
        self.assertAlmostEqual(rho_square_pp["accuracy"], 0.5)
        self.assertAlmostEqual(rho_square_pp["mean_M_TF"], 1.0)
        self.assertAlmostEqual(rho_square_pp["base_accuracy"], 1.0)
        self.assertAlmostEqual(rho_square_pp["delta_accuracy"], -0.5)
        self.assertAlmostEqual(rho_square_pp["base_mean_M_TF"], 4.0)
        self.assertAlmostEqual(rho_square_pp["delta_mean_M_TF"], -3.0)

        rand_pp = find_row(
            result["rand_zero"]["by_cell"],
            analysis_stratum="square_valid",
            cell="++",
        )
        self.assertAlmostEqual(rand_pp["rand_accuracy_mean"], 5.0 / 6.0)
        self.assertAlmostEqual(rand_pp["rand_accuracy_min"], 0.5)
        self.assertAlmostEqual(rand_pp["rand_accuracy_max"], 1.0)
        self.assertAlmostEqual(rand_pp["rand_accuracy_spread"], 0.5)
        self.assertAlmostEqual(rand_pp["rand_mean_M_TF_mean"], 5.0)
        self.assertTrue(result["rand_zero"]["need_extra_seeds"])
        self.assertEqual(len(result["rand_zero"]["trigger_cells"]), 1)

        square_effect = find_row(
            result["confirmatory"]["by_cell"],
            analysis_stratum="square_valid",
            cell="++",
        )
        self.assertAlmostEqual(square_effect["zero_NEx_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(square_effect["zero_NEx_margin"], 4.0)
        self.assertAlmostEqual(
            square_effect["square_specificity_accuracy"],
            0.5,
        )

        all_u_effect = find_row(
            result["confirmatory"]["by_cell"],
            analysis_stratum="all_u",
            cell="++",
        )
        self.assertAlmostEqual(
            all_u_effect["all_u_specificity_accuracy"],
            0.5,
        )
        self.assertAlmostEqual(all_u_effect["all_u_specificity_G_U"], 0.0)
        self.assertAlmostEqual(
            all_u_effect["all_u_specificity_gold_margin_3"],
            -1.5,
        )
        headline = result["confirmatory"]["headline"][0]
        self.assertAlmostEqual(headline["headline_worst_cell_min"], 1.0 / 3.0)
        self.assertAlmostEqual(
            headline["all_u_specificity_G_U_worst_cell_min"],
            0.0,
        )
        self.assertAlmostEqual(
            headline["all_u_specificity_gold_margin_3_worst_cell_min"],
            -1.5,

        )
    def test_random_trigger_uses_greater_than_or_equal_threshold(self):
        aggregates = aggregate_intervention_rows(
            fixture_rows(include_portability=False),
            expected_random_seeds=(0, 1, 2),
        )
        at_threshold = summarize_rand_zero(
            aggregates,
            expected_random_seeds=(0, 1, 2),
            accuracy_spread_threshold=0.5,
        )
        above_threshold = summarize_rand_zero(
            aggregates,
            expected_random_seeds=(0, 1, 2),
            accuracy_spread_threshold=0.500001,
        )

        self.assertTrue(at_threshold["need_extra_seeds"])
        self.assertFalse(above_threshold["need_extra_seeds"])


    def test_square_only_summary_uses_declared_coverage(self):
        rows = [
            row
            for row in fixture_rows()
            if row["analysis_stratum"] == "square_valid"
        ]
        result = summarize_ablation(
            rows,
            expected_random_seeds=(0, 1, 2),
            expected_strata=("square_valid",),
        )

        self.assertEqual(
            result["coverage"]["expected_strata"],
            ["square_valid"],
        )
        self.assertEqual(len(result["by_cell"]), 24)
        self.assertNotIn(
            "all_u_specificity_accuracy_worst_cell_min",
            result["confirmatory"]["headline"][0],
        )


class AblationCoverageTests(unittest.TestCase):
    def test_rejects_duplicate_sample_id_within_condition(self):
        rows = fixture_rows()
        duplicate = next(
            row
            for row in rows
            if row["condition"] == "rho_zero"
        )
        rows.append(dict(duplicate))

        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "Duplicate sample_id",
        ):
            validate_condition_coverage(rows)

    def test_rejects_missing_seed_variant_and_misaligned_sample_sets(self):
        rows = fixture_rows()
        missing_variant = [
            row
            for row in rows
            if not (
                row["analysis_stratum"] == "square_valid"
                and row["cell"] == "++"
                and row["condition"] == "rand_zero"
                and row["random_seed"] == 2
            )
        ]
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "coverage mismatch",
        ):
            validate_condition_coverage(missing_variant)

        one_missing_sample = list(rows)
        one_missing_sample.remove(
            next(
                row
                for row in one_missing_sample
                if row["analysis_stratum"] == "square_valid"
                and row["cell"] == "++"
                and row["condition"] == "rand_zero"
                and row["random_seed"] == 1
            )
        )
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "sample_id sets differ",
        ):
            validate_condition_coverage(one_missing_sample)

    def test_rejects_extra_seed_and_seed_on_deterministic_condition(self):
        rows = fixture_rows()
        extra = dict(
            next(
                row
                for row in rows
                if row["condition"] == "rand_zero"
                and row["random_seed"] == 0
            )
        )
        extra["random_seed"] = 3
        rows.append(extra)
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "unexpected rand_zero seed",
        ):
            validate_condition_coverage(rows)

        rows = fixture_rows()
        next(row for row in rows if row["condition"] == "m_zero")[
            "random_seed"
        ] = 0
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "requires a blank random_seed",
        ):
            validate_condition_coverage(rows)


class PortabilitySummaryTests(unittest.TestCase):
    def test_rejects_missing_group_and_duplicate_direction(self):
        rows = fixture_rows()
        missing_group = [
            row
            for row in rows
            if not (
                row["condition"] == PORTABILITY_CONDITION
                and row["analysis_stratum"] == "square_valid"
                and row["portability_group"] == "negation_count_parity"
            )
        ]
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "coverage mismatch",
        ):
            validate_condition_coverage(missing_group)

        rows = fixture_rows()
        duplicate = dict(
            next(
                row
                for row in rows
                if row["condition"] == PORTABILITY_CONDITION
            )
        )
        duplicate["sample_id"] = "duplicate_direction"
        rows.append(duplicate)
        with self.assertRaisesRegex(
            Section62AblationMetricsError,
            "duplicate cell",
        ):
            validate_condition_coverage(rows)

    def test_separates_both_mechanisms_pair_scope_and_whole_square_subset(self):
        portability = summarize_ablation(fixture_rows())["portability"]

        parity = find_row(
            portability["by_group"],
            analysis_stratum="square_valid",
            scope="pair_eligible",
            portability_group="negation_count_parity",
        )
        position = find_row(
            portability["by_group"],
            analysis_stratum="square_valid",
            scope="pair_eligible",
            portability_group="negation_position",
        )
        self.assertEqual(parity["n"], 4)
        self.assertAlmostEqual(parity["accuracy"], 0.5)
        self.assertAlmostEqual(position["accuracy"], 1.0)

        whole_parity = find_row(
            portability["by_group"],
            analysis_stratum="square_valid",
            scope="whole_square_subset",
            portability_group="negation_count_parity",
        )
        whole_position = find_row(
            portability["by_group"],
            analysis_stratum="square_valid",
            scope="whole_square_subset",
            portability_group="negation_position",
        )
        self.assertAlmostEqual(whole_parity["accuracy"], 1.0)
        self.assertAlmostEqual(whole_position["accuracy"], 1.0)

        parity_pairs = find_row(
            portability["pair_outcomes"],
            analysis_stratum="square_valid",
            portability_group="negation_count_parity",
        )
        position_pairs = find_row(
            portability["pair_outcomes"],
            analysis_stratum="square_valid",
            portability_group="negation_position",
        )
        self.assertAlmostEqual(parity_pairs["pair_accuracy"], 0.5)
        self.assertAlmostEqual(position_pairs["pair_accuracy"], 1.0)

        square_outcome = find_row(
            portability["whole_square_outcomes"],
            analysis_stratum="square_valid",
        )
        self.assertEqual(square_outcome["n_squares"], 1)
        self.assertAlmostEqual(square_outcome["whole_square_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
