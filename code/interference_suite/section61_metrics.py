"""Confirmatory metrics and table aggregation for Section 6.1."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .section61_data import RHO_CONTROLS


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None


def behavioral_summary(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    gate_threshold: float = 0.90,
) -> dict[str, Any]:
    by_form_cell: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["form"]), str(row["cell"]))].append(row)

    for (form, cell), values in sorted(grouped.items()):
        accuracy = mean(float(row["is_correct"]) for row in values)
        sign_accuracy = mean(
            float((float(row["R"]) > 0) == (int(float(row["expected_R_sign"])) > 0))
            for row in values
        )
        by_form_cell.append(
            {
                "model_name": model_name,
                "form": form,
                "cell": cell,
                "n": len(values),
                "accuracy": accuracy,
                "sign_accuracy": sign_accuracy,
                "mean_R": mean(float(row["R"]) for row in values),
                "mean_U_gap": mean(float(row["U_gap"]) for row in values),
                "U_rate": mean(float(str(row["pred_label"]) == "U") for row in values),
            }
        )

    forms: list[dict[str, Any]] = []
    for form in sorted({record["form"] for record in by_form_cell}):
        cells = [record for record in by_form_cell if record["form"] == form]
        values = [row for row in rows if str(row["form"]) == form]
        min_cell = min(float(record["accuracy"]) for record in cells)
        forms.append(
            {
                "model_name": model_name,
                "form": form,
                "n": len(values),
                "accuracy": mean(float(row["is_correct"]) for row in values),
                "min_cell_accuracy": min_cell,
                "gate_threshold": gate_threshold,
                "eligible": min_cell >= gate_threshold,
                "mean_R_same": mean(
                    float(row["R"]) for row in values if int(float(row["rho"])) == 1
                ),
                "mean_R_opposite": mean(
                    float(row["R"]) for row in values if int(float(row["rho"])) == -1
                ),
            }
        )
        forms[-1]["phase_effect"] = (
            float(forms[-1]["mean_R_same"]) - float(forms[-1]["mean_R_opposite"])
        )

    return {
        "model_name": model_name,
        "gate_threshold": gate_threshold,
        "by_form_cell": by_form_cell,
        "by_form": forms,
    }


def das_summary(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    rotation: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty DAS result")
    by_control: list[dict[str, Any]] = []
    controls = sorted({str(row["control_type"]) for row in rows})
    for control in controls:
        values = [row for row in rows if str(row["control_type"]) == control]
        by_control.append(
            {
                "control_type": control,
                "n": len(values),
                "IIA": mean(float(row["is_counterfactual_correct"]) for row in values),
                "mean_R": mean(float(row["R"]) for row in values),
                "mean_U_gap": mean(float(row["U_gap"]) for row in values),
            }
        )

    control_map = {record["control_type"]: record for record in by_control}
    rho_values = [
        control_map[name]["IIA"]
        for name in RHO_CONTROLS
        if name in control_map and control_map[name]["IIA"] is not None
    ]
    rho_complete = all(name in control_map for name in RHO_CONTROLS)
    all_values = [
        float(record["IIA"])
        for record in by_control
        if record["IIA"] is not None
    ]
    first = rows[0]
    return {
        "model_name": model_name,
        "section61_experiment": first.get("section61_experiment"),
        "target_var": first.get("target_var"),
        "form": first.get("form"),
        "direction": first.get("direction", ""),
        "base_form": first.get("base_form"),
        "source_form": first.get("source_form"),
        "layer": int(rotation["layer"]),
        "site": str(rotation["site"]),
        "reported_site": (
            "claim_final"
            if str(rotation["site"]) == "row" and str(first.get("base_site")) == "claim_final"
            else str(rotation["site"])
        ),
        "rank": int(rotation["rank"]),
        "n": len(rows),
        "IIA": mean(float(row["is_counterfactual_correct"]) for row in rows),
        "by_control": by_control,
        "control_min_IIA": min(all_values) if all_values else None,
        "rho_full_audit_complete": rho_complete,
        "rho_full_audit_min_IIA": min(float(value) for value in rho_values)
        if rho_complete
        else None,
    }


def model_short(model_name: str) -> str:
    value = model_name.lower()
    if "qwen" in value:
        return "qwen"
    if "phi" in value:
        return "phi"
    return value.replace("/", "_")


def flatten_das_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key != "by_control"
    }


def main_table_rows(
    behavioral: Iterable[dict[str, Any]],
    das: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate = {
        (model_short(str(record["model_name"])), str(form["form"])): form
        for record in behavioral
        for form in record.get("by_form", [])
    }
    cells = {
        (
            model_short(str(record["model_name"])),
            str(record["section61_experiment"]),
            str(record["form"]),
        ): record
        for record in das
        if str(record.get("reported_site")) == "claim_final"
        and str(record.get("target_var")) == "rho"
        and str(record.get("section61_experiment")) in {"E2", "E3"}
    }
    forms = ("did_not", "didnt", "never", "did_not_ever", "failed_to", "not_the_case")
    rows: list[dict[str, Any]] = []
    for form in forms:
        out: dict[str, Any] = {"form": form}
        for model in ("phi", "qwen"):
            gate_record = gate.get((model, form), {})
            out[f"{model}_behavior_min"] = gate_record.get("min_cell_accuracy")
            out[f"{model}_eligible"] = gate_record.get("eligible")
            within = cells.get((model, "E2", form), {})
            cross = cells.get((model, "E3", form), {})
            if gate_record.get("eligible") is False:
                out[f"{model}_within"] = None
                out[f"{model}_cross"] = None
                out[f"{model}_interpretation"] = "behavioral_gate_failed"
            else:
                out[f"{model}_within"] = within.get("rho_full_audit_min_IIA")
                out[f"{model}_cross"] = cross.get("rho_full_audit_min_IIA")
                out[f"{model}_interpretation"] = "eligible"
        rows.append(out)

    worst: dict[str, Any] = {"form": "Worst form (min)"}
    for model in ("phi", "qwen"):
        eligible_forms = [
            row for row in rows if row.get(f"{model}_eligible") is True
        ]
        worst[f"{model}_behavior_min"] = min(
            float(row[f"{model}_behavior_min"]) for row in eligible_forms
        ) if eligible_forms else None
        for metric in ("within", "cross"):
            values = [
                float(row[f"{model}_{metric}"])
                for row in eligible_forms
                if row.get(f"{model}_{metric}") is not None
            ]
            worst[f"{model}_{metric}"] = min(values) if values else None
        worst[f"{model}_eligible"] = len(eligible_forms) == len(rows)
        worst[f"{model}_n_eligible"] = len(eligible_forms)
    rows.append(worst)
    return rows
