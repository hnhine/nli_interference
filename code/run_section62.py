"""Prepare and evaluate Section 6.2 naturalistic NLI polarity squares."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

from interference_suite.das_pyvene import (
    encode_to_device,
    load_hf_model,
)
from interference_suite.io_utils import read_rows_csv, write_rows_csv
from interference_suite.model import (
    DEFAULT_CACHE_DIR,
    normalize_cache_dir,
    progress_iter,
    resolve_label_tokens,
)
from interference_suite.section62_data import RHO_SAME_DONOR, PreparedCorpus
from interference_suite.section62_data import prepare_corpus, sha256_file
from interference_suite.section62_metrics import behavioral_gate_summary


REQUIRED_SOURCE_SPLITS = {
    "MNLI": frozenset({"train", "dev_matched", "dev_mismatched"}),
    "SNLI": frozenset({"train", "dev"}),
}


def parse_source_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected SOURCE as split=/path/to/file.tsv, got {value!r}"
        )
    split, path = value.split("=", 1)
    if not split.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            f"Expected SOURCE as split=/path/to/file.tsv, got {value!r}"
        )
    return split.strip(), path.strip()


def source_mapping(values: list[tuple[str, str]], corpus: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split, path in values:
        normalized_split = split.lower().replace("-", "_")
        if normalized_split in mapping:
            raise ValueError(f"{corpus}: duplicate source split {normalized_split!r}")
        mapping[normalized_split] = path
    return mapping


def validate_source_scope(
    mapping: dict[str, str],
    corpus: str,
    *,
    allow_partial: bool,
) -> None:
    resolved_paths = [str(Path(path).resolve()) for path in mapping.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(f"{corpus}: the same source path was supplied twice")
    expected = REQUIRED_SOURCE_SPLITS[corpus]
    if allow_partial:
        return
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{corpus}: source splits must be {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in source join overrides: {key!r}")
        result[key] = value
    return result


def load_source_join_overrides(
    path: str | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not path:
        return {}
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_keys,
    )
    if not isinstance(value, dict) or set(value) != {"schema_version", "corpora"}:
        raise ValueError(
            "Source join override file must contain exactly schema_version and corpora"
        )
    if value["schema_version"] != 1:
        raise ValueError(
            f"Unsupported source join override schema: {value['schema_version']!r}"
        )
    corpora = value["corpora"]
    if not isinstance(corpora, dict):
        raise ValueError("Source join override corpora must be a JSON object")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_corpus, raw_entries in corpora.items():
        corpus = str(raw_corpus).upper()
        if corpus not in REQUIRED_SOURCE_SPLITS:
            raise ValueError(f"Unknown source join override corpus: {raw_corpus!r}")
        if not isinstance(raw_entries, dict):
            raise ValueError(
                f"Source join overrides for {corpus} must be a JSON object"
            )
        result[corpus] = {
            str(triple_id): dict(spec)
            for triple_id, spec in raw_entries.items()
            if isinstance(spec, dict)
        }
        if len(result[corpus]) != len(raw_entries):
            raise ValueError(f"Every source join override for {corpus} must be an object")
    return result


def load_quarantine(path: str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Quarantine file must be a JSON object keyed by corpus")
    result: dict[str, dict[str, str]] = {}
    for corpus, entries in value.items():
        if not isinstance(entries, dict):
            raise ValueError(
                f"Quarantine entries for {corpus!r} must map triple ids to reasons"
            )
        normalized: dict[str, str] = {}
        for key, reason in entries.items():
            if not str(reason).strip():
                raise ValueError(f"Empty quarantine reason for {corpus}:{key}")
            normalized[str(key)] = str(reason)
        result[str(corpus).upper()] = normalized
    return result


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv_artifact(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    schema_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    write_rows_csv(rows, path, schema_rows=schema_rows)
    return {
        "path": path.name,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def validate_r0_manifest(
    manifest_path: str | Path,
    *,
    square_path: Path,
    square_rows: list[dict[str, Any]],
    all_u_path: Path,
    all_u_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("R0 manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported R0 manifest schema: {manifest.get('schema_version')!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("R0 manifest is missing an outputs object")

    requested = {
        "square_valid": (square_path, square_rows),
        "all_u": (all_u_path, all_u_rows),
    }
    matched: dict[str, Any] = {}
    for role, (input_path, rows) in requested.items():
        resolved_input = input_path.resolve()
        candidates: list[tuple[str, dict[str, Any], Path]] = []
        for artifact_name, raw_spec in outputs.items():
            if not isinstance(raw_spec, dict) or not raw_spec.get("path"):
                continue
            artifact_path = Path(str(raw_spec["path"]))
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            if artifact_path.resolve() == resolved_input:
                candidates.append((str(artifact_name), raw_spec, artifact_path))
        if len(candidates) != 1:
            raise ValueError(
                f"R0 manifest must identify {input_path} exactly once; "
                f"matches={[name for name, _, _ in candidates]}"
            )
        artifact_name, spec, artifact_path = candidates[0]
        valid_name = artifact_name == role or artifact_name.endswith(f"_{role}")
        if not valid_name:
            raise ValueError(
                f"R0 artifact {artifact_name!r} cannot be used as {role!r}"
            )
        expected_rows = int(spec.get("rows", -1))
        expected_sha256 = str(spec.get("sha256", ""))
        expected_bytes = int(spec.get("bytes", -1))
        actual_sha256 = sha256_file(input_path)
        actual_bytes = input_path.stat().st_size
        if expected_rows != len(rows):
            raise ValueError(
                f"R0 artifact {artifact_name!r} row mismatch: "
                f"manifest={expected_rows}, actual={len(rows)}"
            )
        if expected_sha256 != actual_sha256:
            raise ValueError(
                f"R0 artifact {artifact_name!r} SHA256 mismatch: "
                f"manifest={expected_sha256}, actual={actual_sha256}"
            )
        if expected_bytes != actual_bytes:
            raise ValueError(
                f"R0 artifact {artifact_name!r} byte-size mismatch: "
                f"manifest={expected_bytes}, actual={actual_bytes}"
            )
        matched[role] = {
            "artifact_name": artifact_name,
            "path": str(artifact_path),
            "rows": len(rows),
            "sha256": actual_sha256,
            "bytes": actual_bytes,
        }

    return {
        "path": str(path),
        "schema_version": 1,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "matched_artifacts": matched,
    }


def validate_claim_span(
    row: dict[str, Any],
    prefix: str,
    *,
    context: str,
) -> None:
    prompt = str(row[f"{prefix}_prompt"])
    claim = str(row[f"{prefix}_claim"])
    try:
        start = int(row[f"{prefix}_claim_span_start"])
        end = int(row[f"{prefix}_claim_span_end"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid {prefix} claim span") from exc
    if not 0 <= start < end <= len(prompt):
        raise ValueError(
            f"{context} has out-of-range {prefix} claim span [{start}, {end})"
        )
    if prompt[start:end] != claim:
        raise ValueError(
            f"{context} {prefix} claim span does not recover {prefix}_claim"
        )


def validate_behavioral_rows(
    square_rows: list[dict[str, Any]],
    all_u_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_by_stratum = {
        "square_valid": {"++": "T", "-+": "F", "+-": "F", "--": "T"},
        "all_u": {"++": "U", "-+": "U", "+-": "U", "--": "U"},
    }
    expected_signatures = {
        "square_valid": "(E,C,C,E)",
        "all_u": "(N,N,N,N)",
    }
    supplied = [
        ("square_valid", square_rows),
        ("all_u", all_u_rows),
    ]
    square_corpora = {str(row.get("corpus", "")) for row in square_rows}
    all_u_corpora = {str(row.get("corpus", "")) for row in all_u_rows}
    if square_corpora != all_u_corpora:
        raise ValueError(
            "Behavioral square/all-U inputs must cover the same corpora; "
            f"square={sorted(square_corpora)}, all_u={sorted(all_u_corpora)}"
        )

    seen_sample_ids: set[str] = set()
    seen_groups: dict[tuple[str, str, str], set[str]] = {}
    combined: list[dict[str, Any]] = []
    rows_by_sample_id: dict[str, dict[str, Any]] = {}
    for expected_stratum, rows in supplied:
        if not rows:
            raise ValueError(
                f"No rows found for behavioral stratum {expected_stratum!r}"
            )
        for index, row in enumerate(rows):
            context = f"{expected_stratum} row {index}"
            missing = [
                field
                for field in (
                    "sample_id",
                    "corpus",
                    "triple_id",
                    "analysis_stratum",
                    "cell",
                    "expected_label",
                    "target_label",
                    "base_label",
                    "target_var",
                    "prompt",
                    "base_prompt",
                    "base_claim",
                    "base_site",
                    "base_claim_span_start",
                    "base_claim_span_end",
                    "source_prompt",
                    "source_label",
                    "source_claim",
                    "source_site",
                    "source_claim_span_start",
                    "source_claim_span_end",
                    "rho_base",
                    "rho_src",
                    "portability_donor_cell",
                    "portability_donor_sample_id",
                    "portability_group",
                    "full_signature",
                )
                if not str(row.get(field, "")).strip()
            ]
            if missing:
                raise ValueError(f"{context} is missing fields: {missing}")

            if str(row["target_var"]) != "rho":
                raise ValueError(f"{context} has target_var={row['target_var']!r}")
            if str(row["prompt"]) != str(row["base_prompt"]):
                raise ValueError(f"{context} prompt does not equal base_prompt")
            if str(row["base_site"]) != "claim_final":
                raise ValueError(f"{context} has base_site={row['base_site']!r}")
            if str(row["source_site"]) != "claim_final":
                raise ValueError(f"{context} has source_site={row['source_site']!r}")
            if str(row.get("m_base", "")).strip():
                raise ValueError(f"{context} unexpectedly defines m_base")

            actual_stratum = str(row["analysis_stratum"])
            if actual_stratum != expected_stratum:
                raise ValueError(
                    f"{context} has analysis_stratum={actual_stratum!r}"
                )
            signature = str(row["full_signature"])
            if signature != expected_signatures[expected_stratum]:
                raise ValueError(
                    f"{context} has full_signature={signature!r}, expected "
                    f"{expected_signatures[expected_stratum]!r}"
                )
            cell = str(row["cell"])
            expected_label = expected_by_stratum[expected_stratum].get(cell)
            if expected_label is None:
                raise ValueError(f"{context} has unknown polarity cell {cell!r}")
            if str(row["expected_label"]) != expected_label:
                raise ValueError(
                    f"{context} cell={cell} has expected_label="
                    f"{row['expected_label']!r}, expected {expected_label!r}"
                )
            if not (
                str(row["base_label"])
                == str(row["target_label"])
                == expected_label
            ):
                raise ValueError(
                    f"{context} requires base_label == target_label == "
                    f"expected_label={expected_label!r}"
                )
            expected_rho = 1.0 if cell in {"++", "--"} else -1.0
            rho_base = float(row["rho_base"])
            rho_src = float(row["rho_src"])
            if rho_base != expected_rho or rho_src != expected_rho:
                raise ValueError(
                    f"{context} cell={cell} has rho_base={rho_base}, "
                    f"rho_src={rho_src}, expected {expected_rho}"
                )
            expected_donor_cell = RHO_SAME_DONOR[cell]
            if str(row["portability_donor_cell"]) != expected_donor_cell:
                raise ValueError(
                    f"{context} has portability_donor_cell="
                    f"{row['portability_donor_cell']!r}, expected "
                    f"{expected_donor_cell!r}"
                )
            if row.get("rho_same_donor_cell") not in (None, "") and str(
                row["rho_same_donor_cell"]
            ) != expected_donor_cell:
                raise ValueError(f"{context} has inconsistent rho_same_donor_cell")
            expected_portability_group = (
                "negation_count_parity"
                if cell in {"++", "--"}
                else "negation_position"
            )
            if str(row["portability_group"]) != expected_portability_group:
                raise ValueError(
                    f"{context} has invalid portability_group="
                    f"{row['portability_group']!r}"
                )
            validate_claim_span(row, "base", context=context)
            validate_claim_span(row, "source", context=context)

            sample_id = str(row["sample_id"])
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate behavioral sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            rows_by_sample_id[sample_id] = row
            group = (
                str(row["corpus"]),
                expected_stratum,
                str(row["triple_id"]),
            )
            seen_groups.setdefault(group, set())
            if cell in seen_groups[group]:
                raise ValueError(f"{group} contains duplicate cell {cell!r}")
            seen_groups[group].add(cell)
            combined.append(row)

    expected_cells = set(expected_by_stratum["square_valid"])
    for group, cells in sorted(seen_groups.items()):
        if cells != expected_cells:
            raise ValueError(
                f"{group} has cells={sorted(cells)}, expected="
                f"{sorted(expected_cells)}"
            )

    for row in combined:
        sample_id = str(row["sample_id"])
        context = f"{row['corpus']}:{row['triple_id']}:{row['cell']}"
        donor_id = str(row["portability_donor_sample_id"])
        donor = rows_by_sample_id.get(donor_id)
        if donor is None:
            raise ValueError(f"{context} donor {donor_id!r} is not in the inputs")
        if (
            str(donor["corpus"]) != str(row["corpus"])
            or str(donor["triple_id"]) != str(row["triple_id"])
            or str(donor["analysis_stratum"]) != str(row["analysis_stratum"])
        ):
            raise ValueError(f"{context} portability donor leaves its square")
        if str(donor["cell"]) != RHO_SAME_DONOR[str(row["cell"])]:
            raise ValueError(f"{context} points to the wrong donor cell")
        if str(donor["portability_donor_sample_id"]) != sample_id:
            raise ValueError(f"{context} portability donor is not reciprocal")
        if row.get("rho_same_donor_sample_id") not in (None, "") and str(
            row["rho_same_donor_sample_id"]
        ) != donor_id:
            raise ValueError(f"{context} has inconsistent rho_same_donor_sample_id")
        materialized_pairs = (
            ("source_prompt", "base_prompt"),
            ("source_claim", "base_claim"),
            ("source_label", "base_label"),
            ("source_site", "base_site"),
        )
        for source_field, donor_field in materialized_pairs:
            if str(row[source_field]) != str(donor[donor_field]):
                raise ValueError(
                    f"{context} {source_field} does not match donor {donor_field}"
                )
        if float(row["rho_src"]) != float(donor["rho_base"]):
            raise ValueError(f"{context} rho_src does not match donor rho_base")
        if (
            int(row["source_claim_span_start"])
            != int(donor["base_claim_span_start"])
            or int(row["source_claim_span_end"])
            != int(donor["base_claim_span_end"])
        ):
            raise ValueError(
                f"{context} source claim span does not match donor base claim span"
            )
    return combined


def score_behavioral_rows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    device: str | None,
    device_map: str | None,
    torch_dtype: str | None,
    label_token_style: str,
    trust_remote_code: bool,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install torch and transformers before behavioral scoring"
        ) from exc

    tokenizer, model = load_hf_model(
        torch=torch,
        auto_model_cls=AutoModelForCausalLM,
        auto_tokenizer_cls=AutoTokenizer,
        model_name=model_name,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    label_tokens = resolve_label_tokens(tokenizer, label_token_style)
    input_device = next(model.parameters()).device
    label_order = ("T", "F", "U")
    label_ids = label_tokens.token_ids
    label_id_set = set(label_ids.values())

    scored: list[dict[str, Any]] = []
    starts = progress_iter(
        range(0, len(rows), batch_size),
        total=ceil(len(rows) / batch_size),
        desc="Section 6.2 behavioral",
    )
    for start in starts:
        batch_rows = rows[start : start + batch_size]
        inputs = encode_to_device(
            tokenizer,
            [str(row["base_prompt"]) for row in batch_rows],
            input_device,
        )
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits
            final_indices = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[
                batch_indices,
                final_indices.to(logits.device),
            ]
            tfu_logits = torch.stack(
                [next_logits[:, label_ids[label]] for label in label_order],
                dim=-1,
            )
            pred_indices = tfu_logits.argmax(dim=-1)
            global_top_ids = next_logits.argmax(dim=-1)

        for row, row_logits, pred_index, top_id in zip(
            batch_rows,
            tfu_logits,
            pred_indices,
            global_top_ids,
        ):
            stable_logits = row_logits.float()
            values = {
                label: float(stable_logits[index].detach().cpu())
                for index, label in enumerate(label_order)
            }
            expected_label = str(row["expected_label"])
            expected_index = label_order.index(expected_label)
            other_indices = [
                index for index in range(len(label_order)) if index != expected_index
            ]
            gold_margin = float(
                (
                    stable_logits[expected_index]
                    - torch.logsumexp(stable_logits[other_indices], dim=0)
                )
                .detach()
                .cpu()
            )
            pred_label = label_order[int(pred_index.detach().cpu())]
            pred_tf_label = "T" if values["T"] >= values["F"] else "F"
            r_value = values["T"] - values["F"]
            top_token_id = int(top_id.detach().cpu())
            out = dict(row)
            out.update(
                {
                    "condition": "base",
                    "subspace_var": "",
                    "random_seed": "",
                    "site_alias": "",
                    "layer": "",
                    "model_name": model_name,
                    "label_token_style": label_tokens.style,
                    "label_token_T": label_tokens.token_texts["T"],
                    "label_token_F": label_tokens.token_texts["F"],
                    "label_token_U": label_tokens.token_texts["U"],
                    "label_token_id_T": label_ids["T"],
                    "label_token_id_F": label_ids["F"],
                    "label_token_id_U": label_ids["U"],
                    "logit_T": values["T"],
                    "logit_F": values["F"],
                    "logit_U": values["U"],
                    "R": r_value,
                    "M_TF": (
                        float(row["rho_base"]) * r_value
                        if str(row["analysis_stratum"]) == "square_valid"
                        else ""
                    ),
                    "gold_margin_3": gold_margin,
                    "pred_label": pred_label,
                    "pred_tf_label": pred_tf_label,
                    "is_correct": int(pred_label == expected_label),
                    "is_tf_correct": (
                        int(pred_tf_label == expected_label)
                        if expected_label in {"T", "F"}
                        else ""
                    ),
                    "U_gap": values["U"] - max(values["T"], values["F"]),
                    "global_top_token_id": top_token_id,
                    "global_top_token": tokenizer.decode([top_token_id]),
                    "global_top_in_TFU": int(top_token_id in label_id_set),
                }
            )
            scored.append(out)

    run_info = {
        "model_name": model_name,
        "batch_size": batch_size,
        "device": device,
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "input_device_resolved": str(input_device),
        "parameter_dtype_resolved": str(next(model.parameters()).dtype),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "model_name_or_path_resolved": str(
            getattr(model.config, "_name_or_path", model_name)
        ),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get(
            "_commit_hash"
        ),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "trust_remote_code": trust_remote_code,
        "cache_dir": normalize_cache_dir(cache_dir),
        "local_files_only": local_files_only,
        "label_token_style_requested": label_token_style,
        "label_token_style_resolved": label_tokens.style,
        "label_token_ids": dict(label_ids),
        "label_token_texts": dict(label_tokens.token_texts),
        "tokenization": {
            "add_special_tokens": False,
            "padding_side": tokenizer.padding_side,
        },
    }
    return scored, run_info


def behavioral_command(args: argparse.Namespace) -> int:
    if not 0.0 <= args.gate_threshold <= 1.0:
        raise ValueError("gate_threshold must lie in [0, 1]")
    if args.min_baseline_correct_per_cell < 1:
        raise ValueError("min_baseline_correct_per_cell must be positive")
    if args.device and args.device_map not in (None, "none"):
        raise ValueError(
            "--device is only honored with --device-map none; choose explicit "
            "placement or device-map sharding, not both"
        )

    square_path = Path(args.square_samples)
    all_u_path = Path(args.all_u_samples)
    square_rows = read_rows_csv(square_path)
    all_u_rows = read_rows_csv(all_u_path)
    r0_manifest = validate_r0_manifest(
        args.r0_manifest,
        square_path=square_path,
        square_rows=square_rows,
        all_u_path=all_u_path,
        all_u_rows=all_u_rows,
    )
    rows = validate_behavioral_rows(square_rows, all_u_rows)
    scored, run_info = score_behavioral_rows(
        rows,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        label_token_style=args.label_token_style,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    summary = behavioral_gate_summary(
        scored,
        model_name=args.model_name,
        gate_threshold=args.gate_threshold,
        min_baseline_correct_per_cell=args.min_baseline_correct_per_cell,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    square_scored = [
        row for row in scored if row["analysis_stratum"] == "square_valid"
    ]
    all_u_scored = [
        row for row in scored if row["analysis_stratum"] == "all_u"
    ]
    baseline_correct_square_rows = [
        row for row in square_scored if int(row["is_correct"]) == 1
    ]
    baseline_correct_all_u_rows = [
        row for row in all_u_scored if int(row["is_correct"]) == 1
    ]
    whole_correct_keys = {
        (
            str(row["corpus"]),
            str(row["analysis_stratum"]),
            str(row["triple_id"]),
        )
        for row in summary["by_square"]
        if int(row["all_cells_correct"]) == 1
    }
    baseline_correct_whole_square_rows = [
        row
        for row in square_scored
        if (str(row["corpus"]), "square_valid", str(row["triple_id"]))
        in whole_correct_keys
    ]
    baseline_correct_whole_all_u_rows = [
        row
        for row in all_u_scored
        if (str(row["corpus"]), "all_u", str(row["triple_id"]))
        in whole_correct_keys
    ]
    artifacts = {
        "scored": write_csv_artifact(scored, output / "scored.csv"),
        "square_scored": write_csv_artifact(
            square_scored, output / "square_scored.csv"
        ),
        "all_u_scored": write_csv_artifact(
            all_u_scored, output / "all_u_scored.csv"
        ),
        "baseline_correct_square_rows": write_csv_artifact(
            baseline_correct_square_rows,
            output / "baseline_correct_square_rows.csv",
            schema_rows=square_scored,
        ),
        "baseline_correct_all_u_rows": write_csv_artifact(
            baseline_correct_all_u_rows,
            output / "baseline_correct_all_u_rows.csv",
            schema_rows=all_u_scored,
        ),
        "baseline_correct_whole_square_rows": write_csv_artifact(
            baseline_correct_whole_square_rows,
            output / "baseline_correct_whole_square_rows.csv",
            schema_rows=square_scored,
        ),
        "baseline_correct_whole_all_u_rows": write_csv_artifact(
            baseline_correct_whole_all_u_rows,
            output / "baseline_correct_whole_all_u_rows.csv",
            schema_rows=all_u_scored,
        ),
        "behavioral_by_cell": write_csv_artifact(
            summary["by_cell"], output / "behavioral_by_cell.csv"
        ),
        "behavioral_by_square": write_csv_artifact(
            summary["by_square"], output / "behavioral_by_square.csv"
        ),
        "behavioral_by_stratum": write_csv_artifact(
            summary["by_stratum"], output / "behavioral_by_stratum.csv"
        ),
    }
    result = {
        "schema_version": 1,
        "run_type": "section62_behavioral_gate",
        "r0_manifest": r0_manifest,
        "inputs": {
            "square_samples": {
                "path": str(square_path),
                "rows": len(square_rows),
                "sha256": sha256_file(square_path),
                "bytes": square_path.stat().st_size,
            },
            "all_u_samples": {
                "path": str(all_u_path),
                "rows": len(all_u_rows),
                "sha256": sha256_file(all_u_path),
                "bytes": all_u_path.stat().st_size,
            },
        },
        "run": run_info,
        "artifacts": artifacts,
        **summary,
    }
    write_json(result, output / "behavioral_summary.json")
    for gate_name in ("square_gate", "all_u_gate"):
        for gate in result[gate_name]:
            print(
                f"{gate_name} {gate['corpus']}: "
                f"min_cell_accuracy={gate['min_cell_accuracy']:.3f}, "
                f"strict={gate['strict_gate_eligible']}, "
                f"min_correct={gate['min_baseline_correct_per_cell']}, "
                f"secondary={gate['secondary_baseline_correct_eligible']}"
            )
    print(f"Wrote Section 6.2 behavioral scores and gates to {output}")
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    quarantine = load_quarantine(args.quarantine_file)
    source_join_overrides = load_source_join_overrides(args.source_join_overrides)
    requested: list[tuple[str, str, list[tuple[str, str]]]] = []
    quarantine_provenance = None
    if args.quarantine_file:
        quarantine_path = Path(args.quarantine_file)
        quarantine_provenance = {
            "path": str(quarantine_path),
            "sha256": sha256_file(quarantine_path),
            "bytes": quarantine_path.stat().st_size,
        }
    source_join_override_provenance = None
    if args.source_join_overrides:
        override_path = Path(args.source_join_overrides)
        source_join_override_provenance = {
            "path": str(override_path),
            "sha256": sha256_file(override_path),
            "bytes": override_path.stat().st_size,
            "schema_version": 1,
        }
    if args.mnli_negation:
        requested.append(("MNLI", args.mnli_negation, args.mnli_source))
    if args.snli_negation:
        requested.append(("SNLI", args.snli_negation, args.snli_source))
    if not requested:
        raise ValueError("Provide at least one of --mnli-negation or --snli-negation")

    prepared: list[PreparedCorpus] = []
    for corpus, negation_path, source_specs in requested:
        sources = source_mapping(source_specs, corpus)
        if not sources:
            raise ValueError(f"{corpus}: at least one --{corpus.lower()}-source is required")
        validate_source_scope(
            sources, corpus, allow_partial=args.allow_partial_source_splits
        )
        prepared.append(
            prepare_corpus(
                corpus=corpus,
                negation_path=negation_path,
                source_paths=sources,
                expected_triples=args.expected_triples,
                quarantine=quarantine.get(corpus, {}),
                require_complete_joins=not args.allow_incomplete_joins,
                allow_normalized_fallback=args.allow_normalized_fallback,
                verify_upstream_blob=not args.allow_unpinned_negation,
                source_join_overrides=source_join_overrides.get(corpus, {}),
            )
        )

    unexpected_quarantine = sorted(
        set(quarantine) - {item.corpus for item in prepared}
    )
    if unexpected_quarantine:
        raise ValueError(
            f"Quarantine contains unrequested corpora: {unexpected_quarantine}"
        )

    unexpected_override_corpora = sorted(
        set(source_join_overrides) - {item.corpus for item in prepared}
    )
    if unexpected_override_corpora:
        raise ValueError(
            "Source join overrides contain unrequested corpora: "
            f"{unexpected_override_corpora}"
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [row for item in prepared for row in item.all_rows]
    square_rows = [row for item in prepared for row in item.square_rows]
    u_rows = [row for item in prepared for row in item.all_u_rows]
    base_ent_rows = [
        row
        for row in all_rows
        if row["square_class"] == "base_ent_non_square"
    ]
    triple_rows = [row for item in prepared for row in item.triples]
    artifacts = {
        "all_rows": write_csv_artifact(all_rows, output / "all_rows.csv"),
        "square_valid": write_csv_artifact(square_rows, output / "square_valid.csv"),
        "all_u": write_csv_artifact(u_rows, output / "all_u.csv"),
        "base_ent_non_square": write_csv_artifact(base_ent_rows, output / "base_ent_non_square.csv"),
        "triples": write_csv_artifact(triple_rows, output / "triples.csv"),
    }

    for item in prepared:
        prefix = item.corpus.lower()
        item_base_ent_rows = [
            row
            for row in item.all_rows
            if row["square_class"] == "base_ent_non_square"
        ]
        artifacts[f"{prefix}_all"] = write_csv_artifact(item.all_rows, output / f"{prefix}_all.csv")
        artifacts[f"{prefix}_square_valid"] = write_csv_artifact(item.square_rows, output / f"{prefix}_square_valid.csv")
        artifacts[f"{prefix}_all_u"] = write_csv_artifact(item.all_u_rows, output / f"{prefix}_all_u.csv")
        artifacts[f"{prefix}_base_ent_non_square"] = write_csv_artifact(
            item_base_ent_rows, output / f"{prefix}_base_ent_non_square.csv"
        )
        artifacts[f"{prefix}_triple_audit"] = write_csv_artifact(item.triples, output / f"{prefix}_triple_audit.csv")

    manifest = {
        "schema_version": 1,
        "corpora": {item.corpus: item.manifest for item in prepared},
        "outputs": artifacts,
        "quarantine_file": quarantine_provenance,
        "source_join_override_file": source_join_override_provenance,
        "configuration": {
            "expected_triples": args.expected_triples,
            "allow_normalized_fallback": args.allow_normalized_fallback,
            "require_complete_joins": not args.allow_incomplete_joins,
            "require_official_negation_blob": not args.allow_unpinned_negation,
            "require_canonical_source_splits": not args.allow_partial_source_splits,
            "source_join_overrides_enabled": bool(source_join_overrides),
        },
        "counts": {
            "all_rows": len(all_rows),
            "square_valid_rows": len(square_rows),
            "all_u_rows": len(u_rows),
            "base_ent_non_square_rows": len(base_ent_rows),
            "triples": len(triple_rows),
        },
    }
    write_json(manifest, output / "manifest.json")
    print(f"Wrote Section 6.2 R0 datasets and manifest to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--mnli-negation")
    prepare.add_argument(
        "--mnli-source",
        action="append",
        type=parse_source_spec,
        default=[],
        metavar="SPLIT=PATH",
    )
    prepare.add_argument("--snli-negation")
    prepare.add_argument(
        "--snli-source",
        action="append",
        type=parse_source_spec,
        default=[],
        metavar="SPLIT=PATH",
    )
    prepare.add_argument("--expected-triples", type=int, default=500)
    prepare.add_argument("--quarantine-file")
    prepare.add_argument("--allow-normalized-fallback", action="store_true")
    prepare.add_argument("--allow-incomplete-joins", action="store_true")
    prepare.add_argument("--source-join-overrides")
    prepare.add_argument("--allow-partial-source-splits", action="store_true")
    prepare.add_argument("--allow-unpinned-negation", action="store_true")
    prepare.add_argument("--output-dir", required=True)

    behavioral = commands.add_parser("behavioral")
    behavioral.add_argument(
        "--r0-manifest",
        "--data-manifest",
        dest="r0_manifest",
        required=True,
    )
    behavioral.add_argument("--square-samples", required=True)
    behavioral.add_argument("--all-u-samples", required=True)
    behavioral.add_argument("--model-name", required=True)
    behavioral.add_argument("--batch-size", type=int, default=16)
    behavioral.add_argument("--gate-threshold", type=float, default=0.90)
    behavioral.add_argument(
        "--min-baseline-correct-per-cell",
        type=int,
        default=50,
    )
    behavioral.add_argument("--device")
    behavioral.add_argument("--device-map", default="auto")
    behavioral.add_argument("--torch-dtype", default="auto")
    behavioral.add_argument(
        "--label-token-style",
        choices=["auto", "bare", "space", "newline"],
        default="auto",
    )
    behavioral.add_argument("--trust-remote-code", action="store_true")
    behavioral.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    behavioral.add_argument("--local-files-only", action="store_true")
    behavioral.add_argument("--output-dir", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return {
        "prepare": prepare_command,
        "behavioral": behavioral_command,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
