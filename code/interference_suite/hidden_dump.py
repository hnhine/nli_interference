"""Raw hidden-state dumping utilities for DAS rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .das_spans import resolve_token_site
from .model import DEFAULT_CACHE_DIR, normalize_cache_dir, progress_iter, resolve_torch_dtype


RAW_HIDDEN_COMPONENTS = {"block_input", "block_output"}


def dump_das_hidden_states(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    model_name: str,
    target_var: str,
    layer: int,
    component: str = "block_output",
    site: str = "row",
    split: str | None = None,
    control_types: list[str] | None = None,
    limit: int | None = 16,
    batch_size: int = 8,
    device: str | None = None,
    device_map: str | None = "auto",
    torch_dtype: str | None = "auto",
    trust_remote_code: bool = False,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    local_files_only: bool = False,
) -> dict[str, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime()
    if component not in RAW_HIDDEN_COMPONENTS:
        raise ValueError(
            f"Raw output_hidden_states only supports {sorted(RAW_HIDDEN_COMPONENTS)}; "
            f"component={component!r} needs a forward hook."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_rows = filter_rows(rows, target_var, split, control_types)
    if limit is not None:
        dump_rows = dump_rows[:limit]
    if not dump_rows:
        raise ValueError("No DAS rows matched hidden dump filters")

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
    input_device = get_input_device(model, torch, device)

    base_tensors = []
    source_tensors = []
    metadata_rows: list[dict[str, Any]] = []
    for batch_rows in progress_iter(chunks(dump_rows, batch_size), total=ceil_div(len(dump_rows), batch_size), desc="Dump hidden", unit="batch"):
        batch = collate_hidden_rows(batch_rows, tokenizer, torch, input_device, site)
        with torch.no_grad():
            base_outputs = model(**batch["base"], output_hidden_states=True, use_cache=False)
            source_outputs = model(**batch["source"], output_hidden_states=True, use_cache=False)
        hidden_index = hidden_state_index(layer, component, len(base_outputs.hidden_states))
        batch_index = torch.arange(len(batch_rows), device=input_device)
        base_pos = batch["base_positions"].to(input_device)
        source_pos = batch["source_positions"].to(input_device)
        base_hidden = base_outputs.hidden_states[hidden_index][batch_index, base_pos].detach().float().cpu()
        source_hidden = source_outputs.hidden_states[hidden_index][batch_index, source_pos].detach().float().cpu()
        base_tensors.append(base_hidden)
        source_tensors.append(source_hidden)

        for idx, row in enumerate(batch_rows):
            bh = base_hidden[idx]
            sh = source_hidden[idx]
            metadata_rows.append(
                {
                    "sample_id": row.get("sample_id"),
                    "split": row.get("split"),
                    "target_var": row.get("target_var"),
                    "control_type": row.get("control_type"),
                    "target_label": row.get("target_label"),
                    "base_label": row.get("base_label"),
                    "source_label": row.get("source_label"),
                    "base_site": batch["base_sites"][idx],
                    "source_site": batch["source_sites"][idx],
                    "base_position": int(batch["base_positions"][idx].detach().cpu()),
                    "source_position": int(batch["source_positions"][idx].detach().cpu()),
                    "base_token": batch["base_tokens"][idx],
                    "source_token": batch["source_tokens"][idx],
                    "base_hidden_norm": float(bh.norm().item()),
                    "source_hidden_norm": float(sh.norm().item()),
                    "diff_norm": float((sh - bh).norm().item()),
                    "cosine": float(torch.nn.functional.cosine_similarity(bh, sh, dim=0).item()),
                }
            )

    base_all = torch.cat(base_tensors, dim=0)
    source_all = torch.cat(source_tensors, dim=0)
    diff_all = source_all - base_all
    tensor_path = output_dir / "hidden_states.pt"
    torch.save(
        {
            "base_hidden": base_all,
            "source_hidden": source_all,
            "diff_hidden": diff_all,
            "metadata_rows": metadata_rows,
        },
        tensor_path,
    )

    summary = {
        "model_name": model_name,
        "target_var": target_var,
        "layer": layer,
        "component": component,
        "hidden_state_index": hidden_state_index(layer, component, model.config.num_hidden_layers + 1),
        "site": site,
        "split": split,
        "control_types": control_types or ["all"],
        "n": len(metadata_rows),
        "hidden_shape": list(base_all.shape),
        "tensor_path": str(tensor_path),
        "mean_base_norm": float(base_all.norm(dim=1).mean().item()),
        "mean_source_norm": float(source_all.norm(dim=1).mean().item()),
        "mean_diff_norm": float(diff_all.norm(dim=1).mean().item()),
        "rows": metadata_rows,
    }
    metadata_path = output_dir / "hidden_metadata.json"
    metadata_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    summary_csv = output_dir / "hidden_summary.csv"
    write_summary_csv(metadata_rows, summary_csv)
    summary["metadata_path"] = str(metadata_path)
    summary["summary_csv"] = str(summary_csv)
    return summary


def filter_rows(
    rows: list[dict[str, Any]],
    target_var: str,
    split: str | None,
    control_types: list[str] | None,
) -> list[dict[str, Any]]:
    out = [row for row in rows if row.get("target_var") == target_var]
    if split and split != "all":
        out = [row for row in out if row.get("split") == split]
    if control_types and control_types != ["all"]:
        allowed = set(control_types)
        out = [row for row in out if str(row.get("control_type")) in allowed]
    return out


def collate_hidden_rows(rows: list[dict[str, Any]], tokenizer: Any, torch: Any, device: Any, site_override: str) -> dict[str, Any]:
    base_texts = [str(row["base_prompt"]) for row in rows]
    source_texts = [str(row["source_prompt"]) for row in rows]
    base_inputs = encode_to_device(tokenizer, base_texts, torch, device)
    source_inputs = encode_to_device(tokenizer, source_texts, torch, device)
    base_positions = []
    source_positions = []
    base_sites = []
    source_sites = []
    base_tokens = []
    source_tokens = []
    for row, base_text, source_text in zip(rows, base_texts, source_texts):
        base_site, source_site = sites_for_row(row, site_override)
        base_pos = resolve_token_site(tokenizer, base_text, row, "base", base_site)
        source_pos = resolve_token_site(tokenizer, source_text, row, "source", source_site)
        base_positions.append(base_pos)
        source_positions.append(source_pos)
        base_sites.append(base_site)
        source_sites.append(source_site)
        base_tokens.append(token_at_position(tokenizer, base_text, base_pos))
        source_tokens.append(token_at_position(tokenizer, source_text, source_pos))
    return {
        "base": base_inputs,
        "source": source_inputs,
        "base_positions": torch.tensor(base_positions, dtype=torch.long, device=device),
        "source_positions": torch.tensor(source_positions, dtype=torch.long, device=device),
        "base_sites": base_sites,
        "source_sites": source_sites,
        "base_tokens": base_tokens,
        "source_tokens": source_tokens,
    }


def sites_for_row(row: dict[str, Any], site_override: str) -> tuple[str, str]:
    if site_override and site_override != "row":
        return site_override, site_override
    return str(row["base_site"]), str(row["source_site"])


def token_at_position(tokenizer: Any, text: str, position: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if position < 0 or position >= len(ids):
        return ""
    return tokenizer.decode([ids[position]])


def encode_to_device(tokenizer: Any, texts: list[str], torch: Any, device: Any) -> dict[str, Any]:
    encoded = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def hidden_state_index(layer: int, component: str, n_hidden_states: int) -> int:
    if component == "block_input":
        index = layer
    elif component == "block_output":
        index = layer + 1
    else:
        raise ValueError(f"Unsupported raw hidden component: {component}")
    if index < 0 or index >= n_hidden_states:
        raise ValueError(f"Hidden-state index {index} out of range for {n_hidden_states} hidden states")
    return index


def load_hf_model(
    torch: Any,
    auto_model_cls: Any,
    auto_tokenizer_cls: Any,
    model_name: str,
    device: str | None,
    device_map: str | None,
    torch_dtype: str | None,
    trust_remote_code: bool,
    cache_dir: str | None,
    local_files_only: bool,
) -> tuple[Any, Any]:
    cache_dir = normalize_cache_dir(cache_dir)
    hub_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        hub_kwargs["cache_dir"] = cache_dir
    tokenizer = auto_tokenizer_cls.from_pretrained(model_name, **hub_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs = dict(hub_kwargs)
    resolved_dtype = resolve_torch_dtype(torch, torch_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map and device_map != "none":
        model_kwargs["device_map"] = device_map
    model = auto_model_cls.from_pretrained(model_name, **model_kwargs)
    if (not device_map or device_map == "none") and device:
        model.to(device)
    elif (not device_map or device_map == "none") and device is None:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def get_input_device(model: Any, torch: Any, requested_device: str | None) -> Any:
    if requested_device:
        return torch.device(requested_device)
    return next(model.parameters()).device


def chunks(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def import_runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install torch and transformers before dumping hidden states") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
