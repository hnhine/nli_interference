"""Runtime primitives for fixed-subspace Section 6.2 interventions."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

from .das_pyvene import encode_to_device
from .das_spans import resolve_token_site
from .joint_gate_intervention import orthonormalize_basis
from .model import progress_iter
from .section62_data import sha256_file


LABEL_ORDER = ("T", "F", "U")


def get_decoder_layers(model: Any) -> Any:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers (model.model.layers)")
    return layers


def row_batches(rows: list[dict[str, Any]], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def effective_site(site: str, row: dict[str, Any]) -> str:
    if site != "row":
        return site
    resolved = str(row.get("base_site", ""))
    if not resolved:
        raise ValueError("A row-level rotation requires base_site metadata")
    return resolved


def resolve_positions(
    tokenizer: Any,
    texts: list[str],
    rows: list[dict[str, Any]],
    site: str,
) -> list[int]:
    positions = []
    for text, row in zip(texts, rows):
        positions.append(
            resolve_token_site(
                tokenizer,
                text,
                row,
                "base",
                effective_site(site, row),
            )
        )
    return positions


def next_token_metrics(
    *,
    torch: Any,
    next_logits: Any,
    rows: list[dict[str, Any]],
    label_token_ids: dict[str, int],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    label_id_set = set(label_token_ids.values())
    tfu = torch.stack(
        [next_logits[:, label_token_ids[label]] for label in LABEL_ORDER],
        dim=-1,
    ).float()
    pred_indices = tfu.argmax(dim=-1)
    global_top_ids = next_logits.argmax(dim=-1)
    output: list[dict[str, Any]] = []
    for row, values_tensor, pred_index, top_id in zip(
        rows,
        tfu,
        pred_indices,
        global_top_ids,
    ):
        values = {
            label: float(values_tensor[index].detach().cpu())
            for index, label in enumerate(LABEL_ORDER)
        }
        expected = str(row["expected_label"])
        expected_index = LABEL_ORDER.index(expected)
        other_indices = [index for index in range(3) if index != expected_index]
        gold_margin = float(
            (
                values_tensor[expected_index]
                - torch.logsumexp(values_tensor[other_indices], dim=0)
            )
            .detach()
            .cpu()
        )
        r_value = values["T"] - values["F"]
        pred_label = LABEL_ORDER[int(pred_index.detach().cpu())]
        pred_tf_label = "T" if r_value >= 0.0 else "F"
        top_token_id = int(top_id.detach().cpu())
        output.append(
            {
                "logit_T": values["T"],
                "logit_F": values["F"],
                "logit_U": values["U"],
                "R": r_value,
                "M_TF": (
                    float(row["rho_base"]) * r_value
                    if str(row["analysis_stratum"]) == "square_valid"
                    else ""
                ),
                "G_U": values["U"]
                - float(torch.logsumexp(values_tensor[:2], dim=0).detach().cpu()),
                "gold_margin_3": gold_margin,
                "U_gap": values["U"] - max(values["T"], values["F"]),
                "pred_label": pred_label,
                "pred_tf_label": pred_tf_label,
                "is_correct": int(pred_label == expected),
                "is_tf_correct": (
                    int(pred_tf_label == expected) if expected in {"T", "F"} else ""
                ),
                "global_top_token_id": top_token_id,
                "global_top_token": tokenizer.decode([top_token_id]),
                "global_top_in_TFU": int(top_token_id in label_id_set),
            }
        )
    return output


def score_base_rows(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: list[dict[str, Any]],
    label_token_ids: dict[str, int],
    batch_size: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    batches = row_batches(rows, batch_size)
    batches = progress_iter(
        batches,
        total=ceil(len(rows) / batch_size),
        desc="R2 base verification",
    )
    for batch_rows in batches:
        texts = [str(row["base_prompt"]) for row in batch_rows]
        encoded = encode_to_device(tokenizer, texts, device)
        with torch.no_grad():
            logits = model(**encoded, use_cache=False).logits
            final = encoded["attention_mask"].sum(dim=1) - 1
            indices = torch.arange(logits.shape[0], device=logits.device)
            next_logits = logits[indices, final.to(logits.device)]
        scored.extend(
            next_token_metrics(
                torch=torch,
                next_logits=next_logits,
                rows=batch_rows,
                label_token_ids=label_token_ids,
                tokenizer=tokenizer,
            )
        )
    return scored


def run_single_subspace_condition(
    *,
    model: Any,
    layers: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: list[dict[str, Any]],
    label_token_ids: dict[str, int],
    batch_size: int,
    layer: int,
    site: str,
    basis: Any,
    condition: str,
    donor_coordinates: Any | None = None,
) -> list[dict[str, Any]]:
    if donor_coordinates is not None and int(donor_coordinates.shape[0]) != len(rows):
        raise ValueError("donor coordinates must align one-to-one with rows")
    state: dict[str, Any] = {
        "positions": None,
        "donor": None,
        "diagnostics": None,
    }

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        positions = state["positions"]
        if positions is None:
            raise RuntimeError("Intervention hook ran without token positions")
        positions = positions.to(hidden_states.device)
        row_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        hidden = hidden_states[row_indices, positions].to(torch.float32)
        u_here = basis.to(hidden_states.device, dtype=torch.float32)
        coordinates = hidden @ u_here
        projection = coordinates @ u_here.T
        if state["donor"] is None:
            patched = hidden - projection
            coordinate_change = coordinates
        else:
            donor = state["donor"].to(hidden_states.device, dtype=torch.float32)
            patched = hidden - projection + donor @ u_here.T
            coordinate_change = donor - coordinates
        delta = patched - hidden
        updated = hidden_states.clone()
        updated[row_indices, positions] = patched.to(hidden_states.dtype)
        state["diagnostics"] = {
            "hidden_l2": hidden.norm(dim=1).detach().cpu(),
            "projected_l2": projection.norm(dim=1).detach().cpu(),
            "intervention_l2": delta.norm(dim=1).detach().cpu(),
            "coordinate_change_l2": coordinate_change.norm(dim=1).detach().cpu(),
        }
        if isinstance(output, tuple):
            return (updated,) + tuple(output[1:])
        return updated

    handle = layers[layer].register_forward_hook(hook)
    scored: list[dict[str, Any]] = []
    try:
        offset = 0
        batches = row_batches(rows, batch_size)
        batches = progress_iter(
            batches,
            total=ceil(len(rows) / batch_size),
            desc=condition,
        )
        for batch_rows in batches:
            texts = [str(row["base_prompt"]) for row in batch_rows]
            encoded = encode_to_device(tokenizer, texts, device)
            state["positions"] = torch.tensor(
                resolve_positions(tokenizer, texts, batch_rows, site),
                dtype=torch.long,
                device=device,
            )
            if donor_coordinates is None:
                state["donor"] = None
            else:
                state["donor"] = donor_coordinates[
                    offset : offset + len(batch_rows)
                ]
            state["diagnostics"] = None
            offset += len(batch_rows)
            with torch.no_grad():
                logits = model(**encoded, use_cache=False).logits
                final = encoded["attention_mask"].sum(dim=1) - 1
                indices = torch.arange(logits.shape[0], device=logits.device)
                next_logits = logits[indices, final.to(logits.device)]
            diagnostics = state["diagnostics"]
            if diagnostics is None:
                raise RuntimeError(f"Layer {layer} intervention hook did not run")
            batch_metrics = next_token_metrics(
                torch=torch,
                next_logits=next_logits,
                rows=batch_rows,
                label_token_ids=label_token_ids,
                tokenizer=tokenizer,
            )
            for index, metrics in enumerate(batch_metrics):
                for key, values in diagnostics.items():
                    metrics[key] = float(values[index])
                scored.append(metrics)
    finally:
        handle.remove()
        state["positions"] = None
        state["donor"] = None
        state["diagnostics"] = None
    return scored


def collect_subspace_coordinates(
    *,
    model: Any,
    layers: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    layer: int,
    site: str,
    basis: Any,
) -> Any:
    state: dict[str, Any] = {"positions": None, "coordinates": None}

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        positions = state["positions"].to(hidden_states.device)
        indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        hidden = hidden_states[indices, positions].to(torch.float32)
        u_here = basis.to(hidden_states.device, dtype=torch.float32)
        state["coordinates"] = (hidden @ u_here).detach().cpu()
        return output

    handle = layers[layer].register_forward_hook(hook)
    coordinates = []
    core_model = getattr(model, "model", model)
    try:
        batches = row_batches(rows, batch_size)
        batches = progress_iter(
            batches,
            total=ceil(len(rows) / batch_size),
            desc=f"collect coordinates L{layer}/{site}",
        )
        for batch_rows in batches:
            texts = [str(row["base_prompt"]) for row in batch_rows]
            encoded = encode_to_device(tokenizer, texts, device)
            state["positions"] = torch.tensor(
                resolve_positions(tokenizer, texts, batch_rows, site),
                dtype=torch.long,
                device=device,
            )
            state["coordinates"] = None
            with torch.no_grad():
                core_model(**encoded, use_cache=False)
            if state["coordinates"] is None:
                raise RuntimeError(f"Layer {layer} coordinate hook did not run")
            coordinates.append(state["coordinates"])
    finally:
        handle.remove()
        state["positions"] = None
        state["coordinates"] = None
    return torch.cat(coordinates, dim=0)


def load_rotation_basis(
    *,
    rotation_dir: str | Path,
    expected: dict[str, Any],
    hidden_size: int,
    torch: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    path = Path(rotation_dir)
    metadata_path = path / "rotation_weight_metadata.json"
    weight_path = path / "rotation_weight.npy"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in ("target_var", "model_name", "layer", "rank", "component"):
        if str(metadata.get(field)) != str(expected[field]):
            raise ValueError(
                f"{path}: metadata {field}={metadata.get(field)!r}, "
                f"expected={expected[field]!r}"
            )
    allowed_sites = {str(value) for value in expected["allowed_metadata_sites"]}
    if str(metadata.get("site")) not in allowed_sites:
        raise ValueError(
            f"{path}: metadata site={metadata.get('site')!r}, "
            f"expected one of {sorted(allowed_sites)}"
        )
    raw = np.load(weight_path)
    rank = int(expected["rank"])
    if tuple(raw.shape) != (hidden_size, rank):
        raise ValueError(
            f"{path}: rotation shape={tuple(raw.shape)}, "
            f"expected={(hidden_size, rank)}"
        )
    tensor = torch.tensor(raw, dtype=torch.float32, device=device)
    matrix_rank = int(torch.linalg.matrix_rank(tensor).detach().cpu())
    if matrix_rank != rank:
        raise ValueError(f"{path}: rotation rank={matrix_rank}, expected={rank}")
    identity = torch.eye(rank, dtype=torch.float32, device=device)
    raw_gram_error = float((tensor.T @ tensor - identity).abs().max().detach().cpu())
    basis = orthonormalize_basis(torch, tensor)
    qr_gram_error = float((basis.T @ basis - identity).abs().max().detach().cpu())
    provenance = {
        "directory": str(path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "matrix_rank": matrix_rank,
        "raw_gram_max_abs_error": raw_gram_error,
        "qr_gram_max_abs_error": qr_gram_error,
        "metadata": metadata,
    }
    return basis, provenance


def basis_overlap_metrics(torch: Any, rho_basis: Any, m_basis: Any) -> dict[str, Any]:
    if tuple(rho_basis.shape) != tuple(m_basis.shape):
        raise ValueError("rho and m bases must have identical shape for overlap audit")
    hidden, rank = (int(value) for value in rho_basis.shape)
    cross = rho_basis.T @ m_basis
    singular = torch.linalg.svdvals(cross)
    overlap = float((cross.square().sum() / rank).detach().cpu())
    random_baseline = rank / hidden
    return {
        "hidden_size": hidden,
        "rank": rank,
        "normalized_frobenius_overlap": overlap,
        "random_baseline_rank_over_hidden": random_baseline,
        "overlap_to_random_ratio": overlap / random_baseline,
        "max_principal_cosine": float(singular.max().detach().cpu()),
        "n_principal_cosine_gt_0_7": int((singular > 0.7).sum().detach().cpu()),
    }
