"""pyvene runner for atomic DAS experiments."""

from __future__ import annotations

import json
import random
from math import ceil
from pathlib import Path
from typing import Any, Iterable

from .das_spans import resolve_token_site
from .io_utils import write_rows_csv
from .model import DEFAULT_CACHE_DIR, LabelTokenStyle, normalize_cache_dir, progress_iter, resolve_label_tokens, resolve_torch_dtype

LABEL_TO_INDEX = {"T": 0, "F": 1, "U": 2}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def run_pyvene_das(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    model_name: str,
    target_var: str,
    layer: int,
    rank: int,
    component: str = "block_output",
    site: str = "row",
    steps: int = 1000,
    batch_size: int = 16,
    eval_batch_size: int | None = None,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    device_map: str | None = "auto",
    torch_dtype: str | None = "auto",
    label_token_style: LabelTokenStyle = "auto",
    trust_remote_code: bool = False,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    local_files_only: bool = False,
    eval_interval: int = 100,
    save_intervention: bool = False,
    export_rotation_weight: bool = False,
    train_control_types: list[str] | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    eval_train: bool = True,
) -> dict[str, Any]:
    torch, pv, AutoModelForCausalLM, AutoTokenizer = import_runtime()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_batch_size = eval_batch_size or batch_size

    target_rows = [row for row in rows if row.get("target_var") == target_var]
    if not target_rows:
        raise ValueError(f"No DAS rows found for target_var={target_var!r}")

    all_train_rows = rows_for_split(target_rows, "train")
    train_rows = filter_train_rows(all_train_rows, target_var, train_control_types)
    val_rows = rows_for_split(target_rows, "val") or all_train_rows
    test_rows = rows_for_split(target_rows, "test") or val_rows
    if not train_rows:
        raise ValueError(f"No train rows found for target_var={target_var!r}")

    if model is None or tokenizer is None:
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
    intervenable = build_intervenable(pv, model, layer=layer, rank=rank, component=component)
    input_device = get_input_device(model, torch, device)
    set_intervenable_device(intervenable, input_device)

    params = [param for param in intervenable.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("pyvene did not expose any trainable intervention parameters")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    step_iter = progress_iter(
        range(1, steps + 1),
        total=steps,
        desc=f"DAS train {target_var}@L{layer}/{site}",
        unit="step",
    )
    for step in step_iter:
        batch_rows = sample_with_replacement(train_rows, batch_size, rng)
        batch = collate_rows(
            rows=batch_rows,
            tokenizer=tokenizer,
            torch=torch,
            device=input_device,
            site_override=site,
        )
        outputs = call_intervenable(intervenable, batch)
        tfu_logits = tfu_logits_from_outputs(torch, outputs, batch["input_lengths"], label_tokens.token_ids)
        loss = torch.nn.functional.cross_entropy(tfu_logits, batch["targets"])
        optimizer.zero_grad()
        loss.backward()
        grad_norm = parameter_grad_norm(params)
        optimizer.step()
        #this point we orthonorgal
        orthonormalize_low_rank_interventions(intervenable, torch)

        if eval_interval > 0 and (step == 1 or step % eval_interval == 0 or step == steps):
            val_metrics, _ = evaluate_pyvene_das(
                intervenable=intervenable,
                rows=val_rows,
                tokenizer=tokenizer,
                torch=torch,
                device=input_device,
                label_token_ids=label_tokens.token_ids,
                batch_size=eval_batch_size,
                site_override=site,
                progress_desc=f"Eval val step {step}",
            )
            loss_value = float(loss.detach().cpu())
            entry = {"step": step, "loss": loss_value, "grad_norm": grad_norm, "split": "val", **val_metrics}
            history.append(entry)
            if hasattr(step_iter, "set_postfix"):
                val_iia = val_metrics.get("IIA")
                step_iter.set_postfix(
                    loss=f"{loss_value:.4g}",
                    grad=f"{grad_norm:.3g}",
                    val_iia="None" if val_iia is None else f"{float(val_iia):.3f}",
                )
            print(json.dumps(entry, sort_keys=True))

    if eval_train:
        train_metrics, _ = evaluate_pyvene_das(
            intervenable, train_rows, tokenizer, torch, input_device, label_tokens.token_ids, eval_batch_size, site, "Final train"
        )
    else:
        train_metrics = None
    val_metrics, val_scored = evaluate_pyvene_das(
        intervenable, val_rows, tokenizer, torch, input_device, label_tokens.token_ids, eval_batch_size, site, "Final val"
    )
    test_metrics, test_scored = evaluate_pyvene_das(
        intervenable, test_rows, tokenizer, torch, input_device, label_tokens.token_ids, eval_batch_size, site, "Final test"
    )

    write_rows_csv(val_scored, output_dir / "val_scored.csv")
    write_rows_csv(test_scored, output_dir / "test_scored.csv")
    rotation_export = None
    if export_rotation_weight:
        rotation_export = export_rotation_weights(
            intervenable=intervenable,
            torch=torch,
            output_dir=output_dir,
            metadata={
                "target_var": target_var,
                "model_name": model_name,
                "layer": layer,
                "rank": rank,
                "component": component,
                "site": site,
                "steps": steps,
                "learning_rate": learning_rate,
            },
        )
    summary = {
        "target_var": target_var,
        "model_name": model_name,
        "layer": layer,
        "rank": rank,
        "component": component,
        "site": site,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "rotation_export": rotation_export,
        "label_token_style": label_tokens.style,
        "n_train": len(train_rows),
        "n_train_all": len(all_train_rows),
        "train_control_types": resolved_train_control_types(target_var, train_control_types),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    if save_intervention:
        save_dir = output_dir / "intervention"
        save_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(intervenable, "save"):
            intervenable.save(str(save_dir))
        else:
            torch.save(intervenable.state_dict(), save_dir / "intervenable_state.pt")

    return summary


def evaluate_pyvene_das(
    intervenable: Any,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    torch: Any,
    device: Any,
    label_token_ids: dict[str, int],
    batch_size: int,
    site_override: str,
    progress_desc: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    row_chunks = chunks(rows, batch_size)
    if progress_desc:
        row_chunks = progress_iter(row_chunks, total=ceil(len(rows) / batch_size), desc=progress_desc)
    label_id_set = set(label_token_ids.values())
    for batch_rows in row_chunks:
        batch = collate_rows(batch_rows, tokenizer, torch, device, site_override)
        with torch.no_grad():
            outputs = call_intervenable(intervenable, batch)
            next_logits = next_token_logits(torch, outputs, batch["input_lengths"])
            tfu_logits = tfu_logits_from_next(torch, next_logits, label_token_ids)
            global_top_ids = next_logits.argmax(dim=-1)
        preds = tfu_logits.argmax(dim=-1)
        for row, row_logits, pred_idx, top_id in zip(batch_rows, tfu_logits, preds, global_top_ids):
            logit_t = float(row_logits[0].detach().cpu())
            logit_f = float(row_logits[1].detach().cpu())
            logit_u = float(row_logits[2].detach().cpu())
            pred_label = INDEX_TO_LABEL[int(pred_idx.detach().cpu())]
            top_token_id = int(top_id.detach().cpu())
            out = dict(row)
            out.update(
                {
                    "logit_T": logit_t,
                    "logit_F": logit_f,
                    "logit_U": logit_u,
                    "R": logit_t - logit_f,
                    "U_gap": logit_u - max(logit_t, logit_f),
                    "pred_label": pred_label,
                    "is_correct": int(pred_label == row["target_label"]),
                    "global_top_token_id": top_token_id,
                    "global_top_token": tokenizer.decode([top_token_id]),
                    "global_top_in_TFU": int(top_token_id in label_id_set),
                }
            )
            scored.append(out)
    return summarize_scored(scored), scored


def collate_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    torch: Any,
    device: Any,
    site_override: str,
) -> dict[str, Any]:
    base_texts = [str(row["base_prompt"]) for row in rows]
    source_texts = [str(row["source_prompt"]) for row in rows]
    base_inputs = encode_to_device(tokenizer, base_texts, device)
    source_inputs = encode_to_device(tokenizer, source_texts, device)
    base_positions: list[int] = []
    source_positions: list[int] = []
    targets: list[int] = []
    for row, base_text, source_text in zip(rows, base_texts, source_texts):
        base_site, source_site = sites_for_row(row, site_override)
        base_positions.append(resolve_token_site(tokenizer, base_text, row, "base", base_site))
        source_positions.append(resolve_token_site(tokenizer, source_text, row, "source", source_site))
        targets.append(LABEL_TO_INDEX[str(row["target_label"])])
    input_lengths = base_inputs["attention_mask"].sum(dim=1)
    return {
        "base": base_inputs,
        "sources": source_inputs,
        "unit_locations": make_unit_locations(source_positions, base_positions),
        "input_lengths": input_lengths,
        "targets": torch.tensor(targets, dtype=torch.long, device=device),
    }


def call_intervenable(intervenable: Any, batch: dict[str, Any]) -> Any:
    result = intervenable(
        base=batch["base"],
        sources=[batch["sources"]],
        unit_locations=batch["unit_locations"],
        output_original_output=True,
    )
    if isinstance(result, tuple) and len(result) >= 2:
        return result[1]
    return result


def tfu_logits_from_outputs(torch: Any, outputs: Any, input_lengths: Any, label_token_ids: dict[str, int]) -> Any:
    next_logits = next_token_logits(torch, outputs, input_lengths)
    return tfu_logits_from_next(torch, next_logits, label_token_ids)


def next_token_logits(torch: Any, outputs: Any, input_lengths: Any) -> Any:
    logits = extract_logits(outputs)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    final_idx = input_lengths.to(logits.device) - 1
    return logits[batch_idx, final_idx]


def tfu_logits_from_next(torch: Any, next_logits: Any, label_token_ids: dict[str, int]) -> Any:
    return torch.stack(
        [
            next_logits[:, label_token_ids["T"]],
            next_logits[:, label_token_ids["F"]],
            next_logits[:, label_token_ids["U"]],
        ],
        dim=-1,
    )


def extract_logits(outputs: Any) -> Any:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if hasattr(item, "logits"):
                return item.logits
    raise TypeError(f"Could not extract logits from pyvene output of type {type(outputs)!r}")


def make_unit_locations(source_positions: list[int], base_positions: list[int]) -> dict[str, Any]:
    # pyvene expects each side as num_interventions x batch x max_units.
    source_locs = [[[int(pos)] for pos in source_positions]]
    base_locs = [[[int(pos)] for pos in base_positions]]
    return {"sources->base": (source_locs, base_locs)}


def sites_for_row(row: dict[str, Any], site_override: str) -> tuple[str, str]:
    if site_override and site_override != "row":
        return site_override, site_override
    return str(row["base_site"]), str(row["source_site"])


def encode_to_device(tokenizer: Any, texts: list[str], device: Any) -> dict[str, Any]:
    encoded = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


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
    hub_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
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
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def build_intervenable(pv: Any, model: Any, layer: int, rank: int, component: str) -> Any:
    register_pyvene_model_mappings(model)
    intervention_type = make_stable_low_rank_intervention_type(pv)
    try:
        representation = pv.RepresentationConfig(
            layer,
            component,
            "pos",
            1,
            low_rank_dimension=rank,
        )
        config = pv.IntervenableConfig(
            model_type=type(model),
            representations=[representation],
            intervention_types=intervention_type,
        )
    except Exception:
        config = pv.IntervenableConfig(
            {
                "layer": layer,
                "component": component,
                "unit": "pos",
                "max_number_of_units": 1,
                "low_rank_dimension": rank,
            },
            intervention_types=intervention_type,
        )
    try:
        intervenable = pv.IntervenableModel(config, model=model, use_fast=False)
    except TypeError:
        intervenable = pv.IntervenableModel(config, model=model)
    if hasattr(intervenable, "disable_model_gradients"):
        intervenable.disable_model_gradients()
    return intervenable


def make_stable_low_rank_intervention_type(pv: Any) -> Any:
    """Return a low-rank rotated-space intervention with stable gradients.

    pyvene 0.1.x's LowRankRotatedSpaceIntervention wraps the low-rank matrix in
    torch.nn.utils.parametrizations.orthogonal. With current torch/pyvene, that
    can produce zero gradients after the first optimizer update. This class keeps
    the pyvene intervention contract but uses a plain trainable low-rank matrix.
    The train loop re-orthonormalizes it after optimizer steps.
    """

    import torch
    from pyvene.models.interventions import _can_use_fast
    from pyvene.models.layers import LowRankRotateLayer

    class StableLowRankRotatedSpaceIntervention(
        pv.TrainableIntervention,
        pv.DistributedRepresentationIntervention,
    ):
        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self.rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])

        def forward(self, base: Any, source: Any, subspaces: Any = None, **kwargs: Any) -> Any:
            rotated_base = self.rotate_layer(base)
            rotated_source = self.rotate_layer(source)
            weight = self.rotate_layer.weight
            if subspaces is not None:
                if self.use_fast or _can_use_fast(subspaces):
                    if self.subspace_partition is None:
                        sel_subspace_indices = subspaces[0]
                    else:
                        sel_subspace_indices = []
                        for subspace in subspaces[0]:
                            sel_subspace_indices.extend(self.subspace_partition[subspace])
                    diff = rotated_source - rotated_base
                    batched_subspace = diff[..., sel_subspace_indices].unsqueeze(dim=1)
                    batched_weights = weight[..., sel_subspace_indices].T
                    output = base + torch.matmul(batched_subspace, batched_weights).squeeze(dim=1)
                else:
                    if self.subspace_partition is None:
                        raise ValueError("subspace_partition is required for non-fast subspace intervention")
                    diff = rotated_source - rotated_base
                    batched_subspace = []
                    batched_weights = []
                    for example_i in range(len(subspaces)):
                        sel_subspace_indices = []
                        for subspace in subspaces[example_i]:
                            sel_subspace_indices.extend(self.subspace_partition[subspace])
                        batched_subspace.append(diff[example_i, sel_subspace_indices].unsqueeze(dim=0))
                        batched_weights.append(weight[..., sel_subspace_indices].T)
                    output = base + torch.matmul(
                        torch.stack(batched_subspace, dim=0),
                        torch.stack(batched_weights, dim=0),
                    ).squeeze(dim=1)
            else:
                output = base + torch.matmul(rotated_source - rotated_base, weight.T)
            return output.to(base.dtype)

        def __str__(self) -> str:
            return "StableLowRankRotatedSpaceIntervention()"

    return StableLowRankRotatedSpaceIntervention


def register_pyvene_model_mappings(model: Any) -> None:
    """Register HF model classes missing from pyvene 0.1.x mappings.

    pyvene 0.1.8 has Qwen2 mappings but not the HF Qwen3 classes, and no
    Granite mappings. Both families keep the llama-style decoder block paths
    (model.layers[N]) for the block components used by these DAS runs.
    """

    try:
        from pyvene.models import modeling_utils as mu
    except ImportError:
        return

    model_type = type(model)
    if model_type in mu.type_to_dimension_mapping and model_type in mu.type_to_module_mapping:
        return

    config_model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    class_name = model_type.__name__.lower()
    tag = f"{config_model_type}:{class_name}"
    if "qwen3" in tag:
        source_name = "Qwen2ForCausalLM"
    elif "granite" in tag:
        source_name = "LlamaForCausalLM"
    else:
        return
    source_type = find_mapping_type(mu, source_name)
    if source_type is not None:
        mu.type_to_dimension_mapping[model_type] = dict(mu.type_to_dimension_mapping[source_type])
        mu.type_to_module_mapping[model_type] = dict(mu.type_to_module_mapping[source_type])


def find_mapping_type(modeling_utils: Any, class_name: str) -> Any | None:
    for mapped_type in modeling_utils.type_to_dimension_mapping:
        if getattr(mapped_type, "__name__", "") == class_name:
            return mapped_type
    return None


def set_intervenable_device(intervenable: Any, device: Any) -> None:
    if hasattr(intervenable, "set_device"):
        intervenable.set_device(str(device))


def get_input_device(model: Any, torch: Any, requested_device: str | None) -> Any:
    if requested_device:
        return torch.device(requested_device)
    return next(model.parameters()).device


def rows_for_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split") == split]


def filter_train_rows(rows: list[dict[str, Any]], target_var: str, train_control_types: list[str] | None) -> list[dict[str, Any]]:
    controls = resolved_train_control_types(target_var, train_control_types)
    if controls == ["all"]:
        return rows
    allowed = set(controls)
    return [row for row in rows if str(row.get("control_type")) in allowed]


def resolved_train_control_types(target_var: str, train_control_types: list[str] | None) -> list[str]:
    values = train_control_types or ["auto"]
    if len(values) == 1 and values[0] == "auto":
        if target_var in {"pc", "pi"}:
            return ["main"]
        if target_var == "m":
            return ["match_to_nomatch", "nomatch_to_match"]
    return values


def orthonormalize_low_rank_interventions(intervenable: Any, torch: Any) -> None:
    with torch.no_grad():
        interventions = getattr(intervenable, "interventions", {})
        for intervention in interventions.values():
            rotate_layer = getattr(intervention, "rotate_layer", None)
            weight = getattr(rotate_layer, "weight", None)
            if weight is None or len(getattr(weight, "shape", ())) != 2:
                continue
            if weight.shape[0] < weight.shape[1]:
                continue
            q, _ = torch.linalg.qr(weight.detach().float(), mode="reduced")
            weight.copy_(q.to(device=weight.device, dtype=weight.dtype))


def export_rotation_weights(intervenable: Any, torch: Any, output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    weights = collect_rotation_weights(intervenable)
    if not weights:
        raise RuntimeError("No rotate_layer.weight tensors found to export")

    output_dir.mkdir(parents=True, exist_ok=True)
    export_meta = dict(metadata)
    export_meta["weights"] = {}

    if len(weights) == 1:
        name, weight = next(iter(weights.items()))
        tensor = weight.detach().float().cpu()
        pt_path = output_dir / "rotation_weight.pt"
        npy_path = output_dir / "rotation_weight.npy"
        torch.save(tensor, pt_path)
        save_tensor_npy(tensor, npy_path)
        export_meta["weights"][name] = {
            "shape": list(tensor.shape),
            "pt_path": str(pt_path),
            "npy_path": str(npy_path),
        }
    else:
        pt_path = output_dir / "rotation_weights.pt"
        tensor_dict = {name: weight.detach().float().cpu() for name, weight in weights.items()}
        torch.save(tensor_dict, pt_path)
        export_meta["pt_path"] = str(pt_path)
        for name, tensor in tensor_dict.items():
            safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
            npy_path = output_dir / f"rotation_weight_{safe_name}.npy"
            save_tensor_npy(tensor, npy_path)
            export_meta["weights"][name] = {
                "shape": list(tensor.shape),
                "npy_path": str(npy_path),
            }

    metadata_path = output_dir / "rotation_weight_metadata.json"
    metadata_path.write_text(json.dumps(to_jsonable(export_meta), indent=2), encoding="utf-8")
    export_meta["metadata_path"] = str(metadata_path)
    return export_meta


def collect_rotation_weights(intervenable: Any) -> dict[str, Any]:
    weights = {}
    interventions = getattr(intervenable, "interventions", {})
    for name, intervention in interventions.items():
        rotate_layer = getattr(intervention, "rotate_layer", None)
        weight = getattr(rotate_layer, "weight", None)
        if weight is not None:
            weights[str(name)] = weight
    return weights


def save_tensor_npy(tensor: Any, path: Path) -> None:
    try:
        import numpy as np
    except ImportError:
        return
    np.save(path, tensor.numpy())


def parameter_grad_norm(params: list[Any]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().pow(2).sum().cpu())
    return total ** 0.5


def sample_with_replacement(rows: list[dict[str, Any]], batch_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) >= batch_size:
        return rng.sample(rows, batch_size)
    return [rng.choice(rows) for _ in range(batch_size)]


def chunks(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def summarize_scored(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "IIA": None}
    correct = [float(row["is_correct"]) for row in rows]
    pred_counts = {label: sum(1 for row in rows if row["pred_label"] == label) for label in LABEL_TO_INDEX}
    by_control: dict[str, dict[str, Any]] = {}
    controls = sorted({str(row.get("control_type", "")) for row in rows})
    for control in controls:
        control_rows = [row for row in rows if str(row.get("control_type", "")) == control]
        by_control[control] = {
            "n": len(control_rows),
            "IIA": mean(float(row["is_correct"]) for row in control_rows),
            "mean_R": mean(float(row["R"]) for row in control_rows),
            "mean_U_gap": mean(float(row["U_gap"]) for row in control_rows),
            "global_top_in_TFU_rate": mean(
                float(row["global_top_in_TFU"]) for row in control_rows if "global_top_in_TFU" in row
            ),
        }
    return {
        "n": len(rows),
        "IIA": mean(correct),
        "mean_R": mean(float(row["R"]) for row in rows),
        "mean_U_gap": mean(float(row["U_gap"]) for row in rows),
        "T_rate": pred_counts["T"] / len(rows),
        "F_rate": pred_counts["F"] / len(rows),
        "U_rate": pred_counts["U"] / len(rows),
        "global_top_in_TFU_rate": mean(float(row["global_top_in_TFU"]) for row in rows if "global_top_in_TFU" in row),
        "by_control": by_control,
    }


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def import_runtime():
    try:
        import torch
        import pyvene as pv
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install pyvene, torch, and transformers before running DAS. "
            "The behavioral scorer does not require pyvene."
        ) from exc
    return torch, pv, AutoModelForCausalLM, AutoTokenizer


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
