"""Model scoring utilities for zero-shot T/F/U prompts."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal

from .base import LABELS

LabelTokenStyle = Literal["auto", "bare", "space", "newline"]
DEFAULT_CACHE_DIR = "/workspace/huggingface/hub"


@dataclass(frozen=True)
class LabelTokenConfig:
    token_ids: dict[str, int]
    token_texts: dict[str, str]
    style: str


class TFULogitScorer:
    """Scores the label logits for T, F, and U."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        device_map: str | None = "auto",
        torch_dtype: str | None = "auto",
        label_token_style: LabelTokenStyle = "auto",
        trust_remote_code: bool = False,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers before running model scoring") from exc

        self.torch = torch
        cache_dir = normalize_cache_dir(cache_dir)
        hub_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            hub_kwargs["cache_dir"] = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = dict(hub_kwargs)
        resolved_dtype = resolve_torch_dtype(torch, torch_dtype)
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if device_map and device_map != "none":
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if (not device_map or device_map == "none") and device:
            self.model.to(device)
        elif (not device_map or device_map == "none") and device is None:
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.label_tokens = resolve_label_tokens(self.tokenizer, label_token_style)

    @property
    def input_device(self):
        return next(self.model.parameters()).device

    def score_prompts(self, prompts: Iterable[str], batch_size: int = 8, show_progress: bool = True) -> list[dict[str, Any]]:
        prompts = list(prompts)
        results: list[dict[str, Any]] = []
        starts = range(0, len(prompts), batch_size)
        if show_progress:
            starts = progress_iter(starts, total=ceil(len(prompts) / batch_size), desc="Scoring prompts")
        for start in starts:
            batch = prompts[start : start + batch_size]
            encoded = self.tokenizer(batch, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
            with self.torch.no_grad():
                outputs = self.model(**encoded)
            last_indices = encoded["attention_mask"].sum(dim=1) - 1
            batch_indices = self.torch.arange(len(batch), device=self.input_device)
            logits = outputs.logits[batch_indices, last_indices]
            for row_logits in logits:
                results.append(logits_to_metrics(row_logits, self.label_tokens))
        return results


def evaluate_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    batch_size: int = 8,
    device: str | None = None,
    device_map: str | None = "auto",
    torch_dtype: str | None = "auto",
    label_token_style: LabelTokenStyle = "auto",
    trust_remote_code: bool = False,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    local_files_only: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    scorer = TFULogitScorer(
        model_name=model_name,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
        label_token_style=label_token_style,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    scored = scorer.score_prompts((row["prompt"] for row in rows), batch_size=batch_size, show_progress=show_progress)
    for row, metrics in zip(rows, scored):
        row.update(metrics)
        row["is_correct"] = int(row["pred_label"] == row["expected_label"])
    return rows


def logits_to_metrics(row_logits: Any, label_tokens: LabelTokenConfig) -> dict[str, Any]:
    values = {label: float(row_logits[token_id].detach().cpu()) for label, token_id in label_tokens.token_ids.items()}
    pred_label = max(values, key=values.get)
    r_value = values["T"] - values["F"]
    u_gap = values["U"] - max(values["T"], values["F"])
    return {
        "label_token_T": label_tokens.token_texts["T"],
        "label_token_F": label_tokens.token_texts["F"],
        "label_token_U": label_tokens.token_texts["U"],
        "label_token_style": label_tokens.style,
        "logit_T": values["T"],
        "logit_F": values["F"],
        "logit_U": values["U"],
        "R": r_value,
        "U_gap": u_gap,
        "pred_label": pred_label,
    }


def resolve_label_tokens(tokenizer: Any, style: LabelTokenStyle = "auto") -> LabelTokenConfig:
    styles = ["bare", "space", "newline"] if style == "auto" else [style]
    variants = {
        "bare": lambda label: label,
        "space": lambda label: f" {label}",
        "newline": lambda label: f"\n{label}",
    }
    for candidate_style in styles:
        token_ids: dict[str, int] = {}
        token_texts: dict[str, str] = {}
        for label in LABELS:
            text = variants[candidate_style](label)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                break
            token_ids[label] = ids[0]
            token_texts[label] = text
        if len(token_ids) == len(LABELS):
            return LabelTokenConfig(token_ids=token_ids, token_texts=token_texts, style=candidate_style)
    raise ValueError(
        "Could not resolve T/F/U as single tokens. Try a different tokenizer or inspect "
        "tokenizer.encode('T', add_special_tokens=False)."
    )


def resolve_torch_dtype(torch: Any, dtype: str | None) -> Any:
    if dtype is None or dtype == "none":
        return None
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]


def progress_iter(iterable: Iterable[int], total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="batch")


def normalize_cache_dir(cache_dir: str | None) -> str | None:
    if cache_dir is None:
        return None
    value = str(cache_dir).strip()
    if not value or value.lower() in {"none", "default"}:
        return None
    return value
