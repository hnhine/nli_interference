"""Pure helpers for DAS event-match transfer experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


MISMATCH_TYPES = ("object", "subject", "verb", "no_overlap")
CORE_M_CONTROLS = ("match_to_nomatch", "nomatch_to_match")
M2N = "match_to_nomatch"
N2M = "nomatch_to_match"


def match_pair_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Key shared by the two exact reverse m minimal-pair rows."""
    return (
        str(row.get("base_event_id", "")),
        str(row.get("matched_idx", "")),
        str(row.get("p_i_base", "")),
        str(row.get("p_c_base", "")),
        str(row.get("mismatch_type", "")),
    )


def validate_and_pair_core_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], dict[str, dict[str, Any]]]:
    """Pair m core rows and assert the generator's exact-reversal invariant."""
    pairs: dict[tuple[str, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        control = str(row.get("control_type", ""))
        if control not in CORE_M_CONTROLS:
            continue
        key = match_pair_key(row)
        if control in pairs[key]:
            raise ValueError(f"Duplicate {control} row for m key {key}")
        pairs[key][control] = row

    if not pairs:
        raise ValueError("No core m rows found")
    for key, pair in pairs.items():
        if set(pair) != set(CORE_M_CONTROLS):
            raise ValueError(f"Incomplete m reverse pair for {key}: found {sorted(pair)}")
        m2n, n2m = pair[M2N], pair[N2M]
        checks = {
            "m2n.base == n2m.source": m2n.get("base_prompt") == n2m.get("source_prompt"),
            "m2n.source == n2m.base": m2n.get("source_prompt") == n2m.get("base_prompt"),
            "m2n.base_label == n2m.source_label": m2n.get("base_label") == n2m.get("source_label"),
            "m2n.source_label == n2m.base_label": m2n.get("source_label") == n2m.get("base_label"),
            "m2n has m=1 base": str(m2n.get("m_base")) == "1",
            "n2m has m=0 base": str(n2m.get("m_base")) == "0",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"m reverse-pair regression for {key}: {failed}")
    return dict(pairs)


def source_as_base(row: dict[str, Any]) -> dict[str, Any]:
    """Promote a stored source prompt/state to the base-side interface."""
    out = dict(row)
    for key, value in row.items():
        if key.startswith("source_"):
            out[f"base_{key[len('source_'):]}"] = value
    for stem in ("m", "p_i", "p_c"):
        source_key = f"{stem}_src"
        if source_key in row:
            out[f"{stem}_base"] = row[source_key]
    out["base_prompt"] = row["source_prompt"]
    out["base_label"] = row["source_label"]
    out["base_site"] = row["source_site"]
    out["eval_side"] = "source"
    out["sample_id"] = f"{row.get('sample_id', '')}::source"
    return out
