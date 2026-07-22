"""Condition semantics for joint ``m``/``rho`` knockout and downstream rescue."""

from __future__ import annotations

from dataclasses import dataclass

from .joint_gate_data import label_from_m_rho


@dataclass(frozen=True)
class RescueCondition:
    name: str
    claim_patch: str
    answer_restore: str = "none"
    family: str = "control"


RESCUE_CONDITIONS = (
    RescueCondition("none", "none"),
    RescueCondition("claim_m_flip", "m_flip", family="knockout"),
    RescueCondition("claim_rho_flip", "rho_flip", family="knockout"),
    RescueCondition("claim_both_flip", "both_flip", family="knockout"),
    RescueCondition("claim_m_same", "m_same", family="purity"),
    RescueCondition("claim_rho_same", "rho_same", family="purity"),
    RescueCondition("claim_both_same", "both_same", family="purity"),
    RescueCondition("claim_random_m_flip", "random_m_flip", family="random"),
    RescueCondition("claim_random_rho_flip", "random_rho_flip", family="random"),
    RescueCondition("claim_random_both_flip", "random_both_flip", family="random"),
    RescueCondition(
        "claim_m_flip_answer_m_restore",
        "m_flip",
        "m",
        "rescue",
    ),
    RescueCondition(
        "claim_rho_flip_answer_rho_restore",
        "rho_flip",
        "rho",
        "rescue",
    ),
    RescueCondition(
        "claim_both_flip_answer_m_restore",
        "both_flip",
        "m",
        "selective_rescue",
    ),
    RescueCondition(
        "claim_both_flip_answer_rho_restore",
        "both_flip",
        "rho",
        "selective_rescue",
    ),
    RescueCondition(
        "claim_both_flip_answer_both_restore",
        "both_flip",
        "both",
        "rescue",
    ),
    RescueCondition(
        "claim_both_flip_answer_random_restore",
        "both_flip",
        "random_both",
        "random_rescue",
    ),
    RescueCondition(
        "answer_both_restore_only",
        "none",
        "both",
        "purity",
    ),
)


def state_after_condition(condition: RescueCondition, m_base: int, rho_base: int) -> tuple[int, int]:
    """Return the high-level state predicted after corruption and rescue."""

    m_value = int(m_base)
    rho_value = int(rho_base)

    if condition.claim_patch in {"m_flip", "both_flip"}:
        m_value = 1 - m_value
    if condition.claim_patch in {"rho_flip", "both_flip"}:
        rho_value = -rho_value

    if condition.answer_restore in {"m", "both"}:
        m_value = int(m_base)
    if condition.answer_restore in {"rho", "both"}:
        rho_value = int(rho_base)

    # Same-value and random controls do not have a semantic intervention target.
    if condition.claim_patch in {
        "m_same",
        "rho_same",
        "both_same",
        "random_m_flip",
        "random_rho_flip",
        "random_both_flip",
    }:
        m_value, rho_value = int(m_base), int(rho_base)
    if condition.answer_restore == "random_both":
        m_value, rho_value = 1 - int(m_base), -int(rho_base)

    return m_value, rho_value


def expected_label(condition: RescueCondition, m_base: int, rho_base: int) -> str:
    m_value, rho_value = state_after_condition(condition, m_base, rho_base)
    return label_from_m_rho(m_value, rho_value)


def condition_by_name(name: str) -> RescueCondition:
    for condition in RESCUE_CONDITIONS:
        if condition.name == name:
            return condition
    raise KeyError(name)
