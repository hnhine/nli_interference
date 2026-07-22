"""Linear-algebra helpers for simultaneous fixed-subspace interventions."""

from __future__ import annotations

from typing import Any, Sequence


def orthonormalize_basis(torch: Any, basis: Any) -> Any:
    """Return an orthonormal basis for the same column span."""

    q, _ = torch.linalg.qr(basis.to(torch.float32), mode="reduced")
    return q


def constrained_patch(
    torch: Any,
    h: Any,
    bases: Sequence[Any],
    target_coordinates: Sequence[Any],
    *,
    rtol: float = 1e-6,
) -> tuple[Any, dict[str, Any]]:
    """Set several possibly overlapping subspace coordinates at once.

    Hidden states are row vectors.  For B=[U1 ... Uk] and desired coordinates
    z*=[z1 ... zk], this computes the minimum-norm update satisfying h'B=z*:

        h' = h + (z* - hB) (B^T B)^+ B^T.
    """

    if not bases or len(bases) != len(target_coordinates):
        raise ValueError("bases and target_coordinates must be non-empty and aligned")
    batch = int(h.shape[0])
    hidden = int(h.shape[1])
    for basis, coords in zip(bases, target_coordinates):
        if int(basis.shape[0]) != hidden:
            raise ValueError(f"Basis shape {tuple(basis.shape)} incompatible with hidden={hidden}")
        if int(coords.shape[0]) != batch or int(coords.shape[1]) != int(basis.shape[1]):
            raise ValueError(
                f"Coordinate shape {tuple(coords.shape)} incompatible with batch={batch}, "
                f"rank={int(basis.shape[1])}"
            )

    combined = torch.cat([basis.to(torch.float32) for basis in bases], dim=1)
    target = torch.cat([coords.to(torch.float32) for coords in target_coordinates], dim=1)
    h32 = h.to(torch.float32)
    gram = combined.T @ combined
    gram_pinv = torch.linalg.pinv(gram, rtol=rtol)
    delta = (target - h32 @ combined) @ gram_pinv @ combined.T
    patched = h32 + delta
    residual = patched @ combined - target
    singular = torch.linalg.svdvals(gram)
    positive = singular[singular > max(float(singular.max()) * rtol, 1e-12)]
    condition_number = (
        float((positive.max() / positive.min()).detach().cpu()) if int(positive.numel()) else float("inf")
    )
    diagnostics = {
        "coordinate_residual_mean": float(residual.norm(dim=1).mean().detach().cpu()),
        "coordinate_residual_max": float(residual.norm(dim=1).max().detach().cpu()),
        "update_norm_mean": float(delta.norm(dim=1).mean().detach().cpu()),
        "update_norm_max": float(delta.norm(dim=1).max().detach().cpu()),
        "gram_condition_number": condition_number,
        "combined_rank": int(torch.linalg.matrix_rank(combined).detach().cpu()),
        "requested_rank": int(combined.shape[1]),
    }
    return patched, diagnostics


def sequential_patch(
    torch: Any,
    h: Any,
    bases: Sequence[Any],
    target_coordinates: Sequence[Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply ordinary single-subspace replacements in the supplied order."""

    if not bases or len(bases) != len(target_coordinates):
        raise ValueError("bases and target_coordinates must be non-empty and aligned")
    original = h.to(torch.float32)
    patched = original
    for basis, target in zip(bases, target_coordinates):
        u = basis.to(torch.float32)
        patched = patched - (patched @ u) @ u.T + target.to(torch.float32) @ u.T
    combined = torch.cat([basis.to(torch.float32) for basis in bases], dim=1)
    desired = torch.cat([coords.to(torch.float32) for coords in target_coordinates], dim=1)
    residual = patched @ combined - desired
    delta = patched - original
    return patched, {
        "coordinate_residual_mean": float(residual.norm(dim=1).mean().detach().cpu()),
        "coordinate_residual_max": float(residual.norm(dim=1).max().detach().cpu()),
        "update_norm_mean": float(delta.norm(dim=1).mean().detach().cpu()),
        "update_norm_max": float(delta.norm(dim=1).max().detach().cpu()),
    }


def random_orthonormal_basis(
    torch: Any,
    hidden_size: int,
    rank: int,
    *,
    device: Any,
    seed: int,
) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    raw = torch.randn(hidden_size, rank, generator=generator, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q

