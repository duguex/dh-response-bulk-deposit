"""Joint H + dH/dR loss algebra (Track B).

Canonical objective for one structure:

    L = L_H + lambda_d * mean_c L_G_c

where each L_G_c is a scalar loss on one (atom, axis) derivative component.
Gradients MUST be taken on the single scalar L (or equivalently each
component term scaled by 1/C before accumulating). Summing unscaled
per-component backwards implements lambda_d * sum_c, which is wrong.

Eval / train / checkpoint selection must use the same L_H and L_G.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch


def relfro(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Relative Frobenius: ||pred-target|| / ||target|| (active elements already selected)."""
    denom = torch.linalg.vector_norm(target)
    if float(denom.detach()) <= float(eps):
        # Keep graph; caller should avoid empty/zero targets.
        return torch.linalg.vector_norm(pred - target) / (denom + 1e-12)
    return torch.linalg.vector_norm(pred - target) / denom


def mean_component_loss(component_losses: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean of per-component scalar losses (requires non-empty)."""
    if not component_losses:
        raise ValueError("component_losses must be non-empty for mean_component_loss")
    return torch.stack(list(component_losses), dim=0).mean()


def joint_loss(
    loss_h: torch.Tensor,
    component_losses_g: Sequence[torch.Tensor] | None,
    lambda_d: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build joint scalar and factors.

    Returns
    -------
    loss : L_H + lambda_d * mean_c L_G_c   (or L_H if no G terms / lambda_d==0)
    loss_h : L_H
    loss_g : mean_c L_G_c  (0-scalar detached-friendly zero if unused)
    """
    if component_losses_g is None or float(lambda_d) == 0.0 or len(component_losses_g) == 0:
        zero = loss_h * 0.0
        return loss_h, loss_h, zero
    loss_g = mean_component_loss(component_losses_g)
    loss = loss_h + float(lambda_d) * loss_g
    return loss, loss_h, loss_g


def component_fd_losses(
    model: object,
    *,
    atoms: Iterable[int],
    axes: Iterable[int],
    delta: float,
    dh_dr: torch.Tensor,
    mask: torch.Tensor,
    model_component_fd,
) -> list[torch.Tensor]:
    """Per-component relfro losses using central-difference G_θ vs label rows.

    dh_dr : (C, n_block, nao^2) ordered as nested loop atoms × axes
    mask  : (n_block, nao^2) bool
    """
    axes_list = [int(a) for a in axes]
    atoms_list = [int(a) for a in atoms]
    expect = len(atoms_list) * len(axes_list)
    if int(dh_dr.shape[0]) != expect:
        raise ValueError(f"dh_dr has {dh_dr.shape[0]} rows, expected {expect}")
    losses: list[torch.Tensor] = []
    k = 0
    for atom in atoms_list:
        for axis in axes_list:
            d_pred = model_component_fd(model, atom, axis, delta)
            d_tgt = dh_dr[k]
            losses.append(relfro(d_pred[mask], d_tgt[mask]))
            k += 1
    return losses


def grad_norm_l2(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5

def backward_joint(
    loss_h: torch.Tensor,
    component_losses_g: Sequence[torch.Tensor] | None,
    lambda_d: float,
) -> tuple[float, float]:
    """Backward for L_H + λ mean_c L_G_c without retaining all graphs.

    Equivalent to ``joint_loss(...).backward()`` but runs
    ``(λ/C) * L_G_c.backward()`` per component so peak memory stays O(1)
    FD pair instead of O(C).
    """
    loss_h.backward()
    h_err = float(loss_h.detach())
    if component_losses_g is None or float(lambda_d) == 0.0 or len(component_losses_g) == 0:
        return h_err, float("nan")
    c_count = len(component_losses_g)
    scale = float(lambda_d) / float(c_count)
    g_vals: list[float] = []
    for loss_g_c in component_losses_g:
        g_vals.append(float(loss_g_c.detach()))
        (scale * loss_g_c).backward()
    return h_err, float(sum(g_vals) / c_count)
