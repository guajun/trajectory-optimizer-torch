from __future__ import annotations

import torch


def full_image_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predicted - target).pow(2))


def all_finite_tensors(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all() for parameter in module.parameters())


def model_regularization_terms(model: torch.nn.Module, loss_weights: dict[str, float]) -> dict[str, torch.Tensor]:
    centers = model.centers()
    moments = model.moment_vec()

    smoothness_weight = loss_weights.get("smoothness", 0.0)
    curvature_weight = loss_weights.get("curvature", 0.0)
    moment_l2_weight = loss_weights.get("moment_l2", 0.0)
    z_l2_weight = loss_weights.get("z_l2", 0.0)
    total_dwell_time_weight = loss_weights.get("total_dwell_time", 0.0)

    if centers.requires_grad:
        smoothness = (
            ((centers[1:] - centers[:-1]).pow(2).sum(dim=-1)).mean() if centers.shape[0] > 1 else centers.new_tensor(0.0)
        )
        curvature = (
            (centers[2:] - 2.0 * centers[1:-1] + centers[:-2]).pow(2).sum(dim=-1).mean()
            if centers.shape[0] > 2
            else centers.new_tensor(0.0)
        )
        z_l2 = centers[:, 2].pow(2).mean()
    else:
        smoothness = centers.new_tensor(0.0)
        curvature = centers.new_tensor(0.0)
        z_l2 = centers.new_tensor(0.0)

    moment_l2 = moments.pow(2).mean()
    total_dwell_time = model.total_dwell_time_s()

    smoothness_penalty = smoothness_weight * smoothness
    curvature_penalty = curvature_weight * curvature
    moment_l2_penalty = moment_l2_weight * moment_l2
    z_l2_penalty = z_l2_weight * z_l2
    total_dwell_time_penalty = total_dwell_time_weight * total_dwell_time

    return {
        "smoothness": smoothness,
        "curvature": curvature,
        "moment_l2": moment_l2,
        "z_l2": z_l2,
        "total_dwell_time": total_dwell_time,
        "smoothness_penalty": smoothness_penalty,
        "curvature_penalty": curvature_penalty,
        "moment_l2_penalty": moment_l2_penalty,
        "z_l2_penalty": z_l2_penalty,
        "total_dwell_time_penalty": total_dwell_time_penalty,
        "total": smoothness_penalty + curvature_penalty + moment_l2_penalty + z_l2_penalty + total_dwell_time_penalty,
    }