from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch

from .config import ExperimentConfig


@dataclass(slots=True)
class TrajectorySpec:
    centers: torch.Tensor
    mode: str
    shape: tuple[int, int, int] | None
    step_m: float | None
    step_text: str
    shape_text: str
    xy_half_extent_m: float
    z_min_m: float
    z_max_m: float
    layer_specs: list[dict[str, float | int]]


class TrajectoryInitializer(Protocol):
    def build(self, config: ExperimentConfig, device: torch.device) -> TrajectorySpec:
        ...


def build_uniform_axis(start: float, end: float, step: float, device: torch.device) -> torch.Tensor:
    if step <= 0.0:
        raise ValueError("trajectory_step_m must be positive")
    span = end - start
    if span < 0.0:
        raise ValueError("trajectory axis max must be greater than or equal to min")
    step_count = int(round(span / step))
    if not math.isclose(start + step_count * step, end, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("trajectory axis range must be evenly divisible by trajectory_step_m")
    return torch.linspace(start, end, step_count + 1, device=device)


def build_xy_plane_centers(xs: torch.Tensor, ys: torch.Tensor, z_value: float, device: torch.device) -> torch.Tensor:
    x_coords = xs.repeat(ys.shape[0])
    y_coords = ys.repeat_interleave(xs.shape[0])
    z_coords = torch.full((x_coords.shape[0],), float(z_value), device=device, dtype=xs.dtype)
    return torch.stack((x_coords, y_coords, z_coords), dim=-1)


class FixedGridTrajectoryInitializer:
    def build(self, config: ExperimentConfig, device: torch.device) -> TrajectorySpec:
        if config.model.trajectory_mode.strip().lower() == "pyramid":
            return self._build_pyramid(config, device)
        return self._build_uniform(config, device)

    def _build_uniform(self, config: ExperimentConfig, device: torch.device) -> TrajectorySpec:
        step_m = float(config.model.trajectory_step_m)
        xy_half_extent_m = 2.0 * float(config.grid.plane_half_size_m)
        z_min_m = float(config.model.trajectory_z_min_m)
        z_max_m = float(config.model.trajectory_z_max_m)

        xs = build_uniform_axis(-xy_half_extent_m, xy_half_extent_m, step_m, device)
        ys = build_uniform_axis(-xy_half_extent_m, xy_half_extent_m, step_m, device)
        zs = build_uniform_axis(z_min_m, z_max_m, step_m, device)
        layer_centers = [build_xy_plane_centers(xs, ys, z_value.item(), device) for z_value in zs]
        centers = torch.cat(layer_centers, dim=0)
        layer_specs = [
            {
                "z_m": float(z_value.item()),
                "xy_step_m": step_m,
                "x_count": int(xs.shape[0]),
                "y_count": int(ys.shape[0]),
            }
            for z_value in zs
        ]
        return TrajectorySpec(
            centers=centers,
            mode="uniform",
            shape=(int(xs.shape[0]), int(ys.shape[0]), int(zs.shape[0])),
            step_m=step_m,
            step_text=f"uniform xyz = {1000.0 * step_m:.3f} mm",
            shape_text=f"{int(xs.shape[0])} x {int(ys.shape[0])} x {int(zs.shape[0])}",
            xy_half_extent_m=xy_half_extent_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            layer_specs=layer_specs,
        )

    def _build_pyramid(self, config: ExperimentConfig, device: torch.device) -> TrajectorySpec:
        xy_half_extent_m = 2.0 * float(config.grid.plane_half_size_m)
        layer_cfgs = config.model.trajectory_pyramid_layers
        if not layer_cfgs:
            raise ValueError("trajectory_pyramid_layers must be provided when trajectory_mode is 'pyramid'")

        centers_by_layer = []
        layer_specs = []
        z_values = []
        step_values = []

        sorted_layers = sorted(layer_cfgs, key=lambda item: float(item.z_m), reverse=True)
        for layer in sorted_layers:
            xs = build_uniform_axis(-xy_half_extent_m, xy_half_extent_m, float(layer.xy_step_m), device)
            ys = build_uniform_axis(-xy_half_extent_m, xy_half_extent_m, float(layer.xy_step_m), device)
            centers_by_layer.append(build_xy_plane_centers(xs, ys, float(layer.z_m), device))
            layer_specs.append(
                {
                    "z_m": float(layer.z_m),
                    "xy_step_m": float(layer.xy_step_m),
                    "x_count": int(xs.shape[0]),
                    "y_count": int(ys.shape[0]),
                }
            )
            z_values.append(float(layer.z_m))
            step_values.append(float(layer.xy_step_m))

        return TrajectorySpec(
            centers=torch.cat(centers_by_layer, dim=0),
            mode="pyramid",
            shape=None,
            step_m=None,
            step_text="pyramid xy = " + ", ".join(f"{1000.0 * value:.3f} mm" for value in step_values),
            shape_text=" + ".join(
                f"{item['x_count']} x {item['y_count']} @ z={1000.0 * item['z_m']:.2f} mm" for item in layer_specs
            ),
            xy_half_extent_m=xy_half_extent_m,
            z_min_m=min(z_values),
            z_max_m=max(z_values),
            layer_specs=layer_specs,
        )


def build_trajectory_initializer(config: ExperimentConfig) -> TrajectoryInitializer:
    initializer_name = config.model.trajectory_initializer.strip().lower()
    if initializer_name != "fixed_grid":
        raise ValueError(f"Unsupported trajectory_initializer: {config.model.trajectory_initializer}")
    return FixedGridTrajectoryInitializer()