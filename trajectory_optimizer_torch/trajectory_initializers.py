from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from .config import ExperimentConfig

if TYPE_CHECKING:
    from .targets import VisibleTargetSamples


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
    def build(
        self,
        config: ExperimentConfig,
        device: torch.device,
        target_samples: "VisibleTargetSamples | None" = None,
    ) -> TrajectorySpec:
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
    def build(
        self,
        config: ExperimentConfig,
        device: torch.device,
        target_samples: "VisibleTargetSamples | None" = None,
    ) -> TrajectorySpec:
        mode = config.model.trajectory_mode.strip().lower()
        if mode == "pyramid":
            return self._build_pyramid(config, device)
        if mode == "curve_band":
            if target_samples is None:
                raise ValueError("trajectory_mode='curve_band' requires target_samples to be provided")
            return self._build_curve_band(config, device, target_samples)
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

    def _build_curve_band(
        self,
        config: ExperimentConfig,
        device: torch.device,
        target_samples: "VisibleTargetSamples",
    ) -> TrajectorySpec:
        layer_cfgs = config.model.trajectory_curve_band_layers
        if not layer_cfgs:
            raise ValueError(
                "trajectory_curve_band_layers must be provided when trajectory_mode is 'curve_band'"
            )
        if not target_samples.polyline_xy:
            raise ValueError("curve_band requires at least one visible target polyline")

        xy_half_extent_m = 2.0 * float(config.grid.plane_half_size_m)
        plane_half_size_m = float(config.grid.plane_half_size_m)

        layer_centers: list[torch.Tensor] = []
        layer_specs: list[dict[str, float | int]] = []
        z_values: list[float] = []
        step_values: list[float] = []

        for layer in layer_cfgs:
            along_step = float(layer.along_step_m)
            band_half = float(layer.band_half_width_m)
            band_step = float(layer.band_step_m) if layer.band_step_m is not None else along_step
            if along_step <= 0.0:
                raise ValueError("curve_band along_step_m must be positive")
            if band_half < 0.0:
                raise ValueError("curve_band band_half_width_m must be non-negative")
            if band_step <= 0.0:
                raise ValueError("curve_band band_step_m must be positive")

            offset_count_each_side = int(math.floor(band_half / band_step + 1.0e-9))
            offset_values = torch.tensor(
                [k * band_step for k in range(-offset_count_each_side, offset_count_each_side + 1)],
                device=device,
                dtype=torch.float32,
            )

            xy_chunks: list[torch.Tensor] = []
            for polyline in target_samples.polyline_xy:
                if polyline.shape[0] < 2:
                    continue
                poly = polyline.to(device=device, dtype=torch.float32)
                seg = poly[1:] - poly[:-1]
                seg_len = torch.linalg.norm(seg, dim=-1)
                cum = torch.cat((torch.zeros(1, device=device, dtype=poly.dtype), torch.cumsum(seg_len, dim=0)))
                total_len = float(cum[-1].item())
                if total_len <= 0.0:
                    continue
                sample_count = max(int(math.floor(total_len / along_step)) + 1, 2)
                s = torch.linspace(0.0, total_len, sample_count, device=device, dtype=poly.dtype)
                # piecewise-linear interpolation of polyline at arc-length positions s
                idx = torch.searchsorted(cum, s, right=True).clamp(min=1, max=poly.shape[0] - 1)
                s0 = cum[idx - 1]
                s1 = cum[idx]
                denom = (s1 - s0).clamp_min(1.0e-12)
                t = ((s - s0) / denom).clamp(0.0, 1.0).unsqueeze(-1)
                p0 = poly[idx - 1]
                p1 = poly[idx]
                centers_xy = p0 + t * (p1 - p0)
                tangents = (p1 - p0) / denom.unsqueeze(-1).clamp_min(1.0e-12)
                tangents = tangents / torch.linalg.norm(tangents, dim=-1, keepdim=True).clamp_min(1.0e-12)
                normals = torch.stack((-tangents[:, 1], tangents[:, 0]), dim=-1)
                # broadcast band offsets along normal
                if offset_values.shape[0] > 0:
                    band = centers_xy[:, None, :] + offset_values[None, :, None] * normals[:, None, :]
                else:
                    band = centers_xy[:, None, :]
                band = band.reshape(-1, 2)
                # clip to physical work plane
                band = torch.clamp(band, min=-plane_half_size_m, max=plane_half_size_m)
                xy_chunks.append(band)

            if not xy_chunks:
                raise ValueError("curve_band produced no waypoints; check polylines and along_step_m")

            xy_layer = torch.cat(xy_chunks, dim=0)
            z_col = torch.full((xy_layer.shape[0], 1), float(layer.z_m), device=device, dtype=xy_layer.dtype)
            layer_centers.append(torch.cat((xy_layer, z_col), dim=-1))
            layer_specs.append(
                {
                    "z_m": float(layer.z_m),
                    "along_step_m": along_step,
                    "band_half_width_m": band_half,
                    "band_step_m": band_step,
                    "waypoint_count": int(xy_layer.shape[0]),
                }
            )
            z_values.append(float(layer.z_m))
            step_values.append(along_step)

        centers = torch.cat(layer_centers, dim=0)
        return TrajectorySpec(
            centers=centers,
            mode="curve_band",
            shape=None,
            step_m=None,
            step_text="curve_band along = " + ", ".join(f"{1000.0 * value:.3f} mm" for value in step_values),
            shape_text=" + ".join(
                f"{item['waypoint_count']} pts @ z={1000.0 * item['z_m']:.2f} mm "
                f"(band={1000.0 * item['band_half_width_m']:.3f} mm)"
                for item in layer_specs
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