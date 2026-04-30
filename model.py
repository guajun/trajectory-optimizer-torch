from __future__ import annotations

import math

import torch

from .config import ExperimentConfig
from .targets import VisibleTargetSamples
from .trajectory_initializers import TrajectorySpec
from .utils import EPS, PHYSICAL_EPS, inverse_sigmoid


def dipole_field_tensor(points: torch.Tensor, centers: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
    relative = points[:, None, :] - centers[None, :, :]
    r2 = (relative * relative).sum(dim=-1).clamp_min(EPS)
    r = torch.sqrt(r2)
    inv_r3 = 1.0 / (r2 * r)
    inv_r5 = inv_r3 / r2
    mdotr = (relative * moments[None, :, :]).sum(dim=-1)
    return 3.0 * relative * mdotr[:, :, None] * inv_r5[:, :, None] - moments[None, :, :] * inv_r3[:, :, None]


def safe_normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(EPS)


def rotate_vectors(vectors: torch.Tensor, axes: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    axis_norm = torch.linalg.norm(axes, dim=-1, keepdim=True)
    axis_unit = axes / axis_norm.clamp_min(EPS)
    cos_angle = torch.cos(angles)[:, None]
    sin_angle = torch.sin(angles)[:, None]
    rotated = (
        vectors * cos_angle
        + torch.cross(axis_unit, vectors, dim=-1) * sin_angle
        + axis_unit * (axis_unit * vectors).sum(dim=-1, keepdim=True) * (1.0 - cos_angle)
    )
    return torch.where(axis_norm > 1.0e-6, rotated, vectors)


def exact_align_vectors(
    orientation: torch.Tensor,
    field_dir: torch.Tensor,
    field_mag: torch.Tensor,
    particle_moment: float,
    rotational_drag: float,
    dwell_time: torch.Tensor,
    static_friction_torque: float,
):
    cross_term = torch.cross(orientation, field_dir, dim=-1)
    sin_theta = torch.linalg.norm(cross_term, dim=-1).clamp(0.0, 1.0)
    cos_theta = (orientation * field_dir).sum(dim=-1).clamp(-1.0, 1.0)
    torque_magnitude = particle_moment * field_mag * sin_theta
    active_mask = torque_magnitude > static_friction_torque

    angular_rate = particle_moment * field_mag / max(rotational_drag, PHYSICAL_EPS)
    tan_half_theta = sin_theta / (1.0 + cos_theta).clamp_min(EPS)
    theta = torch.atan2(sin_theta, cos_theta)
    theta_new = 2.0 * torch.atan(tan_half_theta * torch.exp(-angular_rate * dwell_time))
    step_rotation = torch.where(active_mask, torch.clamp(theta - theta_new, 0.0, math.pi), torch.zeros_like(theta))
    updated_orientation = rotate_vectors(orientation, cross_term, step_rotation)
    return updated_orientation, step_rotation, active_mask, torque_magnitude


def dipole_effective_radius(moment_norm: torch.Tensor, friction_field_threshold: float) -> torch.Tensor | float:
    if friction_field_threshold <= 0.0:
        return float("inf")
    return torch.pow((2.0 * moment_norm.clamp_min(0.0)) / friction_field_threshold, 1.0 / 3.0)


class SequentialOmniMagnetTrajectoryField(torch.nn.Module):
    def __init__(
        self,
        config: ExperimentConfig,
        target_samples: VisibleTargetSamples,
        trajectory_spec: TrajectorySpec,
        device: torch.device,
    ):
        super().__init__()
        model_cfg = config.model
        grid_cfg = config.grid
        target_xy = target_samples.xy
        tangent_xy = target_samples.tangent_xy
        waypoint_count = trajectory_spec.centers.shape[0]

        sample_indices = torch.linspace(0, target_xy.shape[0] - 1, waypoint_count, device=device).round().long()
        init_tangent_xy = tangent_xy[sample_indices]

        self.max_moment_strength = float(model_cfg.max_moment_strength)
        init_strength = float(model_cfg.init_moment_strength)
        min_active_init_strength = (
            float(model_cfg.init_activation_margin)
            * (float(model_cfg.static_friction_torque) / max(float(model_cfg.particle_moment_magnitude), PHYSICAL_EPS))
            * (float(trajectory_spec.z_min_m) ** 3)
        )
        if init_strength < min_active_init_strength:
            init_strength = min(min_active_init_strength, self.max_moment_strength)
            if init_strength < min_active_init_strength:
                print(
                    "warning_init_moment_strength_dead_zone "
                    f"required_min={min_active_init_strength:.3e} max_moment_strength={self.max_moment_strength:.3e}"
                )

        init_moment = torch.zeros((waypoint_count, 3), dtype=torch.float32, device=device)
        init_moment[:, :2] = init_strength * init_tangent_xy
        init_moment[:, 2] = -0.20 * init_strength
        self.raw_moment = torch.nn.Parameter(torch.atanh(torch.clamp(init_moment / self.max_moment_strength, -0.999, 0.999)))

        self.optimize_dwell_time = bool(model_cfg.optimize_dwell_time)
        self.min_dwell_time_s = float(model_cfg.min_dwell_time_s)
        self.max_dwell_time_s = float(model_cfg.max_dwell_time_s)
        self.init_dwell_time_s = float(model_cfg.init_dwell_time_s)
        if self.max_dwell_time_s < self.min_dwell_time_s:
            raise ValueError("max_dwell_time_s must be greater than or equal to min_dwell_time_s")
        dwell_span = max(self.max_dwell_time_s - self.min_dwell_time_s, 1.0e-9)
        init_dwell_ratio = (self.init_dwell_time_s - self.min_dwell_time_s) / dwell_span
        if self.optimize_dwell_time:
            self.raw_dwell_time = torch.nn.Parameter(
                torch.full((waypoint_count,), inverse_sigmoid(init_dwell_ratio), dtype=torch.float32, device=device)
            )
        else:
            self.register_buffer(
                "fixed_dwell_time_s",
                torch.full((waypoint_count,), self.init_dwell_time_s, dtype=torch.float32, device=device),
                persistent=False,
            )

        self.freeze_centers = bool(model_cfg.freeze_centers)
        self.register_buffer("base_centers", trajectory_spec.centers.clone(), persistent=False)
        if self.freeze_centers:
            self.center_offsets = None
        else:
            self.center_offsets = torch.nn.Parameter(torch.zeros_like(self.base_centers))

        self.register_buffer("waypoint_index", torch.arange(waypoint_count, dtype=torch.long, device=device), persistent=False)
        self.register_buffer("initial_orientation", torch.tensor((0.0, 0.0, 1.0), dtype=torch.float32, device=device), persistent=False)

        self.plane_half_size_m = grid_cfg.plane_half_size_m
        self.trajectory_mode = trajectory_spec.mode
        self.trajectory_step_m = trajectory_spec.step_m
        self.trajectory_step_text = trajectory_spec.step_text
        self.trajectory_grid_shape = trajectory_spec.shape
        self.trajectory_grid_shape_text = trajectory_spec.shape_text
        self.trajectory_xy_half_extent_m = trajectory_spec.xy_half_extent_m
        self.trajectory_z_min_m = trajectory_spec.z_min_m
        self.trajectory_z_max_m = trajectory_spec.z_max_m
        self.trajectory_layer_specs = trajectory_spec.layer_specs
        self.particle_moment_magnitude = float(model_cfg.particle_moment_magnitude)
        self.rotational_drag_coefficient = float(model_cfg.rotational_drag_coefficient)
        self.static_friction_torque = float(model_cfg.static_friction_torque)
        self.brightness_gamma = float(model_cfg.brightness_gamma)
        self.recency_scale = float(model_cfg.recency_scale)
        self.relaxation_keep = math.exp(-max(self.recency_scale, 0.0))
        self.point_chunk_size = int(model_cfg.point_chunk_size)
        self.waypoint_block_size = int(model_cfg.waypoint_block_size)

    def centers(self) -> torch.Tensor:
        if self.center_offsets is None:
            return self.base_centers
        return self.base_centers + self.center_offsets

    def moment_vec(self) -> torch.Tensor:
        return self.max_moment_strength * torch.tanh(self.raw_moment)

    def dwell_time_s(self) -> torch.Tensor:
        if self.optimize_dwell_time:
            return self.min_dwell_time_s + (self.max_dwell_time_s - self.min_dwell_time_s) * torch.sigmoid(self.raw_dwell_time)
        return self.fixed_dwell_time_s

    def total_dwell_time_s(self) -> torch.Tensor:
        return self.dwell_time_s().sum()

    def alignment_rate_per_field(self) -> float:
        return self.particle_moment_magnitude / max(self.rotational_drag_coefficient, PHYSICAL_EPS)

    def friction_field_threshold(self) -> float:
        return self.static_friction_torque / max(self.particle_moment_magnitude, PHYSICAL_EPS)

    def waypoint_candidate_mask(self, points_xy: torch.Tensor, centers: torch.Tensor, moments: torch.Tensor) -> torch.Tensor:
        friction_threshold = self.friction_field_threshold()
        moment_norm = torch.full(
            (moments.shape[0],),
            math.sqrt(3.0) * self.max_moment_strength,
            dtype=moments.dtype,
            device=moments.device,
        )
        effective_radius = dipole_effective_radius(moment_norm, friction_threshold)
        z2 = centers[:, 2].pow(2)
        horizontal_radius2 = (effective_radius.pow(2) - z2).clamp_min(0.0)
        point_xy2 = (points_xy * points_xy).sum(dim=-1, keepdim=True)
        center_xy2 = (centers[:, :2] * centers[:, :2]).sum(dim=-1)[None, :]
        point_center_dot = points_xy @ centers[:, :2].T
        distance_xy2 = point_xy2 + center_xy2 - 2.0 * point_center_dot
        return distance_xy2 <= horizontal_radius2[None, :]

    def _simulate_chunk(self, points: torch.Tensor, centers: torch.Tensor, moments: torch.Tensor, collect_step_stats: bool = False):
        point_count = points.shape[0]
        waypoint_count = centers.shape[0]
        dwell_times = self.dwell_time_s()
        candidate_mask = self.waypoint_candidate_mask(points[:, :2], centers, moments)
        orientation = self.initial_orientation[None, :].expand(point_count, -1).clone()
        active_dwell_time = points.new_zeros(point_count)
        peak_bxy = points.new_zeros(point_count)

        step_stats = None
        if collect_step_stats:
            stats_dtype = torch.float64
            step_stats = {
                "bxy_sum": torch.zeros(waypoint_count, dtype=stats_dtype, device=points.device),
                "max_bxy": torch.zeros(waypoint_count, dtype=stats_dtype, device=points.device),
                "active_time_sum": torch.zeros(waypoint_count, dtype=stats_dtype, device=points.device),
                "rotation_sum": torch.zeros(waypoint_count, dtype=stats_dtype, device=points.device),
                "brightness_gain_sum": torch.zeros(waypoint_count, dtype=stats_dtype, device=points.device),
            }

        block_size = max(self.waypoint_block_size, 1)
        for block_start in range(0, waypoint_count, block_size):
            block_end = min(block_start + block_size, waypoint_count)
            block_slice = slice(block_start, block_end)
            block_centers = centers[block_slice]
            block_moments = moments[block_slice]
            block_dwell_times = dwell_times[block_slice]
            block_candidate_mask = candidate_mask[:, block_slice]
            union_mask = block_candidate_mask.any(dim=1)
            if not bool(union_mask.any().item()):
                continue

            block_points = points[union_mask]
            block_orientation = orientation[union_mask]
            block_active_dwell_time = active_dwell_time[union_mask]
            block_peak_bxy = peak_bxy[union_mask]
            block_fields = dipole_field_tensor(block_points, block_centers, block_moments)
            block_candidate_mask = block_candidate_mask[union_mask]

            for local_waypoint_id in range(block_end - block_start):
                waypoint_id = block_start + local_waypoint_id
                local_candidate_mask = block_candidate_mask[:, local_waypoint_id]
                if not bool(local_candidate_mask.any().item()):
                    continue

                field_step = block_fields[local_candidate_mask, local_waypoint_id, :]
                bxy = torch.linalg.norm(field_step[:, :2], dim=-1).clamp_min(EPS)
                bmag = torch.linalg.norm(field_step, dim=-1).clamp_min(EPS)
                field_dir = field_step / bmag[:, None]
                dwell_time = block_dwell_times[local_waypoint_id]

                if collect_step_stats:
                    prev_inplane = torch.linalg.norm(block_orientation[local_candidate_mask, :2], dim=-1)

                updated_orientation, step_rotation, active_mask, _ = exact_align_vectors(
                    block_orientation[local_candidate_mask],
                    field_dir,
                    bmag,
                    self.particle_moment_magnitude,
                    self.rotational_drag_coefficient,
                    dwell_time,
                    self.static_friction_torque,
                )
                if self.relaxation_keep < 1.0:
                    updated_orientation = safe_normalize(
                        self.relaxation_keep * updated_orientation
                        + (1.0 - self.relaxation_keep) * self.initial_orientation[None, :]
                    )

                step_active_time = dwell_time * active_mask.to(dtype=points.dtype)
                next_orientation = block_orientation.clone()
                next_orientation[local_candidate_mask] = updated_orientation
                block_orientation = next_orientation

                next_active_dwell = block_active_dwell_time.clone()
                next_active_dwell[local_candidate_mask] = next_active_dwell[local_candidate_mask] + step_active_time
                block_active_dwell_time = next_active_dwell

                next_peak_bxy = block_peak_bxy.clone()
                next_peak_bxy[local_candidate_mask] = torch.maximum(next_peak_bxy[local_candidate_mask], bxy)
                block_peak_bxy = next_peak_bxy

                if collect_step_stats:
                    current_inplane = torch.linalg.norm(updated_orientation[:, :2], dim=-1)
                    step_stats["bxy_sum"][waypoint_id] = bxy.sum().to(torch.float64)
                    step_stats["max_bxy"][waypoint_id] = bxy.max().to(torch.float64)
                    step_stats["active_time_sum"][waypoint_id] = step_active_time.sum().to(torch.float64)
                    step_stats["rotation_sum"][waypoint_id] = step_rotation.sum().to(torch.float64)
                    step_stats["brightness_gain_sum"][waypoint_id] = torch.clamp(current_inplane - prev_inplane, min=0.0).sum().to(torch.float64)

            updated_orientation = orientation.clone()
            updated_orientation[union_mask] = block_orientation
            orientation = updated_orientation

            updated_active_dwell_time = active_dwell_time.clone()
            updated_active_dwell_time[union_mask] = block_active_dwell_time
            active_dwell_time = updated_active_dwell_time

            updated_peak_bxy = peak_bxy.clone()
            updated_peak_bxy[union_mask] = block_peak_bxy
            peak_bxy = updated_peak_bxy

        inplane_alignment = torch.linalg.norm(orientation[:, :2], dim=-1).clamp(0.0, 1.0)
        total_dwell_time = dwell_times.sum().clamp_min(EPS)
        activation = torch.clamp(active_dwell_time / total_dwell_time, 0.0, 1.0)
        brightness = inplane_alignment.clamp_min(EPS).pow(self.brightness_gamma)
        result = {
            "brightness": brightness,
            "theta": torch.atan2(orientation[:, 1], orientation[:, 0]),
            "activation": activation,
            "nz": inplane_alignment,
            "total_bxy": peak_bxy,
            "active_time": active_dwell_time,
            "dwell_time_s": dwell_times,
            "total_dwell_time_s": total_dwell_time,
        }
        if step_stats is not None:
            result["step_stats"] = step_stats
        return result

    def forward(self, points: torch.Tensor):
        centers = self.centers()
        moments = self.moment_vec()
        if points.shape[0] > self.point_chunk_size:
            b_list, t_list, a_list, n_list, p_list, active_list = [], [], [], [], [], []
            for chunk in points.split(self.point_chunk_size):
                result = self._simulate_chunk(chunk, centers, moments)
                b_list.append(result["brightness"])
                t_list.append(result["theta"])
                a_list.append(result["activation"])
                n_list.append(result["nz"])
                p_list.append(result["total_bxy"])
                active_list.append(result["active_time"])
            return {
                "brightness": torch.cat(b_list),
                "theta": torch.cat(t_list),
                "activation": torch.cat(a_list),
                "nz": torch.cat(n_list),
                "total_bxy": torch.cat(p_list),
                "active_time": torch.cat(active_list),
                "centers": centers,
                "moment_vec": moments,
                "dwell_time_s": self.dwell_time_s(),
                "total_dwell_time_s": self.total_dwell_time_s(),
                "waypoint_index": self.waypoint_index,
            }

        result = self._simulate_chunk(points, centers, moments)
        result["centers"] = centers
        result["moment_vec"] = moments
        result["waypoint_index"] = self.waypoint_index
        return result