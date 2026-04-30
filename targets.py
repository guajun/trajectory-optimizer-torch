from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import GridConfig, TargetConfig
from .utils import ALLOWED_FUNCTIONS, EPS, compile_expression


@dataclass(slots=True)
class GridData:
    size: int
    points: torch.Tensor
    pixels: torch.Tensor


@dataclass(slots=True)
class VisibleTargetSamples:
    xy: torch.Tensor
    tangent_xy: torch.Tensor
    polyline_xy: list[torch.Tensor]


def build_grid(size: int, plane_half_size_m: float, device: torch.device) -> GridData:
    axis = torch.linspace(-plane_half_size_m, plane_half_size_m, size, device=device)
    yy, xx = torch.meshgrid(axis.flip(0), axis, indexing="ij")
    zz = torch.zeros_like(xx)
    points = torch.stack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)), dim=-1)
    pixel_x = torch.linspace(0.0, size - 1.0, size, device=device)
    pixel_y = torch.linspace(0.0, size - 1.0, size, device=device)
    py, px = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    pixels = torch.stack((px.reshape(-1), py.reshape(-1)), dim=-1)
    return GridData(size=size, points=points, pixels=pixels)


class AnalyticGraphTarget:
    def __init__(self, grid_cfg: GridConfig, target_cfg: TargetConfig):
        self.plane_half_size_m = grid_cfg.plane_half_size_m
        self.expression_text = target_cfg.expression
        self.valid_expression_text = target_cfg.valid_expression
        self.parameters = {name: float(value) for name, value in target_cfg.parameters.items()}
        self.x_min_m = float(target_cfg.x_min_m if target_cfg.x_min_m is not None else -self.plane_half_size_m)
        self.x_max_m = float(target_cfg.x_max_m if target_cfg.x_max_m is not None else self.plane_half_size_m)
        self.y_min_m = float(target_cfg.y_min_m if target_cfg.y_min_m is not None else -self.plane_half_size_m)
        self.y_max_m = float(target_cfg.y_max_m if target_cfg.y_max_m is not None else self.plane_half_size_m)
        self.init_sample_count = int(target_cfg.init_sample_count)
        self.plot_sample_count = int(target_cfg.plot_sample_count)
        self.newton_steps = int(target_cfg.projection.newton_steps)
        self.batch_size = int(target_cfg.projection.batch_size)
        self.global_seed_count = int(target_cfg.projection.global_seed_count)
        self.local_initial_offsets_m = [float(value) for value in target_cfg.projection.local_initial_offsets_m]
        self.interval_scan_count = int(target_cfg.projection.interval_scan_count)

        allowed_names = set(ALLOWED_FUNCTIONS.keys()) | {"x", "pi", "plane_half_size_m"} | set(self.parameters.keys())
        self.expression_code = compile_expression(self.expression_text, allowed_names)
        self.valid_expression_code = (
            compile_expression(self.valid_expression_text, allowed_names) if self.valid_expression_text else None
        )
        self.branch_intervals = self._build_branch_intervals()

    def _build_branch_intervals(self):
        x = torch.linspace(self.x_min_m, self.x_max_m, self.interval_scan_count, dtype=torch.float64)
        _, valid = self.evaluate(x)
        valid_list = valid.detach().cpu().tolist()
        x_list = x.detach().cpu().tolist()

        intervals = []
        start = None
        for index, flag in enumerate(valid_list):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                if index - start >= 2:
                    intervals.append((x_list[start], x_list[index - 1]))
                start = None
        if start is not None and len(valid_list) - start >= 2:
            intervals.append((x_list[start], x_list[-1]))

        if not intervals:
            raise ValueError("No visible function branches remain after applying the validity mask")
        return intervals

    def scope(self, x: torch.Tensor) -> dict[str, object]:
        scope = dict(ALLOWED_FUNCTIONS)
        scope.update(self.parameters)
        scope["x"] = x
        scope["pi"] = math.pi
        scope["plane_half_size_m"] = self.plane_half_size_m
        return scope

    def _eval_code(self, code, x: torch.Tensor) -> torch.Tensor:
        value = eval(code, {"__builtins__": {}}, self.scope(x))
        if torch.is_tensor(value):
            return value
        return torch.full_like(x, float(value))

    def evaluate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self._eval_code(self.expression_code, x)
        valid = torch.isfinite(y)
        valid &= x >= self.x_min_m
        valid &= x <= self.x_max_m
        valid &= y >= self.y_min_m
        valid &= y <= self.y_max_m
        if self.valid_expression_code is not None:
            mask = eval(self.valid_expression_code, {"__builtins__": {}}, self.scope(x))
            if torch.is_tensor(mask):
                mask = mask.to(dtype=torch.bool)
            else:
                mask = torch.full(x.shape, bool(mask), device=x.device, dtype=torch.bool)
            valid &= mask
        return y, valid

    def evaluate_with_derivatives(self, x: torch.Tensor):
        x_work = x.clone().detach().requires_grad_(True)
        y = self._eval_code(self.expression_code, x_work)
        y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        dy = torch.autograd.grad(y_safe.sum(), x_work, create_graph=True)[0]
        dy_safe = torch.where(torch.isfinite(dy), dy, torch.zeros_like(dy))
        d2y = torch.autograd.grad(dy_safe.sum(), x_work, create_graph=False)[0]
        _, valid = self.evaluate(x_work.detach())
        valid &= torch.isfinite(dy.detach())
        valid &= torch.isfinite(d2y.detach())
        return x_work.detach(), y.detach(), dy.detach(), d2y.detach(), valid.detach()

    def interval_tensors(self, device: torch.device, dtype: torch.dtype):
        starts = torch.tensor([item[0] for item in self.branch_intervals], device=device, dtype=dtype)
        ends = torch.tensor([item[1] for item in self.branch_intervals], device=device, dtype=dtype)
        return starts, ends

    def seed_candidates(self, point_x: torch.Tensor):
        interval_starts, interval_ends = self.interval_tensors(point_x.device, point_x.dtype)
        interval_count = interval_starts.shape[0]

        candidate_parts = []
        lower_parts = []
        upper_parts = []

        def append_candidates(values, lowers, uppers):
            candidate_parts.append(values)
            lower_parts.append(lowers)
            upper_parts.append(uppers)

        if self.local_initial_offsets_m:
            for offset in self.local_initial_offsets_m:
                seed = point_x[:, None] + offset
                append_candidates(
                    torch.maximum(torch.minimum(seed, interval_ends[None, :]), interval_starts[None, :]),
                    interval_starts[None, :].expand(point_x.shape[0], interval_count),
                    interval_ends[None, :].expand(point_x.shape[0], interval_count),
                )

        branch_mid = 0.5 * (interval_starts + interval_ends)
        append_candidates(
            branch_mid[None, :].expand(point_x.shape[0], interval_count),
            interval_starts[None, :].expand(point_x.shape[0], interval_count),
            interval_ends[None, :].expand(point_x.shape[0], interval_count),
        )
        append_candidates(
            interval_starts[None, :].expand(point_x.shape[0], interval_count),
            interval_starts[None, :].expand(point_x.shape[0], interval_count),
            interval_ends[None, :].expand(point_x.shape[0], interval_count),
        )
        append_candidates(
            interval_ends[None, :].expand(point_x.shape[0], interval_count),
            interval_starts[None, :].expand(point_x.shape[0], interval_count),
            interval_ends[None, :].expand(point_x.shape[0], interval_count),
        )

        if self.global_seed_count > 0:
            global_seeds = torch.linspace(
                self.x_min_m,
                self.x_max_m,
                self.global_seed_count,
                dtype=point_x.dtype,
                device=point_x.device,
            )
            global_seeds = global_seeds[None, :, None].expand(point_x.shape[0], self.global_seed_count, interval_count)
            append_candidates(
                torch.maximum(torch.minimum(global_seeds, interval_ends[None, None, :]), interval_starts[None, None, :]).reshape(point_x.shape[0], -1),
                interval_starts[None, None, :].expand(point_x.shape[0], self.global_seed_count, interval_count).reshape(point_x.shape[0], -1),
                interval_ends[None, None, :].expand(point_x.shape[0], self.global_seed_count, interval_count).reshape(point_x.shape[0], -1),
            )

        if not candidate_parts:
            append_candidates(
                torch.maximum(torch.minimum(point_x[:, None], interval_ends[None, :]), interval_starts[None, :]),
                interval_starts[None, :].expand(point_x.shape[0], interval_count),
                interval_ends[None, :].expand(point_x.shape[0], interval_count),
            )

        return torch.cat(candidate_parts, dim=1), torch.cat(lower_parts, dim=1), torch.cat(upper_parts, dim=1)

    def min_distance2(self, point_xy: torch.Tensor) -> torch.Tensor:
        point_x = point_xy[:, 0]
        point_y = point_xy[:, 1]
        candidates, lower_bounds, upper_bounds = self.seed_candidates(point_x)

        for _ in range(self.newton_steps):
            flat_x = candidates.reshape(-1)
            x_eval, y_eval, dy_eval, d2y_eval, valid = self.evaluate_with_derivatives(flat_x)
            x_eval = x_eval.reshape_as(candidates)
            y_eval = y_eval.reshape_as(candidates)
            dy_eval = dy_eval.reshape_as(candidates)
            d2y_eval = d2y_eval.reshape_as(candidates)
            valid = valid.reshape_as(candidates)

            g = (x_eval - point_x[:, None]) + (y_eval - point_y[:, None]) * dy_eval
            gp = 1.0 + dy_eval * dy_eval + (y_eval - point_y[:, None]) * d2y_eval
            step_ok = valid & torch.isfinite(g) & torch.isfinite(gp) & (gp.abs() > 1.0e-6)
            step = torch.where(step_ok, g / gp, torch.zeros_like(g))
            next_x = (x_eval - step).detach()
            candidates = torch.maximum(torch.minimum(next_x, upper_bounds), lower_bounds)

        final_y, final_valid = self.evaluate(candidates.reshape(-1))
        final_y = final_y.reshape_as(candidates)
        final_valid = final_valid.reshape_as(candidates)
        dist2 = (candidates - point_x[:, None]).pow(2) + (final_y - point_y[:, None]).pow(2)
        dist2 = torch.where(final_valid, dist2, torch.full_like(dist2, float("inf")))
        return dist2.min(dim=1).values

    def build_visible_samples(self, sample_count: int, device: torch.device) -> VisibleTargetSamples:
        x = torch.linspace(self.x_min_m, self.x_max_m, sample_count, device=device)
        x_work = x.clone().detach().requires_grad_(True)
        y = self._eval_code(self.expression_code, x_work)
        y_safe = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        dy = torch.autograd.grad(y_safe.sum(), x_work, create_graph=False)[0]
        _, valid = self.evaluate(x_work.detach())
        valid &= torch.isfinite(dy.detach())

        x_cpu = x_work.detach().cpu()
        y_cpu = y.detach().cpu()
        dy_cpu = dy.detach().cpu()
        valid_cpu = valid.detach().cpu().tolist()

        polylines = []
        xy_chunks = []
        tangent_chunks = []
        start = None
        for index, flag in enumerate(valid_cpu):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                if index - start >= 2:
                    x_seg = x_cpu[start:index].to(device)
                    y_seg = y_cpu[start:index].to(device)
                    dy_seg = dy_cpu[start:index].to(device)
                    xy = torch.stack((x_seg, y_seg), dim=-1)
                    tangent = torch.stack((torch.ones_like(dy_seg), dy_seg), dim=-1)
                    tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(EPS)
                    polylines.append(xy)
                    xy_chunks.append(xy)
                    tangent_chunks.append(tangent)
                start = None
        if start is not None and len(valid_cpu) - start >= 2:
            x_seg = x_cpu[start:].to(device)
            y_seg = y_cpu[start:].to(device)
            dy_seg = dy_cpu[start:].to(device)
            xy = torch.stack((x_seg, y_seg), dim=-1)
            tangent = torch.stack((torch.ones_like(dy_seg), dy_seg), dim=-1)
            tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(EPS)
            polylines.append(xy)
            xy_chunks.append(xy)
            tangent_chunks.append(tangent)

        if not xy_chunks:
            raise ValueError("No visible function branches remain after applying the validity mask")

        return VisibleTargetSamples(
            xy=torch.cat(xy_chunks, dim=0),
            tangent_xy=torch.cat(tangent_chunks, dim=0),
            polyline_xy=polylines,
        )


def target_ridge_map(points: torch.Tensor, target: AnalyticGraphTarget, sigma: float, intensity_gain: float) -> torch.Tensor:
    chunks = []
    for point_chunk in points.split(target.batch_size, dim=0):
        dist2 = target.min_distance2(point_chunk[:, :2])
        chunks.append(torch.clamp(intensity_gain * torch.exp(-0.5 * dist2 / (sigma * sigma + EPS)), 0.0, 1.0))
    return torch.cat(chunks, dim=0)