from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig, load_experiment_config
from .losses import all_finite_tensors, full_image_mse, model_regularization_terms
from .model import SequentialOmniMagnetTrajectoryField
from .reporting import (
    center_pixels,
    convergence_svg,
    function_plot_svg,
    polyline_pixels,
    summarize_waypoint_contributions,
    svg_raster,
    write_function_description,
    write_index,
    write_waypoint_contribution_report,
)
from .targets import AnalyticGraphTarget, build_grid, target_ridge_map
from .training_logging import create_tensorboard_writer, log_tensorboard_step
from .trajectory_initializers import TrajectoryInitializer, build_trajectory_initializer
from .utils import resolve_device


@dataclass(slots=True)
class TrainingResult:
    best_loss: float
    output_dir: Path
    tensorboard_log_dir: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the modular torch trajectory optimizer.")
    parser.add_argument("--config", default="function_dipole_trajectory_config.yaml", help="Path to YAML config file.")
    return parser.parse_args()


def _attach_target_metadata(target: AnalyticGraphTarget, model: SequentialOmniMagnetTrajectoryField) -> None:
    target.trajectory_mode = model.trajectory_mode
    target.trajectory_grid_shape = model.trajectory_grid_shape
    target.trajectory_grid_shape_text = model.trajectory_grid_shape_text
    target.trajectory_step_m = model.trajectory_step_m
    target.trajectory_step_text = model.trajectory_step_text
    target.trajectory_xy_half_extent_m = model.trajectory_xy_half_extent_m
    target.trajectory_z_min_m = model.trajectory_z_min_m
    target.trajectory_z_max_m = model.trajectory_z_max_m
    target.trajectory_layer_specs = model.trajectory_layer_specs
    target.particle_moment_magnitude = model.particle_moment_magnitude
    target.rotational_drag_coefficient = model.rotational_drag_coefficient
    target.static_friction_torque = model.static_friction_torque
    target.alignment_rate_per_field = model.alignment_rate_per_field()
    target.friction_field_threshold = model.friction_field_threshold()
    target.min_dwell_time_s = model.min_dwell_time_s
    target.max_dwell_time_s = model.max_dwell_time_s


def run_training(
    config: ExperimentConfig,
    config_path: Path | None = None,
    trajectory_initializer: TrajectoryInitializer | None = None,
) -> TrainingResult:
    torch.manual_seed(config.seed)
    device = resolve_device(config.device_preference, config.device_ids)
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    output_dir = base_dir / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()
    tensorboard_writer, tensorboard_log_dir = create_tensorboard_writer(output_dir, config_dict)

    optimize_grid = build_grid(config.grid.optimize_size, config.grid.plane_half_size_m, device)
    render_grid = build_grid(config.grid.render_size, config.grid.plane_half_size_m, device)

    target = AnalyticGraphTarget(config.grid, config.target)
    target_samples = target.build_visible_samples(config.target.init_sample_count, device)
    plot_samples = target.build_visible_samples(config.target.plot_sample_count, device)

    target_optimize = target_ridge_map(
        optimize_grid.points,
        target,
        config.target.blur_sigma_m,
        config.target.intensity_gain,
    )
    target_render = target_ridge_map(
        render_grid.points,
        target,
        config.target.blur_sigma_m,
        config.target.intensity_gain,
    )

    initializer = trajectory_initializer or build_trajectory_initializer(config)
    trajectory_spec = initializer.build(config, device)
    model = SequentialOmniMagnetTrajectoryField(config, target_samples, trajectory_spec, device)
    _attach_target_metadata(target, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
    history: list[tuple[int, float]] = []
    best_loss = float("inf")
    best_state = None
    patience = 0
    loss_weights = dict(config.loss_weights)

    for step in range(config.optimizer.steps):
        optimizer.zero_grad(set_to_none=True)
        forward = model(optimize_grid.points)
        data_loss = full_image_mse(forward["brightness"], target_optimize)
        reg_terms = model_regularization_terms(model, loss_weights)
        reg_loss = reg_terms["total"]
        loss = loss_weights["full_image_mse"] * data_loss + reg_loss
        if not torch.isfinite(loss):
            print(f"non_finite_loss at step={step:04d}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        optimizer.step()

        if not all_finite_tensors(model):
            print(f"non_finite_parameters at step={step:04d}")
            break

        loss_value = float(loss.detach().cpu().item())
        data_loss_value = float(data_loss.detach().cpu().item())
        history.append((step, loss_value))
        log_tensorboard_step(tensorboard_writer, step, loss, data_loss, reg_terms, model, forward)

        if loss_value + config.optimizer.min_delta < best_loss:
            best_loss = loss_value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if step % config.optimizer.print_every == 0 or step == config.optimizer.steps - 1:
            active_mask = forward["active_time"] > 0.0
            active_count = int(active_mask.sum().detach().cpu().item())
            active_fraction = float(active_mask.to(dtype=forward["active_time"].dtype).mean().detach().cpu().item())
            mean_activation = float(forward["activation"].mean().detach().cpu().item())
            print(
                f"step={step:04d} loss={loss_value:.6f} "
                f"data={data_loss_value:.6f} "
                f"moment_l2={float(reg_terms['moment_l2'].detach().cpu().item()):.3e} "
                f"moment_l2_penalty={float(reg_terms['moment_l2_penalty'].detach().cpu().item()):.3e} "
                f"time_penalty={float(reg_terms['total_dwell_time_penalty'].detach().cpu().item()):.3e} "
                f"active_points={active_count}/{forward['active_time'].numel()} "
                f"active_ratio={active_fraction:.3f} "
                f"mean_activation={mean_activation:.3f} "
                f"device={device.type}"
            )

        if patience >= config.optimizer.early_stop_patience:
            print(f"early_stop at step={step:04d} best_loss={best_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    del optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        fitted = model(render_grid.points)
        brightness = fitted["brightness"].detach().cpu()
        centers = fitted["centers"].detach().cpu()
        moments = fitted["moment_vec"].detach().cpu()
        dwell_times = fitted["dwell_time_s"].detach().cpu()
        waypoint_index = fitted["waypoint_index"].detach().cpu()
        target.total_dwell_time_s = float(fitted["total_dwell_time_s"].detach().cpu().item())
        residual = (target_render.detach().cpu() - brightness).abs()
        contribution_stats = summarize_waypoint_contributions(model, render_grid.points)

    polyline = [
        polyline_pixels(poly.detach().cpu(), config.grid.plane_half_size_m, config.grid.render_size)
        for poly in plot_samples.polyline_xy
    ]
    markers = center_pixels(centers, config.grid.plane_half_size_m, config.grid.render_size)
    render_grid_cpu = {"size": render_grid.size, "pixels": render_grid.pixels.detach().cpu()}

    (output_dir / "input_function_plot.svg").write_text(
        function_plot_svg(
            title="Input analytic function",
            description="Visible branches of the configured analytic target y = f(x).",
            size=render_grid_cpu["size"],
            polylines=polyline,
        ),
        encoding="utf-8",
    )
    (output_dir / "target_ridge.svg").write_text(
        svg_raster(
            title="Analytic target ridge",
            description="Blurred brightness target generated from point-to-curve distance.",
            grid=render_grid_cpu,
            brightness=target_render.detach().cpu(),
            polyline=polyline,
        ),
        encoding="utf-8",
    )
    (output_dir / "fitted_ridge.svg").write_text(
        svg_raster(
            title="Sequential OmniMagnet fit",
            description="Training result with sequential write waypoints and full-image loss.",
            grid=render_grid_cpu,
            brightness=brightness,
            polyline=polyline,
            markers=markers,
        ),
        encoding="utf-8",
    )
    (output_dir / "residual_map.svg").write_text(
        svg_raster(
            title="Absolute residual map",
            description="Per-pixel residual over the full 1 cm x 1 cm plane.",
            grid=render_grid_cpu,
            brightness=residual,
            polyline=polyline,
        ),
        encoding="utf-8",
    )
    (output_dir / "convergence.svg").write_text(convergence_svg(history), encoding="utf-8")
    (output_dir / "optimized_waypoints.txt").write_text(
        "\n".join(
            f"waypoint={int(path_id):02d}, center=({1000.0 * center[0]:+.3f}, {1000.0 * center[1]:+.3f}, {1000.0 * center[2]:.3f}) mm, "
            f"moment=({moment[0]:+.4e}, {moment[1]:+.4e}, {moment[2]:+.4e}), dwell_time={dwell_time:.5f} s"
            for path_id, center, moment, dwell_time in zip(
                waypoint_index.tolist(), centers.tolist(), moments.tolist(), dwell_times.tolist()
            )
        ),
        encoding="utf-8",
    )
    write_index(output_dir, best_loss, history, centers, moments, dwell_times, waypoint_index, device.type, target)
    write_function_description(output_dir, target)
    write_waypoint_contribution_report(output_dir, contribution_stats)

    print(f"best_loss={best_loss:.6f}")
    print(f"device={device.type}")
    print(f"trajectory_mode={model.trajectory_mode}")
    print(f"trajectory_grid_shape={model.trajectory_grid_shape_text}")
    print(f"waypoints={model.centers().shape[0]}")
    print(f"alignment_rate_per_field={model.alignment_rate_per_field():.6e}")
    print(f"friction_field_threshold={model.friction_field_threshold():.6e}")
    print(f"total_dwell_time_s={float(model.total_dwell_time_s().detach().cpu().item()):.6f}")
    if tensorboard_log_dir is not None:
        print(f"tensorboard_log_dir={tensorboard_log_dir}")
    print(f"expression={target.expression_text}")
    print(f"output_dir={output_dir}")

    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    return TrainingResult(best_loss=best_loss, output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    run_training(config, config_path=config_path)


if __name__ == "__main__":
    main()