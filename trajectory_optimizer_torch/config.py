from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str = "tensorboard"
    flush_secs: int = 10


@dataclass(slots=True)
class GridConfig:
    plane_half_size_m: float
    optimize_size: int
    render_size: int


@dataclass(slots=True)
class ProjectionConfig:
    batch_size: int = 4096
    newton_steps: int = 8
    interval_scan_count: int = 4097
    global_seed_count: int = 9
    local_initial_offsets_m: list[float] = field(default_factory=lambda: [0.0])


@dataclass(slots=True)
class TargetConfig:
    expression: str
    valid_expression: str | None = None
    x_min_m: float | None = None
    x_max_m: float | None = None
    y_min_m: float | None = None
    y_max_m: float | None = None
    blur_sigma_m: float = 0.00024
    intensity_gain: float = 1.0
    init_sample_count: int = 1400
    plot_sample_count: int = 2200
    parameters: dict[str, float] = field(default_factory=dict)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)


@dataclass(slots=True)
class TrajectoryLayerConfig:
    z_m: float
    xy_step_m: float


@dataclass(slots=True)
class TrajectoryCurveBandLayerConfig:
    z_m: float
    along_step_m: float
    band_half_width_m: float = 0.0
    band_step_m: float | None = None


@dataclass(slots=True)
class ModelConfig:
    trajectory_initializer: str = "fixed_grid"
    trajectory_mode: str = "uniform"
    trajectory_step_m: float = 0.001
    trajectory_z_min_m: float = 0.01
    trajectory_z_max_m: float = 0.02
    trajectory_pyramid_layers: list[TrajectoryLayerConfig] = field(default_factory=list)
    trajectory_curve_band_layers: list[TrajectoryCurveBandLayerConfig] = field(default_factory=list)
    freeze_centers: bool = True
    init_moment_strength: float = 2.4e-8
    max_moment_strength: float = 8.0e-8
    particle_moment_magnitude: float = 2.0e-11
    rotational_drag_coefficient: float = 2.0e-13
    static_friction_torque: float = 1.0e-12
    optimize_dwell_time: bool = True
    min_dwell_time_s: float = 0.003
    max_dwell_time_s: float = 0.050
    init_dwell_time_s: float = 0.008
    init_activation_margin: float = 1.05
    brightness_gamma: float = 0.92
    recency_scale: float = 0.0
    waypoint_block_size: int = 32
    point_chunk_size: int = 2048


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "adam"
    steps: int = 1500
    learning_rate: float = 0.005
    max_grad_norm: float = 1.0
    print_every: int = 1
    early_stop_patience: int = 120
    min_delta: float = 1.0e-5
    # L-BFGS-specific options (ignored for other optimizers).
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 20
    lbfgs_tolerance_grad: float = 1.0e-7
    lbfgs_tolerance_change: float = 1.0e-9
    lbfgs_line_search: str = "strong_wolfe"


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    device_preference: str
    device_ids: list[int] = field(default_factory=list)
    output_dir: str = "optimized_function_dipole_torch"
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    grid: GridConfig = field(default_factory=lambda: GridConfig(0.005, 36, 180))
    target: TargetConfig = field(default_factory=lambda: TargetConfig(expression="0.0"))
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        tensorboard = TensorBoardConfig(**payload.get("tensorboard", {}))
        grid = GridConfig(**payload["grid"])
        projection = ProjectionConfig(**payload["target"]["projection"])
        target_payload = dict(payload["target"])
        target_payload["parameters"] = {
            key: float(value) for key, value in target_payload.get("parameters", {}).items()
        }
        target_payload["projection"] = projection
        target = TargetConfig(**target_payload)
        model_payload = dict(payload["model"])
        model_payload["trajectory_pyramid_layers"] = [
            TrajectoryLayerConfig(**item) for item in model_payload.get("trajectory_pyramid_layers", [])
        ]
        model_payload["trajectory_curve_band_layers"] = [
            TrajectoryCurveBandLayerConfig(**item)
            for item in model_payload.get("trajectory_curve_band_layers", [])
        ]
        model = ModelConfig(**model_payload)
        optimizer = OptimizerConfig(**payload["optimizer"])
        loss_weights = {key: float(value) for key, value in payload.get("loss_weights", {}).items()}
        return cls(
            seed=int(payload["seed"]),
            device_preference=str(payload["device_preference"]),
            device_ids=[int(item) for item in payload.get("device_ids", [])],
            output_dir=str(payload.get("output_dir", "optimized_function_dipole_torch")),
            tensorboard=tensorboard,
            grid=grid,
            target=target,
            model=model,
            optimizer=optimizer,
            loss_weights=loss_weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(payload)