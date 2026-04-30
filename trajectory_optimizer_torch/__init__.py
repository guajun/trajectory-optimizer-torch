from .config import ExperimentConfig, load_experiment_config

__all__ = ["ExperimentConfig", "load_experiment_config", "run_training"]


def __getattr__(name: str):
    # Lazy import so `python -m trajectory_optimizer_torch.runner` does not
    # double-load the runner module (avoids RuntimeWarning from runpy).
    if name == "run_training":
        from .runner import run_training
        return run_training
    raise AttributeError(f"module 'trajectory_optimizer_torch' has no attribute {name!r}")