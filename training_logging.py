from __future__ import annotations

import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def create_tensorboard_writer(output_dir, config):
    tensorboard_cfg = config.get("tensorboard", {})
    enabled = bool(tensorboard_cfg.get("enabled", True))
    if not enabled:
        return None, None
    if SummaryWriter is None:
        print("tensorboard_unavailable missing_tensorboard_package")
        return None, None

    log_dir_name = str(tensorboard_cfg.get("log_dir", "tensorboard"))
    log_dir = (output_dir / log_dir_name).resolve()
    flush_secs = int(tensorboard_cfg.get("flush_secs", 10))
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
    writer.add_text("run/config_yaml", yaml.safe_dump(config, sort_keys=False), 0)
    return writer, log_dir


def log_tensorboard_step(writer, step, loss, data_loss, reg_terms, model, forward):
    if writer is None:
        return

    active_mask = forward["active_time"] > 0.0
    active_point_fraction = active_mask.to(dtype=forward["active_time"].dtype).mean()

    writer.add_scalar("loss/total", float(loss.detach().cpu().item()), step)
    writer.add_scalar("loss/data_mse", float(data_loss.detach().cpu().item()), step)
    writer.add_scalar("loss/reg_total", float(reg_terms["total"].detach().cpu().item()), step)
    writer.add_scalar("loss/moment_l2_raw", float(reg_terms["moment_l2"].detach().cpu().item()), step)
    writer.add_scalar("loss/moment_l2_penalty", float(reg_terms["moment_l2_penalty"].detach().cpu().item()), step)
    writer.add_scalar(
        "loss/total_dwell_time_penalty",
        float(reg_terms["total_dwell_time_penalty"].detach().cpu().item()),
        step,
    )
    writer.add_scalar("state/total_dwell_time_s", float(reg_terms["total_dwell_time"].detach().cpu().item()), step)
    writer.add_scalar("state/mean_dwell_time_s", float(model.dwell_time_s().mean().detach().cpu().item()), step)
    writer.add_scalar("state/min_dwell_time_s", float(model.dwell_time_s().min().detach().cpu().item()), step)
    writer.add_scalar("state/max_dwell_time_s", float(model.dwell_time_s().max().detach().cpu().item()), step)

    moment_norm = model.moment_vec().norm(dim=-1)
    writer.add_scalar("state/mean_moment_norm", float(moment_norm.mean().detach().cpu().item()), step)
    writer.add_scalar("state/max_moment_norm", float(moment_norm.max().detach().cpu().item()), step)
    writer.add_scalar("state/pred_brightness_mean", float(forward["brightness"].mean().detach().cpu().item()), step)
    writer.add_scalar("state/pred_brightness_max", float(forward["brightness"].max().detach().cpu().item()), step)
    writer.add_scalar("state/active_point_count", float(active_mask.sum().detach().cpu().item()), step)
    writer.add_scalar("state/active_point_fraction", float(active_point_fraction.detach().cpu().item()), step)
    writer.add_scalar("state/mean_activation", float(forward["activation"].mean().detach().cpu().item()), step)
    writer.add_scalar("state/max_activation", float(forward["activation"].max().detach().cpu().item()), step)