from __future__ import annotations

import torch

from .utils import clamp


def summarize_waypoint_contributions(model, points):
    centers = model.centers().detach()
    moments = model.moment_vec().detach()
    dwell_times = model.dwell_time_s().detach()
    waypoint_index = model.waypoint_index.detach().cpu().tolist()
    waypoint_count = centers.shape[0]

    mean_bxy_sum = torch.zeros(waypoint_count, dtype=torch.float64)
    max_bxy = torch.zeros(waypoint_count, dtype=torch.float64)
    mean_active_time_sum = torch.zeros(waypoint_count, dtype=torch.float64)
    mean_rotation_sum = torch.zeros(waypoint_count, dtype=torch.float64)
    mean_brightness_gain_sum = torch.zeros(waypoint_count, dtype=torch.float64)
    total_point_count = 0

    for chunk in points.split(model.point_chunk_size):
        step_stats = model._simulate_chunk(chunk, centers, moments, collect_step_stats=True)["step_stats"]
        mean_bxy_sum += step_stats["bxy_sum"].detach().cpu().to(torch.float64)
        max_bxy = torch.maximum(max_bxy, step_stats["max_bxy"].detach().cpu().to(torch.float64))
        mean_active_time_sum += step_stats["active_time_sum"].detach().cpu().to(torch.float64)
        mean_rotation_sum += step_stats["rotation_sum"].detach().cpu().to(torch.float64)
        mean_brightness_gain_sum += step_stats["brightness_gain_sum"].detach().cpu().to(torch.float64)
        total_point_count += chunk.shape[0]

    centers_cpu = centers.cpu()
    moments_cpu = moments.cpu()
    dwell_times_cpu = dwell_times.cpu()
    stats = []
    for index, waypoint_id in enumerate(waypoint_index):
        moment_norm = float(torch.linalg.norm(moments_cpu[index]).item())
        stats.append(
            {
                "index": waypoint_id,
                "center_mm": [1000.0 * value for value in centers_cpu[index].tolist()],
                "moment": moments_cpu[index].tolist(),
                "moment_norm": moment_norm,
                "dwell_time_s": float(dwell_times_cpu[index].item()),
                "mean_bxy": float(mean_bxy_sum[index].item() / total_point_count),
                "max_bxy": float(max_bxy[index].item()),
                "mean_active_time_s": float(mean_active_time_sum[index].item() / total_point_count),
                "mean_step_rotation": float(mean_rotation_sum[index].item() / total_point_count),
                "mean_brightness_gain": float(mean_brightness_gain_sum[index].item() / total_point_count),
            }
        )
    stats.sort(key=lambda item: item["mean_brightness_gain"], reverse=True)
    return stats


def write_waypoint_contribution_report(output_dir, stats):
    header = (
        "index,center_x_mm,center_y_mm,center_z_mm,moment_x,moment_y,moment_z,moment_norm,dwell_time_s,"
        "mean_bxy,max_bxy,mean_active_time_s,mean_step_rotation,mean_brightness_gain"
    )
    rows = [header]
    for item in stats:
        rows.append(
            ",".join(
                [
                    str(item["index"]),
                    f"{item['center_mm'][0]:.6f}",
                    f"{item['center_mm'][1]:.6f}",
                    f"{item['center_mm'][2]:.6f}",
                    f"{item['moment'][0]:.6e}",
                    f"{item['moment'][1]:.6e}",
                    f"{item['moment'][2]:.6e}",
                    f"{item['moment_norm']:.6e}",
                    f"{item['dwell_time_s']:.6e}",
                    f"{item['mean_bxy']:.6e}",
                    f"{item['max_bxy']:.6e}",
                    f"{item['mean_active_time_s']:.6e}",
                    f"{item['mean_step_rotation']:.6e}",
                    f"{item['mean_brightness_gain']:.6e}",
                ]
            )
        )
    (output_dir / "waypoint_contribution_stats.csv").write_text("\n".join(rows), encoding="utf-8")

    top_items = stats[:10]
    bottom_items = list(reversed(stats[-10:]))
    lines = ["top_10_by_mean_brightness_gain:"]
    lines.extend(
        f"index={item['index']:03d}, dwell_time_s={item['dwell_time_s']:.6e}, mean_brightness_gain={item['mean_brightness_gain']:.6e}, mean_step_rotation={item['mean_step_rotation']:.6e}, mean_active_time_s={item['mean_active_time_s']:.6e}"
        for item in top_items
    )
    lines.append("")
    lines.append("bottom_10_by_mean_brightness_gain:")
    lines.extend(
        f"index={item['index']:03d}, dwell_time_s={item['dwell_time_s']:.6e}, mean_brightness_gain={item['mean_brightness_gain']:.6e}, mean_step_rotation={item['mean_step_rotation']:.6e}, mean_active_time_s={item['mean_active_time_s']:.6e}"
        for item in bottom_items
    )
    (output_dir / "waypoint_contribution_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def polyline_pixels(sample_xy, plane_half_size_m, render_size):
    x = (sample_xy[:, 0] + plane_half_size_m) / (2.0 * plane_half_size_m) * (render_size - 1)
    y = (plane_half_size_m - sample_xy[:, 1]) / (2.0 * plane_half_size_m) * (render_size - 1)
    return list(zip(x.tolist(), y.tolist()))


def center_pixels(centers, plane_half_size_m, render_size):
    x = (centers[:, 0] + plane_half_size_m) / (2.0 * plane_half_size_m) * (render_size - 1)
    y = (plane_half_size_m - centers[:, 1]) / (2.0 * plane_half_size_m) * (render_size - 1)
    return list(zip(x.tolist(), y.tolist()))


def brightness_color(value):
    weight = clamp(value, 0.0, 1.0) ** 0.88
    dark = (5, 8, 12)
    bright = (245, 214, 118)
    channels = []
    for start, end in zip(dark, bright):
        channels.append(round(start + (end - start) * weight))
    return "#{:02x}{:02x}{:02x}".format(*channels)


def svg_raster(title, description, grid, brightness, polyline=None, markers=None):
    size = grid["size"]
    rects = []
    for (px, py), value in zip(grid["pixels"].tolist(), brightness.tolist()):
        rects.append(
            f'<rect x="{px:.3f}" y="{py:.3f}" width="1.1" height="1.1" fill="{brightness_color(value)}" />'
        )

    overlays = []
    if polyline:
        polyline_list = polyline if isinstance(polyline[0], list) else [polyline]
        for branch in polyline_list:
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in branch)
            overlays.append(
                f'<polyline points="{path}" fill="none" stroke="#c8d8ef" stroke-width="1.1" opacity="0.58" stroke-dasharray="3 3" />'
            )
    if markers:
        for x, y in markers:
            overlays.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.1" fill="#ff9f43" opacity="0.85" />')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 30}" width="900" height="1050">
  <rect width="100%" height="100%" fill="#05080c" />
  <g>{''.join(rects)}</g>
  <g>{''.join(overlays)}</g>
  <rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="#f1d479" stroke-width="1" opacity="0.55" />
  <text x="8" y="{size + 12}" fill="#f3e5b6" font-family="Georgia, serif" font-size="10">{title}</text>
  <text x="8" y="{size + 22}" fill="#b3b8bc" font-family="Georgia, serif" font-size="6">{description}</text>
</svg>
'''


def function_plot_svg(title, description, size, polylines):
    overlays = []
    for branch in polylines:
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in branch)
        overlays.append(
            f'<polyline points="{path}" fill="none" stroke="#f1d479" stroke-width="2.1" opacity="0.92" />'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 30}" width="900" height="1050">
  <rect width="100%" height="100%" fill="#05080c" />
  <rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="#49606a" stroke-width="1" opacity="0.8" />
  <line x1="0" y1="{0.5 * (size - 1):.2f}" x2="{size}" y2="{0.5 * (size - 1):.2f}" stroke="#2a3a43" stroke-width="0.9" opacity="0.85" />
  <line x1="{0.5 * (size - 1):.2f}" y1="0" x2="{0.5 * (size - 1):.2f}" y2="{size}" stroke="#2a3a43" stroke-width="0.9" opacity="0.85" />
  <g>{''.join(overlays)}</g>
  <text x="8" y="{size + 12}" fill="#f3e5b6" font-family="Georgia, serif" font-size="10">{title}</text>
  <text x="8" y="{size + 22}" fill="#b3b8bc" font-family="Georgia, serif" font-size="6">{description}</text>
</svg>
'''


def convergence_svg(history):
    width = 540
    height = 280
    margin = 28
    loss_values = [item[1] for item in history]
    min_loss = min(loss_values)
    max_loss = max(loss_values)
    span = max(max_loss - min_loss, 1.0e-9)
    points = []
    for step, loss_value in history:
        x = margin + (width - 2 * margin) * step / max(history[-1][0], 1)
        y = height - margin - (height - 2 * margin) * (loss_value - min_loss) / span
        points.append((x, y, loss_value))
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    markers = []
    for x, y, loss_value in points[:: max(1, len(points) // 12)]:
        markers.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#ffcc67" />')
        markers.append(
            f'<text x="{x + 4:.2f}" y="{y - 6:.2f}" fill="#d0d7db" font-size="10" font-family="Consolas, monospace">{loss_value:.4f}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="1080" height="560">
  <rect width="100%" height="100%" fill="#081015" />
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#49606a" stroke-width="1" />
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#49606a" stroke-width="1" />
  <polyline points="{polyline}" fill="none" stroke="#f1d479" stroke-width="2.4" />
  {''.join(markers)}
  <text x="28" y="18" fill="#f3e5b6" font-size="14" font-family="Georgia, serif">Torch Dipole Trajectory Training Loss</text>
</svg>
'''


def write_index(output_dir, best_loss, history, centers, moments, dwell_times, waypoint_index, device_name, target):
    lines = []
    for index, path_id in enumerate(waypoint_index.tolist()):
        center = centers[index].tolist()
        moment = moments[index].tolist()
        dwell_time_s = float(dwell_times[index].item())
        lines.append(
            "<li>"
            f"waypoint={path_id:02d}, center=({1000.0 * center[0]:+.2f}, {1000.0 * center[1]:+.2f}, {1000.0 * center[2]:.2f}) mm, "
            f"moment=({moment[0]:+.3e}, {moment[1]:+.3e}, {moment[2]:+.3e}), dwell_time={dwell_time_s:.4f} s"
            "</li>"
        )

    parameter_lines = "".join(f"<li>{name} = {value:.8g}</li>" for name, value in sorted(target.parameters.items()))
    layer_text = "".join(
        f"<li>z = {1000.0 * item['z_m']:.2f} mm, xy_step = {1000.0 * item['xy_step_m']:.3f} mm, grid = {item['x_count']} x {item['y_count']}</li>"
        for item in target.trajectory_layer_specs
    )
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Function Dipole Trajectory Trainer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #060b0e;
      --panel: #0d161c;
      --ink: #f4e4b8;
      --muted: #aeb6b9;
      --line: #243a44;
    }}
    body {{
      margin: 0;
      padding: 28px;
      background: radial-gradient(circle at 10% 0%, #172830 0%, #071015 42%, #030507 100%);
      color: var(--ink);
      font-family: Georgia, "Noto Serif SC", serif;
    }}
    h1 {{ margin: 0 0 12px 0; }}
    p {{ color: var(--muted); line-height: 1.7; max-width: 980px; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .meta div {{ background: rgba(13, 22, 28, 0.82); border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 22px; }}
    .card {{ background: linear-gradient(180deg, rgba(15, 26, 32, 0.96), rgba(7, 11, 15, 0.96)); border: 1px solid var(--line); border-radius: 18px; padding: 14px; }}
    .card img {{ width: 100%; border-radius: 12px; border: 1px solid rgba(241, 212, 121, 0.2); background: #05080c; }}
    ol, ul {{ color: var(--muted); line-height: 1.6; }}
    code {{ color: #f1d479; font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>Function Dipole Trajectory Trainer</h1>
  <p>目标由解析函数直接定义，不再读取 SVG 输入。高斯目标亮度面来自平面点到曲线 <code>y = f(x)</code> 的最近距离，再通过全图 MSE 优化一条单 OmniMagnet 的顺序写入路径。路径点位置固定为右手坐标系下的三维网格顶点，按 <code>x - y - z</code> 顺序遍历；其中美甲工作平面定义为 <code>z = 0</code>，<code>z</code> 轴垂直于美甲平面。轨迹既支持均匀网格，也支持高 z 粗步长、低 z 细步长的金字塔多尺度模式。当前优化变量包括每个固定驻留点的偶极矩 <code>(m_x, m_y, m_z)</code> 和驻留时间 <code>t</code>。转动模型采用硬静摩擦阈值触发和常场精确时间积分，训练时额外对总驻留时间加权惩罚。训练时优先请求 CUDA，但本次实际运行设备是 <code>{device_name}</code>。</p>
  <p>Function expression: <code>{target.expression_text}</code>. Best full-image MSE: {best_loss:.6f}. Logged points: {len(history)}.</p>
  <section class="meta">
    <div>Trajectory mode: {target.trajectory_mode}</div>
    <div>Grid shape: {target.trajectory_grid_shape_text}</div>
    <div>Grid step: {target.trajectory_step_text}</div>
    <div>XY span: {2000.0 * target.trajectory_xy_half_extent_m:.2f} mm x {2000.0 * target.trajectory_xy_half_extent_m:.2f} mm</div>
    <div>Z range: [{1000.0 * target.trajectory_z_min_m:.2f}, {1000.0 * target.trajectory_z_max_m:.2f}] mm</div>
    <div>Total dwell time: {target.total_dwell_time_s:.4f} s</div>
  </section>
  <h2>Trajectory Layers</h2>
  <ul>{layer_text}</ul>
  <div class="grid">
    <article class="card"><img src="input_function_plot.svg" alt="input function plot" /><p>解析输入函数在 1 cm x 1 cm 平面中的可见分支 plot。</p></article>
    <article class="card"><img src="target_ridge.svg" alt="target ridge" /><p>由解析函数距离场生成的高斯目标亮度面。</p></article>
    <article class="card"><img src="fitted_ridge.svg" alt="fitted ridge" /><p>单 OmniMagnet 顺序写入后的拟合结果。橙点是龙门路径驻留点位置。</p></article>
    <article class="card"><img src="residual_map.svg" alt="residual map" /><p>全图绝对残差图。这里每个像素都参与损失，而不是只看亮区域。</p></article>
    <article class="card"><img src="convergence.svg" alt="convergence" /><p>梯度下降收敛曲线。</p></article>
  </div>
  <p>每个路径点的写入统计已输出到 <code>waypoint_contribution_stats.csv</code> 和 <code>waypoint_contribution_summary.txt</code>，可直接检查哪些驻留点几乎没有带来有效旋转或亮度增益。</p>
  <h2>Function Parameters</h2>
  <ul>{parameter_lines}</ul>
  <h2>Optimized Write Waypoints</h2>
  <ol>{''.join(lines)}</ol>
</body>
</html>
'''
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def write_function_description(output_dir, target):
    lines = [f"expression: {target.expression_text}"]
    if target.valid_expression_text:
        lines.append(f"valid_expression: {target.valid_expression_text}")
    lines.append(f"x_range_m: [{target.x_min_m}, {target.x_max_m}]")
    lines.append(f"y_range_m: [{target.y_min_m}, {target.y_max_m}]")
    lines.append(f"trajectory_mode: {target.trajectory_mode}")
    lines.append(f"trajectory_grid_shape: {target.trajectory_grid_shape_text}")
    lines.append(f"trajectory_step: {target.trajectory_step_text}")
    lines.append(f"trajectory_xy_half_extent_m: {target.trajectory_xy_half_extent_m}")
    lines.append(f"trajectory_z_range_m: [{target.trajectory_z_min_m}, {target.trajectory_z_max_m}]")
    lines.append(f"particle_moment_magnitude: {target.particle_moment_magnitude}")
    lines.append(f"rotational_drag_coefficient: {target.rotational_drag_coefficient}")
    lines.append(f"static_friction_torque: {target.static_friction_torque}")
    lines.append(f"alignment_rate_per_field: {target.alignment_rate_per_field}")
    lines.append(f"friction_field_threshold: {target.friction_field_threshold}")
    lines.append(f"dwell_time_range_s: [{target.min_dwell_time_s}, {target.max_dwell_time_s}]")
    lines.append(f"total_dwell_time_s: {target.total_dwell_time_s}")
    for index, item in enumerate(target.trajectory_layer_specs):
        lines.append(
            f"trajectory_layer_{index}: z_m={item['z_m']}, xy_step_m={item['xy_step_m']}, grid={item['x_count']}x{item['y_count']}"
        )
    for name, value in sorted(target.parameters.items()):
        lines.append(f"{name}: {value}")
    (output_dir / "function_definition.txt").write_text("\n".join(lines), encoding="utf-8")