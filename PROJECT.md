# trajectory_optimizer_torch — 项目描述与上下文

> 本文档解释本仓库在大项目中的位置、对最终任务的建模、实现策略、方法论与运行约定。
> 当本仓库被独立 clone 到训练服务器时，仅凭此文件即可还原它在 FOC-magnetic-gel 主项目中的语义。

---

## 1. 大项目背景

母项目：**FOC-magnetic-gel** —— 用 FOC 驱动的 XYZ 龙门架移动单极磁头，控制猫眼美甲胶中的猫眼颗粒（小磁化圆盘）排布出指定纹理，把"猫眼美甲"变成可编程自动化工艺。整体管线见 [idea.md](../../idea.md)：

1. 设计图 → 颗粒云（带朝向，类似 3D 打印的切片）。
2. 颗粒云 → 磁头 5/6-DoF 轨迹 + 磁矩 + 驻留时间。
3. 硬件按轨迹执行写入。

本仓库 (`trajectory_optimizer_torch`) 负责管线的 **第 2 步**：在给定平面图案目标下，反求一组离散写入路径点 (waypoint) 的位置、磁矩矢量与驻留时间，使得颗粒在过阻尼旋转动力学下被驱动到目标取向场。

兄弟上下文：

- [trajecory_planning/archived/dipole_reachability_proof.md](../archived/dipole_reachability_proof.md) — 单可移动点偶极子对时间积分磁场 $J = \int B\,dt$ 的可达性约束证明（不能任意，只能是调和标势的梯度场）。
- [trajecory_planning/archived/model_layers_and_assumptions.md](../archived/model_layers_and_assumptions.md) — 三种建模层级（线性角冲量 / 单帧稳态跟随 / 时序脉冲终态积分）的区分与适用边界。
- [trajecory_planning/archived/task.md](../archived/task.md) — 原始任务表述与工程化重构指引。
- [trajecory_planning/archived/function_dipole_trainer/](../archived/function_dipole_trainer/) — 本仓库的前身单脚本版本，保留作历史参考。

---

## 2. 物理与数学建模

### 2.1 工作平面与坐标

- 美甲平面位于 $z = 0$，工作区为以原点为中心的方形 $[-L,L]^2$（典型 $L = 5\,\text{mm}$，由 `grid.plane_half_size_m` 控制）。
- 磁头视为 **空间中可移动的点磁偶极子**，仅在 $z > 0$ 半空间运动（`trajectory_z_min_m` / `trajectory_z_max_m`）。
- 颗粒视为平面上的小圆盘，磁化沿径向；只跟踪绕平面法向的转角 $\theta(x,y,t)$。

### 2.2 磁场模型

每个 waypoint $i$ 对应位置 $c_i \in \mathbb{R}^3$、磁矩 $m_i \in \mathbb{R}^3$。点偶极子静磁场：

$$
B(r; c_i, m_i) = \frac{3(m_i\cdot\hat{r})\hat{r} - m_i}{\|r-c_i\|^3},\quad \hat r = \frac{r-c_i}{\|r-c_i\|}.
$$

实现见 `model.dipole_field_tensor`（省略 $\mu_0/4\pi$ 常数，由 `init/max_moment_strength` 吸收量纲）。

### 2.3 颗粒过阻尼旋转动力学

放弃"任意角冲量场"假设，采用过阻尼模型：

$$
\zeta\,\dot\theta = p_0\,|B_\parallel|\,\sin(\phi_B - \theta),
$$

其中 $\zeta$ = `rotational_drag_coefficient`，$p_0$ = `particle_moment_magnitude`，$B_\parallel$ 是平面内分量。
对一个 dwell 内 $B$ 视为常向量的近似下，存在闭式解：

$$
\tan\!\frac{\theta_{\text{new}}}{2} = \tan\!\frac{\theta}{2}\cdot e^{-(p_0|B|/\zeta)\,\Delta t}.
$$

实现见 `model.exact_align_vectors`。

附加项：

- **静摩擦阈值** `static_friction_torque`：当 $p_0|B|\sin\theta \le \tau_s$ 时颗粒不动（摩擦死区）。
- **可达半径** `dipole_effective_radius`：用阈值倒推每个 waypoint 在平面上能影响到的水平半径，预先剪枝候选点集，避免对所有 (point, waypoint) 对反传 → 显存与速度的关键优化。
- **亮度建模**：成像亮度 ≈ 颗粒平面内分量的取向程度 $\|n_{xy}\|^{\gamma}$（`brightness_gamma`），近似猫眼颗粒在径向磁化下被相机看到的"亮带"。

### 2.4 顺序写入语义

waypoint 序列按数组下标顺序顺序执行，颗粒状态在序列内累积（不可微的离散时间步顺序仿真）。这是 `SequentialOmniMagnetTrajectoryField._simulate_chunk` 的核心循环，整段对参数保持可微。

> **显存敏感点**：训练步前向反传保留整条顺序图。`waypoint_block_size` 控制按 waypoint 块切分以缓解，但不能改变图的本质连续性。详见 [/memories/repo/function_dipole_trainer_perf.md](../../) 记录。

---

## 3. 优化目标

### 3.1 决策变量

| 参数 | 张量 | 形状 | 是否优化 | 备注 |
| --- | --- | --- | --- | --- |
| 磁矩 $m_i$ | `raw_moment` (atanh 参数化) | `(W, 3)` | ✓ | 经 `tanh` 限幅到 `max_moment_strength` |
| 驻留时间 $\Delta t_i$ | `raw_dwell_time` (sigmoid 参数化) | `(W,)` | 由 `optimize_dwell_time` 控制 | 限幅到 `[min, max]_dwell_time_s` |
| 中心位置 $c_i$ | `center_offsets` | `(W, 3)` | 由 `freeze_centers` 控制；当前默认冻结 | 基础点由初始化器给定 |

W = waypoint 数量，由初始化器决定（典型 1000~2500）。

### 3.2 初始化器（trajectory_initializers.py）

当前仅 `fixed_grid`，两种几何先验：

- `uniform`: 三维均匀网格 $[x,y,z]$，`trajectory_step_m` 决定步长。
- `pyramid`: 多层 xy 平面，每层指定 `z_m` 与 `xy_step_m`，越靠近平面网格越细（更好的局部分辨率），越高层越稀（覆盖远程贡献）。详见 `function_dipole_trajectory_config.yaml` 示例。

> 几何先验和"位置是否参与优化"必须解耦 —— 见 [todo.md](todo.md) 关键设计决策。

### 3.3 损失函数（losses.py）

$$
\mathcal{L} = w_\text{img}\cdot \text{MSE}(b_{\text{pred}}, b_{\text{tgt}}) + \sum_k w_k R_k
$$

正则项 $R_k$：

- `moment_l2`：抑制磁矩过大（避免数值上的"极端解"）。
- `total_dwell_time`：抑制总驻留时间，鼓励工艺时间短的解。
- `smoothness` / `curvature` / `z_l2`：仅在 `freeze_centers=False` 时才生效（当前默认冻结，故为 0）。

权重通过 YAML `loss_weights` 配置。注意磁矩量纲极小（$10^{-8}$），故 `moment_l2` 权重数量级在 $10^{12}\sim10^{13}$。

### 3.4 目标图（targets.py）

`AnalyticGraphTarget` 接收一个 Python 表达式（如 `amplitude_m * tan(phase_scale_pi * pi * x / plane_half_size_m)`）作为解析曲线 $y=f(x)$，再由 `target_ridge_map` 把"点到曲线最短距离"通过高斯核 (`blur_sigma_m`) 转成像素亮度图。

距离计算：在每个有效分支区间内做多种子点 + Newton 投影（`projection.*` 配置），保证有 $\partial y/\partial x \to \infty$ 渐近线时仍稳定。

---

## 4. 训练流程（runner.py）

```
config.yaml
  └── load_experiment_config
       ├── build_grid (optimize / render)
       ├── AnalyticGraphTarget → target_ridge_map
       ├── build_trajectory_initializer → TrajectorySpec
       └── SequentialOmniMagnetTrajectoryField
            ↓ Adam(lr=5e-3) × steps
            ↓ full_image_mse + Σ regularizers
            ↓ early stop on patience
            ↓ load best_state
            ↓ render @ render_size
            └── reporting.py: SVG / TXT / index / contribution report
```

输出物（写入 `output_dir`）：

- `input_function_plot.svg` 解析目标曲线
- `target_ridge.svg` 模糊后的目标亮度图
- `fitted_ridge.svg` 拟合结果 + waypoint 标记
- `residual_map.svg` 残差
- `convergence.svg` loss 曲线
- `optimized_waypoints.txt` 最终参数表（mm + SI 单位）
- `index.json` / `function_description.txt` / `waypoint_contribution.csv`
- `tensorboard/` Tensorboard 日志

---

## 5. 运行约定

### 5.1 环境

- Python `>=3.10,<3.13`
- 依赖：`torch>=2.4` (cu130 wheel index)、`pyyaml`、`tensorboard`
- 推荐使用 `uv`：`uv sync` / `uv pip install <pkg>`
- 终端使用 bash（不要 PowerShell / cmd），见 [task.md](../archived/task.md)。

### 5.2 启动训练

```bash
source .venv/Scripts/activate    # Windows-MSYS / Git-Bash；Linux 用 bin/activate
python -m trajectory_optimizer_torch.runner --config <path>/function_dipole_trajectory_config.yaml
```

参考配置：[archived/function_dipole_trainer/function_dipole_trajectory_config.yaml](../archived/function_dipole_trainer/function_dipole_trajectory_config.yaml)。

### 5.3 GPU / 显存

- 训练每步显存主要来自 `_simulate_chunk` 的整条可微顺序仿真图；`point_chunk_size` 仅在前向点数 > 该值时分块。
- 提前用 `waypoint_candidate_mask` 剪枝是核心节流：摩擦阈值越小可达半径越大，剪枝越弱，显存越高。
- 默认 `optimize_size=36` (1296 训练点) × `pyramid` 2243 waypoint 在单张 16 GB GPU 上可跑。

---

## 6. 方法论与设计原则

1. **建模层级不混用**：本仓库使用 §2.3 的"时序脉冲终态积分"模型，不要回退到线性角冲量近似。
2. **可达性是物理事实**：单偶极子无法实现任意终态取向场（[dipole_reachability_proof.md](../archived/dipole_reachability_proof.md)）。本仓库的优化目标始终是 **拟合**，不追求 0 残差；评估解的好坏要看可视化与残差图，而不是 loss 数值。
3. **几何先验 vs 可训练参数严格分离**（[todo.md](todo.md)）：初始化器只决定几何，是否优化由 `freeze_centers` / `optimize_dwell_time` 决定。
4. **物理参数显式声明**：`particle_moment_magnitude` / `rotational_drag_coefficient` / `static_friction_torque` 都暴露到 YAML，便于跨颗粒型号 A/B 测试。
5. **配置即实验**：配置 + 随机种子 + 输出目录构成完整可复现单元；不要把训练逻辑参数化进硬编码。
6. **重构方向**（未完成，见 [todo.md](todo.md)）：迁向 `src/` 布局、callback 系统、可插拔初始化器注册表、smoke test。

---

## 7. 与母仓库的接口

| 上游 | 形式 | 当前状态 |
| --- | --- | --- |
| 设计图 → 颗粒云 → 解析曲线 | YAML 表达式 (`target.expression`) | 临时方案，未来应改为接收点云/位图目标 |
| 物理常数（颗粒/胶水） | YAML `model.*` 标量 | 来自 [parameter_calculation/](../../parameter_calculation/) 与磁头 COMSOL 仿真的标定 |

| 下游 | 形式 |
| --- | --- |
| 硬件执行 | `optimized_waypoints.txt`（mm 位置 + SI 磁矩 + 驻留时间），由龙门架 + 磁头驱动器消费 |

磁头侧的电磁建模与三极/三相定子相关结论保存在 [/memories/repo/magnetic_head_modeling.md](../../) 与 [parameter_calculation/teeth_leakage_tradeoff.py](../../parameter_calculation/teeth_leakage_tradeoff.py)；此处轨迹优化不直接使用电磁场仿真结果，而是以等效点偶极子参数 (`max_moment_strength`) 抽象上游能力。

---

## 8. 常见误区

- ❌ 把 `optimize_size` 设得过大：训练点数平方级别影响显存（`_simulate_chunk` 中候选 mask 是 `(P, W)` 形状）。
- ❌ 把 `init_moment_strength` 设小于摩擦死区临界值：所有点都不动，loss 不下降；模型会 print `warning_init_moment_strength_dead_zone`。
- ❌ 期望 loss → 0：物理可达性约束下不可能；评估靠残差图与 SVG 视觉。
- ❌ 误以为断电后颗粒会回零：过阻尼模型下 $B=0 \Rightarrow \dot\theta=0$，状态被冻结而不是回到初始角度（见 [model_layers_and_assumptions.md §3](../archived/model_layers_and_assumptions.md)）。

---

## 9. 仓库文件速查

| 文件 | 职责 |
| --- | --- |
| [config.py](config.py) | YAML → dataclass 配置 |
| [targets.py](targets.py) | 解析曲线、距离场、栅格化亮度目标 |
| [trajectory_initializers.py](trajectory_initializers.py) | waypoint 几何先验（uniform / pyramid） |
| [model.py](model.py) | 偶极子场 + 过阻尼对齐 + 顺序写入仿真 (核心) |
| [losses.py](losses.py) | 数据 loss + 正则项 |
| [runner.py](runner.py) | 训练编排 + 渲染 + 导出物料入口 |
| [reporting.py](reporting.py) | SVG / 索引 / waypoint 贡献报告 |
| [training_logging.py](training_logging.py) | Tensorboard 写入 |
| [utils.py](utils.py) | 表达式编译、数值常数、设备解析 |
| [todo.md](todo.md) | 重构与最佳实践路线图 |
