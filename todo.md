# trajectory_optimizer_torch TODO

## 目标架构

统一目标：把当前仓库收敛成一个独立、可实验、可复现、可扩展的参数优化项目，而不是单脚本搬家版。

建议的最终目录：

```text
trajectory_optimizer_torch/
  pyproject.toml
  README.md
  todo.md
  configs/
    examples/
      tan_curve_pyramid.yaml
  src/
    trajectory_optimizer_torch/
      __init__.py
      cli/
        train.py
        export.py
      config/
        schema.py
        io.py
      targets/
        analytic_graph.py
        ridge_map.py
      initializers/
        base.py
        fixed_grid.py
        pyramid.py
        curve_band.py
      models/
        sequential_dipole_field.py
      physics/
        dipole.py
        rotation.py
      losses/
        image.py
        regularization.py
      training/
        runner.py
        callbacks.py
        checkpointing.py
        logging.py
        state.py
      reports/
        artifacts.py
        svg.py
        contribution.py
      utils/
        devices.py
        math_expr.py
  tests/
    test_config.py
    test_targets.py
    test_initializers.py
    test_model_shapes.py
    test_runner_smoke.py
```

## 分层原则

- `config/` 只负责类型化配置、默认值、YAML 读写、配置校验，不参与训练逻辑。
- `targets/` 只负责目标定义和目标图生成，不知道优化器、TensorBoard、SVG。
- `initializers/` 只负责几何先验和轨迹初始状态构造。
- `models/` 只负责可训练参数与 forward，不直接读取 YAML，不直接写文件。
- `physics/` 放可复用的磁场、旋转、阈值积分实现，避免散落在模型里。
- `losses/` 负责 data loss 和 regularization，避免 runner 直接拼 loss。
- `training/runner.py` 只负责编排训练。
- `training/callbacks.py` 负责 TensorBoard、early stopping、checkpoint、console logging。
- `reports/` 负责训练后导出物料，和主训练循环解耦。

## 关键设计决策

- `train.py` 作为唯一入口，但内部应调用 `training.runner.run_training()`。
- `run_training()` 同时支持 `ExperimentConfig` 和显式传入的 `TrajectoryInitializer` 对象。
- “轨迹初始化器”和“位置是否可训练”必须分离：
  - 初始化器决定初始几何先验。
  - `freeze_centers` 决定位置参数是否参与优化。
- `center_offsets` 是位置训练参数的统一形式；固定中心时不创建该参数。
- moments、dwell times、centers 应支持参数分组和不同学习率。

## 当前代码的主要缺口

- 当前仓库还是扁平文件结构，长期应迁移到 `src/trajectory_optimizer_torch/` 真包结构。
- `runner.py` 仍依赖外部 `train_function_dipole_trajectory_torch`，这说明仓库还不是独立闭环。
- loss、TensorBoard、artifact export 仍未完全内聚到本仓库内部模块。
- 还没有 callback 机制，训练循环仍承担太多职责。
- 还没有测试，难以安全演进物理模型和几何先验。

## 重构优先级

### Phase 1: 独立闭环

- [ ] 去掉 `runner.py` 对外部 `train_function_dipole_trajectory_torch` 的依赖。
- [ ] 把 loss、TensorBoard logging、artifact export 迁回本仓库。
- [ ] 补一个真正可运行的 `cli/train.py`。
- [ ] 把仓库迁移为 `src/trajectory_optimizer_torch/` 布局。

### Phase 2: 训练工程化

- [ ] 引入 callback 系统：console logger、TensorBoard logger、early stop、checkpoint。
- [ ] 引入 `TrainingState` 数据结构，统一记录 step、best metrics、history。
- [ ] 支持 optimizer param groups，给 moments、dwell times、centers 分开配学习率。
- [ ] 增加 scheduler 钩子，支持 plateau 降学习率。

### Phase 3: 初始化器扩展

- [ ] 保留 `fixed_grid` 初始化器。
- [ ] 增加 `single_plane` 初始化器。
- [ ] 增加 `curve_band` 初始化器，用目标曲线窄带替代整平面铺点。
- [ ] 增加可插拔初始化器注册表，而不是在 if/else 里硬编码。

### Phase 4: 质量保障

- [ ] 补 smoke tests，确保 config -> target -> model -> runner 基本链路可跑通。
- [ ] 补 shape tests 和数值稳定性测试。
- [ ] 补 minimal example configs，降低复现实验门槛。
- [ ] 增加 CI：至少做 `uv lock`、导入检查和 smoke test。

## 机器学习与参数优化最佳实践

- 配置和运行状态分离：配置不可变，训练结果单独记录。
- 把实验产物当作 artifact 管理：日志、权重、图像、贡献报告都要有清晰输出路径。
- 所有可优化参数显式建模，不要隐藏在初始化逻辑里。
- 所有近似和物理假设模块化，便于做 A/B 实验。
- 优先保留程序化 API，再叠加 CLI；不要反过来让 CLI 控制核心逻辑。
- 先保证 smoke test 可跑，再做大规模结构迁移。