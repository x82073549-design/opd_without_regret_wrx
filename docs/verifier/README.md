# OPD Verifier 文档索引

本目录保存 OPD Loss Baseline、State 0、Evolving Pipeline、研究提案和会议记录。

## 当前入口

1. [State 0 选择与复现实验计划](./plans/state-0-selection.md)：当前 Baseline 实验的执行依据。
2. [2026-08-14 会议纪要](./meetings/2026-08-14-project.md)：最新会议决定、负责人和待办事项。
3. [OPD Loss 论文索引](./references/opd-loss-literature.md)：State 0 方法及其他相关工作的资料来源。

当前优先事项是完成 Fixed OPD、EOPD、AOPD、TIP、Prune-OPD 和 ExOPD 在统一 setting 下的完整实验。后续 Evolving Pipeline 尚未冻结。

## 目录结构

### `plans/`

- [State 0 选择与复现实验计划](./plans/state-0-selection.md)：当前执行方案。
- [2026-08-04 前期实验计划](./plans/preliminary-experiments-2026-08-04.md)：历史方案，保留用于追踪训练步数、early stopping 和早期 action 设计。

### `proposals/`

以下文件是研究提案，不代表当前已经采用的实现方案：

- [检查点层动作效用控制](./proposals/checkpoint-level-control.md)
- [Token Action-Pair 选择](./proposals/token-action-pair-selection.md)
- [Learning-Utility Verifier Formalization](./proposals/learning-utility-verifier.md)

### `references/`

- [OPD Loss 论文索引](./references/opd-loss-literature.md)

### `reports/`

- [四轮 Loss Search 实验记录](./reports/loss-search-rounds.md)：历史自动搜索的过程、结果和失败原因。
- [Evolving Pipeline Overview](./reports/evolve-pipeline-overview.html)：基于早期搜索记录形成的系统审计和长期路线图。

上述报告用于复盘已有实验，不等同于当前 State 0 执行计划。

### `meetings/`

- [2026-07-31 Loss Evolving 讨论](./meetings/2026-07-31-loss-evolving.md)：历史版本。
- [2026-08-05 Loss Evolving 更新](./meetings/2026-08-05-loss-evolving.md)：早期 memory、action 和 early stopping 决策。
- [2026-08-14 项目会议](./meetings/2026-08-14-project.md)：当前最新会议记录。

### `schemas/`

- [Validation manifest 示例](./schemas/validation-manifest.example.json)

## 文档状态约定

- **当前执行方案**：已被会议确认，可直接用于分工和实验验收。
- **研究提案**：用于讨论和设计，不代表已经进入实现。
- **历史方案/记录**：保留研究过程和证据，但不得覆盖当前执行方案。
- **参考资料**：提供论文和方法信息，不直接作为实验结果。
