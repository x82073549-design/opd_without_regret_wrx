# OPD without regret · Codex Self-Evolving Loss Search

> 整理：Xiao
> 日期：2026-07-31
> 状态：会议后当前主线方案
> 范围：仅记录 OPD / loss 自动搜索相关结论；不包含另一项 multi-agent 通信优化课题及项目协调事项。

## 1. 会议后的核心决策

当前主线不再采用“先训练 learning-utility verifier/controller，再由它选择 token action”的两层架构。第一版直接使用 Codex（候选模型包括 GPT-5.6）作为 training-free 的上层优化器，根据真实的小规模训练反馈迭代提出新的 loss function。

形式化为：

\[
\mathcal O_{\mathrm{Codex}}
\left(
s_k,\,
\mathcal L_k,\,
\mathcal M_k
\right)
\rightarrow
\mathcal L_{k+1},
\]

其中：

- \(s_k\)：当前实验状态与诊断，包括训练/验证指标、student–teacher 分布统计、梯度与数值稳定性信息；
- \(\mathcal L_k\)：当前 loss 的公式、Python 实现、超参数与约束；
- \(\mathcal M_k\)：共享搜索记忆，包括历史 proposal、配置、结果、失败原因和 lineage；
- \(\mathcal L_{k+1}\)：Codex 提出的下一版可执行 loss program。

候选 loss 的质量不由另一个学习到的 verifier 预测，而由实际短程训练产生的验证反馈评价：

\[
R(\mathcal L)
=
\max_{1\le h\le100}
J_{\mathrm{search\text{-}val}}
\left(
\theta_h^{\mathcal L}
\right),
\]

其中所有候选均从相同初始状态 \(\theta_0\) 开始。第一阶段以 **100 optimizer steps 内的最佳 search-validation 表现**作为低成本筛选指标；它是搜索 proxy，不直接等同于完整训练后的最终结论。

## 2. 搜索对象：受约束的 loss program

原来的七个 token actions：

\[
\{\mathrm{skip}\}
\cup
\{\mathrm{FKL},\mathrm{RKL/OPD}\}
\times
\{0.5,1,2\}
\]

不再是最终封闭 action space，而作为初始原子算子的一个子集。Codex 可以组合并演化：

- 基础 divergence：FKL、RKL/OPD 及经过批准的其他可微目标；
- token / trajectory weighting；
- gating、threshold 与 pre-filter；
- clipping、normalization 和 loss-scale matching；
- mixture、margin、temperature 与 regularization；
- 简单的训练阶段 schedule。

条件与算子可以封装成更大的复合算子。例如，一个算子可以先根据 value 与阈值 \(\lambda\) 过滤低价值 trajectory，再对剩余 token 应用带 clipping 的 OPD 权重。

为控制搜索成本和保证可审计性，M0 使用受约束 DSL / Python 模板：

1. 最多两层算子组合；
2. loss 必须可微、数值有限，并通过小 batch smoke test；
3. 不允许读取 search-validation label 作为训练输入；
4. 必须显式记录权重范围、归一化、clipping 和 fallback；
5. 每个 proposal 必须保存公式、代码 diff、父方案和提出理由；
6. 不允许 Codex 任意修改训练框架、数据或 evaluation protocol。

## 3. 降低搜索成本的两阶段策略

### Stage A：原子算子发现

先让 Codex 基于现有 OPD 实现、历史实验与研究直觉提出一组有区分度的原子方案。使用几百至几千条训练样本进行 smoke test 和短程探测，排除：

- loss 不可运行或出现 NaN / Inf；
- 梯度退化、爆炸或几乎等同 baseline；
- 明显违反预算或产生极端权重集中；
- 在短程评估中稳定劣于 fixed OPD 的方案。

会议讨论中，5–20 steps 被认为可能过短；M0 应至少预留约 40–50 steps 才开始判断趋势，并统一使用“100 steps 内最佳表现”进行首轮排序。

### Stage B：正式 evolving

将 Stage A 中保留下来的原子算子及其组合规则交给 Codex 迭代搜索。每轮：

1. 读取当前 loss、实验状态和共享 memory；
2. 提出若干满足 DSL 约束的新 loss；
3. 并行运行相同预算的短程训练；
4. 写回指标、失败信息、代码/config hash 与 lineage；
5. 基于新证据继续保留、组合、修正或淘汰候选。

前中后期动态切换 loss 暂不作为 M0 必需项。先回答“从同一 \(\theta_0\) 出发，能否自动找到优于 fixed OPD 的 loss”；只有固定 loss 搜索得到稳定信号后，再讨论 checkpoint-conditioned switching。

## 4. M0 公平比较协议

所有候选 loss 从同一个完整初始状态 \(\theta_0\) 开始，并固定：

- student、teacher 和 tokenizer；
- inner-training data 与 data order；
- optimizer、learning-rate schedule、batch size 和 token budget；
- rollout / decoding 配置；
- evaluation questions、decoding seeds 与 checkpoint interval；
- codebase 与训练基础设施版本。

每个候选至少记录：

- 100 steps 内各 checkpoint 的 search-validation trajectory；
- best-within-100 与对应 checkpoint；
- fixed-step 指标，用于避免只报告偶然尖峰；
- training loss、gradient norm、clipping、NaN / Inf、crash；
- token / trajectory weight 分布与 effective sample size；
- wall-clock、显存与实际处理 token 数。

`best-within-100` 只能用于早期搜索。进入正式确认的候选必须重新使用预注册 checkpoint rule、更多 seeds 和独立 locked test，不能把参与搜索的 validation set 当成最终测试集。

## 5. 并行搜索与共享实验记忆

多 agent 并行的目的，是提高 loss proposal 与实验执行吞吐量，而不是研究 agent 通信机制本身。工程上需要：

- 一个 coordinator 分配候选和控制并发；
- 多个 worker agent 独立提出、实现或运行候选；
- 一个通过 GitHub 同步的 append-oriented memory；
- 对相同候选、数据与 GPU 资源做去重和锁定；
- 每个实验使用稳定的 run ID 与 parent IDs。

建议每条 memory entry 至少包含：

```json
{
  "run_id": "stable-id",
  "parent_ids": ["..."],
  "proposal_summary": "...",
  "loss_formula": "...",
  "code_commit": "...",
  "config_hash": "...",
  "data_manifest": "...",
  "seed": 0,
  "budget_steps": 100,
  "metrics": {},
  "diagnostics": {},
  "status": "completed|failed|invalid",
  "failure_reason": null,
  "interpretation": "...",
  "next_suggestion": "..."
}
```

搜索过程允许包含随机性，不要求不同机器得到完全相同的 proposal 顺序；但最终选出的 loss、代码、配置和确认实验必须可复现。完整日志还应足以审计“优质方案如何从哪些父方案和实验反馈演化而来”。

## 6. 研究产出与分析重点

该方向预期同时产生两类贡献：

1. **方法贡献**：一套可复用、受约束、可并行的 LLM-driven loss evolving 方法；
2. **结果贡献**：一个可直接复现和使用、优于 fixed OPD 的具体 loss function。

在最小闭环跑通后，再分析搜索过程与中间产物：

- 哪些原子算子被反复保留或淘汰；
- 哪些 state / diagnostic 触发了有效修改；
- 好方案是否共享可提炼的数学结构；
- 短程 proxy 与完整训练收益是否一致；
- 改进来自 divergence、filter、weight、normalization 还是 schedule。

数学解释与理论分析是后续提升研究价值的重要部分，但不阻塞 M0 工程闭环。

## 7. 当前最小下一步

1. 冻结第一版 loss DSL、合法原子算子和两层组合规则；
2. 定义 Codex proposal 输入/输出 schema 与静态校验器；
3. 从相同 \(\theta_0\) 跑 fixed OPD 和 3–5 个初始候选；
4. 验证 40–50 steps 是否开始出现可区分趋势，并完成 100-step 排序；
5. 统计单次 run 的 GPU 时间、显存和失败率，再决定并发规模；
6. 建立 GitHub 共享 memory，并先跑通单 coordinator + 少量 workers 的最小闭环。

## 8. 尚未解决的问题

- 100-step proxy 与完整训练最终收益的相关性；
- 第一版原子集合与两层组合语法的具体边界；
- Codex 每轮 proposal 数、保留率与探索/利用策略；
- 多 agent 的资源调度、重复 proposal 检测与失败恢复；
- 正式确认所需的训练 seeds、locked test 与算力预算；
- 如何将最终 loss 的有效结构转化为数学解释或可验证假设。
