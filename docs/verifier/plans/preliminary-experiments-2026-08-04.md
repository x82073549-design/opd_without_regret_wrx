# OPD Loss Evolving 前期实验计划

- 初版日期：2026-08-04
- 当前版本：2026-08-05 会议后执行版
- 状态：历史方案；当前 Baseline 实验以 [State 0 选择与复现实验计划](./state-0-selection.md) 为准

本版吸收 [2026-08-05 Loss Evolving 更新](../meetings/2026-08-05-loss-evolving.md) 的决定，替代此前“自由生成与组件组合二选一”的设计。本文保留用于追踪早期训练步数、early stopping 和 action 设计，不再作为当前 State 0 的执行入口。

## 1. 目标与范围

当前并行完成两个任务：

1. **Liujiang：确定候选 loss 的短训练步数 $H$**，使不同 loss 能以可接受成本进行初步比较；
2. **Ruxin：建立由 loss memory 支持的自由 loss action**，并确认 evolving 链路确实产生、加载和评估了新的 loss 代码。

已确定的边界：

- memory 提供论文 loss、已有实现和历史实验，不构成封闭 action space；
- action 在固定接口内自由生成 loss program，M0 优先宏观结构变化；
- M0 先搜索固定 loss，不研究训练中动态切换 loss；
- early stopping 先做离线回放，不在未经验证时终止正式实验；
- Validation 用于搜索，Locked Test 在方法冻结前不可访问。

## 2. 共同实验协议

### 2.1 起始状态与训练单位

所有方法必须使用同一个起始状态。起始方式二选一：

1. 完整 trainer checkpoint：使用 `RESUME_MODE=resume_path` 和固定 `RESUME_FROM_PATH`，恢复 model、optimizer、scheduler、global step 及框架能够恢复的 dataloader/RNG state；
2. model checkpoint：使用 `RESUME_MODE=disable`，统一重新初始化 optimizer/scheduler，并记录 `fresh_optimizer=true`。

禁止混用两种方式，禁止 `RESUME_MODE=auto`。

$H$ 表示从起始 checkpoint 新增的 optimizer steps：

\[
H=\text{final global step}-\text{root global step}.
\]

一条 training run 连续训练到 $H=100$，在 20/40/60/80/100 保存 checkpoint；不得为五个步数分别启动五次训练。当前 pilot 使用 batch size 64，后续修改必须建立新配置版本。

### 2.2 固定条件

候选与 Fixed OPD 必须使用相同的：

- student、teacher、tokenizer 和起始状态；
- optimizer、scheduler、learning rate 和 gradient clipping；
- Training data、数据顺序、batch/token budget；
- rollout 数量、temperature、top-p 和最大长度；
- mixed precision、分布式配置和 checkpoint 频率；
- Validation manifest、evaluation seeds 和 grading code；
- 代码 commit 和基础环境。

正式运行前保存完整 resolved config、启动命令、Git commit/patch hash、Python/PyTorch/verl/vLLM/CUDA 版本和 GPU 信息。

### 2.3 数据隔离

- Training：只用于参数更新；
- Validation：用于候选选择和 Codex 分析；
- Locked Test：方法完全冻结后只运行一次，不向 Codex 提供内容或结果。

正式实验前冻结：

- `validation_manifest.json`；
- `test_manifest.json`；
- `train_validation_dedup_report.json`。

Validation manifest 至少记录文件 hash、题目 ID/顺序、样本数、用途、prompt template hash 和 evaluation seed list。Test manifest 在搜索阶段只暴露 dataset-level hash、样本数和访问策略。

当前示例暂列 AIME24 和 Minerva，但必须通过真实 parquet 去重才能正式使用。AIME25、AMC23 或其他数据也不能未经检查直接加入。

```bash
python3 scripts/val/check_data_overlap.py \
  --train datasets/dapo-math-17k.parquet \
  --validation-manifest docs/verifier/validation_manifest.json \
  --output docs/verifier/train_validation_dedup_report.json
```

### 2.4 随机种子

training seed 与 evaluation seed 分开管理。

正式 training seeds：

```text
[0, 1]
```

training seed 至少控制框架允许控制的 Python、NumPy、PyTorch CPU/CUDA、rollout engine 和数据顺序。运行前验证：同方法同 seed 可复现、不同 seed 轨迹确实不同、resolved config 中能看到生效值。

evaluation seeds 预注册为：

```text
E0 = [0, 1, 2, 3]
E1 = [4, 5, 6, 7]
E  = E0 + E1
```

每个 request seed 由 stable question ID 和 evaluation seed 确定并真实传给 vLLM。主结果使用 `mean@8`，`mean@4(E0)` 和 `mean@4(E1)` 用于诊断 evaluation noise。正式评测前必须完成同模型、同题目、同 seed 独立重跑检查。

### 2.5 运行隔离与结果完整性

每次评测使用：

```text
{method}_trainseed{training_seed}_root{root_step}_delta{delta_steps}
```

作为唯一运行标识。merge、generation 和 grading 目录必须隔离，并核对源 checkpoint。每个任务必须产生完整且唯一的：

```text
(task, question_id, evaluation_seed)
```

记录数必须等于 `questions × evaluation seeds`。缺失、重复、checkpoint 来源不一致或复用旧输出时，评测无效。

### 2.6 无效实验

以下结果不进入主比较：

- loss/gradient 出现 NaN 或 Inf；
- checkpoint、输出或逐题结果不完整；
- 起始状态、训练步数、有效 token 数或数据顺序不一致；
- 未预注册地修改 optimizer、数据、rollout 或评价方法；
- evaluation seed 未真实控制生成；
- loss 读取 Validation/Locked Test 信息；
- 候选违反声明的 tensor、support、weight、normalization 或 fallback 约束。

失败记录必须保留日志、原因和是否需要重跑。

## 3. 任务一：确定短训练步数

负责人：Liujiang

### 3.1 方法与矩阵

第一轮冻结三个已有方法：

1. `fixed_opd`：fixed top-k OPD；
2. `sampled_token_gate_opd`；
3. `overlap_prune_opd`：只指 `prune_opd`。

配置文件：

- `configs/presearch/fixed_opd.env`；
- `configs/presearch/sampled_token_gate_opd.env`；
- `configs/presearch/overlap_prune_opd.env`。

标准矩阵：

```text
3 methods × 2 training seeds = 6 training runs
6 runs × 5 horizons = 30 checkpoint evaluations
H ∈ {20, 40, 60, 80, 100}
```

2-GPU 运行必须设置 `SAVE_FREQ=20`；从非零 root 恢复时，绝对终止步数为 `root_step + 100`。

资源不足时按顺序完成：

1. 三个方法的 seed 0；
2. Fixed OPD 的 seed 1；
3. seed 0 最优的非 baseline 方法补 seed 1。

未完成 matched second seed 的方法只能形成初步结论。

### 3.2 正式矩阵前检查

必须先通过：

1. 三个配置的静态检查；
2. 小 batch forward/backward；
3. 各方法 1-step smoke test；
4. training/evaluation seed 审计；
5. checkpoint save → merge → generate → grade → paired report 全链路；
6. 不同 method/seed/H 不会串 merge 或结果目录；
7. 逐题结果可以重新聚合出相同总分。

### 3.3 评价指标

对方法 $L$、training seed $s$ 和步数 $H$：

\[
\Delta_{H,s}(L)
=J_{\mathrm{val}}(\theta_{H,s}^{L})
-J_{\mathrm{val}}(\theta_{H,s}^{\mathrm{Fixed\ OPD}}).
\]

候选与 baseline 必须 matched training seed。逐题按 `(question_id, evaluation_seed)` 配对，并以 question 为单位做 paired bootstrap。

需要报告：

- `mean@8`、`mean@4(E0)`、`mean@4(E1)`；
- 相对 matched Fixed OPD 和 top-vs-runner-up 的差值；
- question-level paired bootstrap 95% CI；
- training-seed、evaluation-seed、horizon 和 task-level variation；
- training loss、pre-clip gradient norm 和 clipping rate；
- response length、truncation、有效 tokens；
- NaN/Inf、失败、wall-clock、显存和 GPU-hour。

```bash
python3 scripts/analysis/paired_eval_report.py \
  --candidate <candidate>/detailed_results.jsonl \
  --baseline <baseline>/detailed_results.jsonl \
  --output <candidate>/paired_report.json
```

### 3.4 训练步数选择

明天下午允许给出“推荐的初步筛选步数”。选择满足以下条件的最小 $H$：

1. winner 与 step 100 相同；
2. 两个 training seeds 下 top-vs-runner-up 差值同号；
3. E0、E1 的 winner 和关键差值方向一致；
4. paired CI 已报告，结论不由单个异常题目驱动；
5. 训练、checkpoint 和输出均有效。

如果并列、tie-breaking 改变 winner，或 step 100 仍不稳定，则输出“当前无法确定”，并指出应增加 seeds、题目、对照强度还是训练到 200/300 steps。

“稳定最小步数”还需要：

- 完整方法排名与 step 100 一致；
- top-vs-runner-up 95% CI 不跨 0；
- leave-one-question-out 后 winner 不变；
- 至少第三个 training seed；
- 后续 300-step 确认早期排序与长期结果一致。

step 100 只是短程 proxy，不代表最终训练效果。

### 3.5 Early stopping 预实验

先收集完整 100-step 轨迹，再离线回放两类策略：

1. 规则策略：数值失败立即停止；连续多个 checkpoint 显著低于 matched baseline 且超过噪声范围时停止；
2. Codex 策略：输入中间 Validation、训练诊断和 uncertainty，输出 `continue/stop`、confidence 和理由。

confidence collapse、梯度异常、weight/ESS 塌缩和长期无改进只是候选信号。回放报告 GPU time 节省、误停率、是否误停最终 winner 和跨 seed 稳定性。审计通过前只记录建议，不实际停止正式 run；Locked Test 不参与停止决策。

### 3.6 Liujiang 交付与验收

提交：

- 完整运行矩阵、config、seed 和有效/无效状态；
- 20/40/60/80/100 结果表、曲线和逐题文件；
- paired statistics、training/evaluation noise；
- 推荐初步 $H$ 或“当前无法确定”；
- early-stopping 回放；
- 单 run 和总计算成本。

当前状态：夜间实验曾因网络/代理故障中断，尚无可验收结论。恢复后需确保 checkpoint 可续跑且代码、配置及时提交。

## 4. 任务二：建立 Loss Memory 与自由 Action

负责人：Ruxin

### 4.1 当前设计

2026-08-05 已决定不再比较“自由生成 vs 组件组合”。当前闭环为：

```text
frozen memory snapshot
→ parent selection
→ sub-action / iteration direction
→ free loss proposal
→ code validation and write
→ forward/backward and smoke test
→ short training
→ result appended to next memory version
```

- **memory**：提供论文 loss、已有代码、正负实验和失败案例；
- **主 action**：在固定接口内生成完整 loss program；
- **副 action**：决定本轮迭代方向；
- **parent selection**：决定从哪个已有 loss 继续演化。

副 action 不能把主 action 限制为旧组件排列组合。

### 4.2 Loss memory

第一版 memory 包含：

- Fixed OPD、Sampled-token gate OPD、Overlap-based OPD；
- FKL、RKL/OPD、mixture、JSD 等论文或现有实现；
- `0.5×/2.0× OPD`、shuffled/sign-flipped 等 diagnostic；
- 历史成功、失败、无差异和无效实验。

每条记录至少包含：

```text
memory_id, parent_ids, source, formula, code/config hash,
tensor/support, parameters, data manifest, seed, budget_steps,
metrics, uncertainty, diagnostics, status, failure_reason,
interpretation, next_suggestion
```

memory 必须版本化。同一批并行候选读取相同 snapshot；该批结果只追加到下一版本。只写 Training/Validation 结果，Locked Test 永不进入搜索 memory。

### 4.3 自由 Action 的范围

M0 优先允许 Codex 改变：

- divergence 和计算 support；
- token/trajectory weighting；
- gating、filtering 和 mask；
- normalization、clipping 和 fallback；
- 多个 loss 结构的组合。

仅修改常数或少量超参数不计为主要 evolve 结果。参数范围可由人或 Codex 预定义，具体搜索优先使用 grid、random 或 Bayesian optimization。

Codex 只能修改受限 loss 函数，不能修改 model、optimizer、数据、rollout、checkpoint 或 evaluation。候选代码禁止文件/网络/子进程访问、动态 import、`eval/exec`、修改 RNG/参数/输入 tensor 或写外部状态。

执行前冻结 `tensor_contract.json` 和输出 schema，明确每个输入的 shape、dtype、device、mask、support、producer、可用性和 stop-gradient。候选必须声明参数范围、loss scale、normalization 和 empty-batch fallback。

### 4.4 Evolving 链路审计

当前发现连续两轮 Python loss 文件基本一致，因此该链路尚未证明有效。每个 proposal 必须保存：

- candidate/parent IDs；
- 主 action、副 action 和 memory version；
- prompt、generation seed；
- 数学定义和理由；
- Python diff、候选文件 hash 和 config hash；
- 训练时实际加载的 loss module hash。

除 identity test 外，新候选必须与 parent 存在可解释的宏观结构差异。如果连续两轮文件相同，或文件改变但运行时 hash 未改变，立即停止并检查 memory reading、parent selection、副 action、代码写回和 module loading。

候选依次通过：

1. schema、公式和 support 检查；
2. AST/import/I/O/副作用安全检查；
3. tensor contract、mask 和 stop-gradient 检查；
4. synthetic boundary tests；
5. recorded real batch forward/backward；
6. finite loss/gradient、normalization 和 fallback；
7. 1–5 step smoke test。

Fixed OPD identity candidate 必须与 reference 在 loss、有效 token、mask、fallback 和 gradient 上一致；建议阈值为 fp32 loss `rtol≤1e-5, atol≤1e-6`，bf16 loss `rtol≤1e-3, atol≤1e-4`，gradient cosine `≥0.999`。

### 4.5 M0 通过标准

同时满足：

1. identity candidate 通过；
2. validator 能阻止违规候选执行；
3. 至少两轮 proposal 完整记录 memory、parent、副 action、diff 和 hash；
4. 至少一个非 identity 候选产生宏观结构变化并通过 forward/backward；
5. 至少一个新候选通过 smoke test；
6. 运行时加载 hash 与候选文件一致；
7. 成功和失败结果都能写回下一版 memory。

### 4.6 Ruxin 交付与验收

在独立 branch 提交：

- 收集的论文/现有 loss 和来源；
- versioned memory、schema 和 hash；
- 主 action、副 action、parent selection 和 memory reading 逻辑；
- `tensor_contract.json`、输出 schema 和 validator；
- 至少两轮 proposal、Python diff/hash、lineage 和 runtime-load 记录；
- identity、forward/backward、boundary 和 smoke-test 结果；
- 第一批可进入短训练的宏观结构候选；
- 失败候选和失败原因。

当前首要验收不是 reward，而是确认 evolving 闭环真实生效。若失败，必须定位到 memory、parent/sub-action、代码写回、runtime load、tensor contract 或 validator。

当前状态：已向模型提供轻量实验记录、验证摘要和 loss 源码，处于链路调试阶段；连续两轮代码基本一致，因此尚不能认定 evolve 已生效。

## 5. 协作、优先级与最终验收

### 5.1 分支与共享

- main 保留稳定可运行底座；
- Ruxin branch 集成 memory、参考 loss 和 evolving；
- Liujiang branch 保存 horizon 和 early-stopping 实验；
- 多机通过 Git 同步 commit、config、memory 和结果摘要；
- Fixed OPD baseline 按相同 seed 共享，不重复运行。

### 5.2 资源优先级

1. seed、目录、manifest 和逐题结果正确；
2. Fixed OPD 两个 training seeds；
3. 三个方法 seed 0；
4. 初步最优候选补 seed 1；
5. 其他候选补 seed；
6. early-stopping 回放和负对照；
7. 200/300-step 长期确认。

不能为了按时给出结果而降低无效实验标准或使用 Locked Test。

### 5.3 联合验收

会议需要回答：

1. 是否得到可用的初步筛选步数，或明确知道缺少什么证据；
2. loss memory 是否可审计、可版本化；
3. 自由 action 是否真实生成并加载了新的宏观 loss；
4. 是否有至少一个新候选通过 smoke test；
5. 是否具备启动第一轮正式 Codex OPD loss search 的条件；
6. early stopping 目前只能记录建议，还是已经通过回放可以实际启用。

如果任一核心前提未通过，结论应为 `not ready`，并给出最小修复实验，而不是启动大规模搜索。

### 5.4 尚未冻结

- parent node 的选择规则；
- 副 action 的集合和更新方式；
- 每轮候选数、保留率和探索/利用策略；
- early-stopping 阈值及 Codex confidence 校准；
- memory 变大后的检索、压缩或微调方案；
- 固定 loss 搜索升级为阶段动态 loss 的条件。
