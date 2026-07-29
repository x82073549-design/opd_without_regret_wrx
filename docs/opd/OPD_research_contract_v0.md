# OPD 跨粒度统一信号：研究契约 v0

> 状态：讨论稿。用途是冻结研究问题、证据标准与第一阶段的 go/no-go 条件；不把探索性结果提前写成确认性结论。

## 1. 工作命题

在可验证推理、共享 tokenizer 的 student–teacher 设定中，OPD 监督的价值不仅取决于师生差异，还取决于 teacher 可靠性、师生兼容性，以及学生是否仍未掌握。若这些因素能够组成一个底层 token transfer-value，并在不针对更高粒度重新拟合的前提下向上聚合，则同一信号应能指导 token 加权、样本选择和 teacher 选择。

本项目优先验证以下较窄命题，不预先声称普遍适用于所有 OPD、开放域任务或跨 tokenizer 场景。

## 2. 核心 claim 与证据门槛

| ID | 拟验证 claim | 最低证据 | 不足以支撑该 claim 的结果 |
| --- | --- | --- | --- |
| C1 | 复合信号比单一 entropy、KL 或 overlap 更能预测 OPD 收益 | 独立 validation/test 上预测提升；完整模型优于各原始量及组件消融 | 同一数据上挑公式并报告相关性 |
| C2 | 同一信号在 token、sample、teacher 三个粒度上方向一致 | 冻结 token 公式；仅用预注册聚合得到 sample/teacher score；三层均出现正向优于随机/均匀和取反 | 三个粒度分别重新选特征或调方向 |
| C3 | 按该信号分配固定预算能够改善 OPD | 多 seed、等计算干预；主指标为 held-out accuracy/reward；报告效应量与区间 | 仅 reverse-KL 下降，或仅最佳 seed 提升 |
| C4 | 三粒度联合控制存在额外价值 | 单粒度、关键两两组合、三粒度联合的消融；联合方法优于最强单粒度与强基线 | 只比较 joint 与 vanilla OPD |
| C5 | 信号反映 compatibility × reliable novelty × non-mastery 机制 | 正确性分层、兼容性/可靠性分层及组件交互分析；小集梯度对齐作为外部诊断 | 只有最终 accuracy 提升或普通组件消融 |

## 3. 统一信号的定义边界

### 3.1 原子信号

先定义 token 级原子量：

\[
v_{i,t}^{(m)} = G_i \cdot R_{i,t}^{(m)} \cdot \Phi\!\left(C_{i,t}^{(m)},N_{i,t}^{(m)}\right)
\]

- \(G\)：non-mastery gate。首版使用 student rollout 是否错误；后续比较 pass-rate 软门控。
- \(R\)：teacher reliability。候选为 teacher confidence、teacher trajectory normalized log-prob，以及 teacher correctness 的 oracle 门控。
- \(C\)：compatibility。首要候选为 top-k overlap；不预设其对价值线性单调。
- \(N\)：novelty / reducible discrepancy。候选为逐 token KL、logprob-ratio、student confident-but-wrong 指示。
- \(\Phi\)：需要在探索集上比较的有限候选族：直接乘积、compatibility 阈值门控、band-pass compatibility。选定后冻结。

“统一”采用强定义：更高粒度不得引入新的价值特征或重新学习一套权重；只允许预注册的聚合和尺度归一化。

### 3.2 向上聚合

\[
V_{\text{sample}}(i,m)=\operatorname{Agg}_{t} v_{i,t}^{(m)},\qquad
V_{\text{teacher}}(m)=\operatorname{Agg}_{i\in\mathcal{P}}V_{\text{sample}}(i,m)
\]

聚合必须在确认实验前固定，并检查长度偏差：

- token→sample：比较 length-normalized mean 与 top-quantile mean；探索集确定一个主方案，另一个只作稳健性分析。
- sample→teacher：在同一固定 probe 集上求均值，并按任务/难度分层；不得让不同 teacher 使用不同 probe 样本。

## 4. 估计目标与数据划分

### 4.1 指标层级

- 主 estimand：固定预算训练后的 held-out accuracy、reward 或 pass@k 增益。
- 筛选代理：held-out reverse-KL 变化。
- 机制诊断：小规模集合上的 gradient alignment 或短程影响量。

reverse-KL 只有在独立条件上能够预测 accuracy 增益时，才可被称为有效代理；否则仅用于排除明显失败条件。

### 4.2 数据隔离

- exploration/probe：观察特征形状、构造 teacher score。
- validation：选择有限候选公式、聚合方式、阈值和权重强度。
- locked test：最终 accuracy/reward，仅在确认实验使用。

三个粒度共享同一 locked test 以保证可比性，但不能用 test 选择公式或超参数。

## 5. 实验路径与决策门

### Stage A：Token 特征形状探索

对应《本周实验计划 · OPD Token 级特征有效性验证》。该阶段当前测试 overlap、teacher confidence、student entropy、FiRe 组合的正向/取反权重，定位为**方向探索和 harness 验证**，尚不能单独验证 C1–C2 的统一复合信号。

必须同时记录：student correctness、teacher correctness（若可得）、KL、overlap、teacher/student entropy、trajectory log-prob、长度和位置。

Go 条件：

1. 至少一个特征在独立 validation 上呈现稳定方向，正向/取反曲线可区分；
2. 正向相对均匀的效应在共同 seed 上方向一致；
3. reverse-KL 的变化与至少一个下游 performance 检查不存在明显反向关系。

否则先检查 weighting harness、headroom、特征方向和代理指标，不进入样本级训练。

### Stage B：复合信号冻结与 Token 确认

在 exploration/validation 上比较原始量、二元组合、完整组合及去门控消融；冻结 \(v\)、聚合方式和主权重强度。随后在 locked test 上进行 token 正向/均匀/随机打乱/取反以及 TIP、FiRe 对照。

Go 条件：完整信号相对均匀存在可实现增益，且不弱于调参预算对齐的 token 强基线；完整信号优于至少主要单量和关键去门控消融。

### Stage C：Sample 零重拟合迁移

仅使用冻结的 \(v\) 聚合得到 sample score。比较等样本、等 token/计算预算的 high/random/low 子集，并在难度、长度、学生正确性箱内复核。

Go 条件：high > random > low 的方向在共同 seed 上稳定；若必须更换特征或反转方向才能成立，则 C2 失败，主张降级为“共享原则”而非“统一信号”。

### Stage D：Teacher 真值排序与零重拟合迁移

对每个 student × task × checkpoint 设定，实际短程训练所有候选 teacher 得到排序；用固定 probe 集和冻结聚合预测排序。抽取部分设定验证短程排序是否能代表完整训练排序。

Go 条件：top-1 命中率显著高于随机选择，排序相关稳定，端到端选择优于随机/取反；否则不进入 per-sample teacher routing。

### Stage E：联合端到端确认

比较 vanilla、三个单粒度、关键两两组合、三粒度联合及各粒度强基线。固定训练 token、优化步数，并额外报告 teacher/probe/verifier 成本。

C4 只有在联合相对最强单粒度仍有增益，且不是来自额外计算时成立。

## 6. 统计约定

- 训练比较使用相同随机种子配对，不选择“最佳 run”后再做显著性检验。
- pilot 的 R=3 只用于排错和估计方差；确认实验的 seed 数根据 pilot 方差与目标最小效应预先确定。
- 题目级 bootstrap 嵌套在 seed 内，或采用 seed/题目的层级 bootstrap；token 不作为独立复制单元。
- teacher 级的复制单元是 student × task × checkpoint，而不是候选 teacher 数量。
- 主假设与主比较预先固定；大量候选特征和强度的探索性结果不作为确认性 p 值。
- 除显著性外，报告均值、标准差、配对差值、置信区间和效应量。

## 7. 成本与可部署性

分别报告两个版本：

- Oracle/diagnostic：允许 student/teacher correctness 和昂贵梯度对齐，用于验证机制上限。
- Deployable：仅使用训练时可得或成本可控的代理量。

总成本需包含 student rollout、teacher forward、额外 teacher rollout、verifier、probe ranking、特征刷新和训练 token。只有 deployable 版本在完整计费后仍接近 vanilla OPD，才可声称低开销。

## 8. 当前待定项（按优先级）

1. 首版 scope：是否正式限定为可验证数学推理、同 tokenizer student–teacher。
2. compatibility 的作用形式：乘积、阈值门控还是 band-pass。
3. non-mastery 使用单 rollout correctness，还是组 pass-rate 软门控。
4. token→sample 的主聚合：length-normalized mean 还是 top-quantile mean。
5. 特征刷新频率：每 rollout、每若干训练步或静态 checkpoint。
6. reverse-KL 与 accuracy 的最低代理一致性要求。

在 1–5 固定之前，本周 Token 计划应被视为探索性实验；不据此冻结论文结论。
