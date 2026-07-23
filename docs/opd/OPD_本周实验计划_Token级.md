# 本周实验计划 · OPD Token 级方向探索与 Harness 验证

**范围**：只做 Token 级（最廉、fail-fast）。使用单一 student–teacher 对、单一有 headroom 的 checkpoint，以及可验证数学推理任务。复合统一信号、样本级和 teacher 级留到后续阶段。

**本周定位**：探索单一特征的有效方向，验证 token weighting harness 是否严格控制总权重和计算预算。该阶段本身不用于声称“三粒度统一信号”成立。

---

## 0. 本周要回答的问题

1. 在固定总 token 权重和训练预算后，重新分配 token 权重是否能稳定改变 OPD 收益？
2. overlap、teacher confidence、student entropy 与 FiRe 组合分别指向哪些 token？它们的有效方向是否稳定？
3. reverse-KL 的筛选方向是否至少不与下游 accuracy 方向冲突？
4. 学生 rollout 正确与错误时，特征方向是否发生变化？

本周不回答：哪个复合公式最优、sample/teacher 层是否成立、三个粒度联合是否优于强基线。

---

## 1. 数据与实验单元

### 1.1 数据隔离

- **exploration/probe**：观察特征分布、排查实现、画分层曲线。
- **validation**：比较特征方向与 α 强度；筛选 1–2 个候选条件。
- **locked test**：仅供后续 accuracy 确认使用，本周探索期间不据此选择特征或 α。

三个集合按问题划分，来自同一任务分布，禁止同题或近重复题跨集合。

### 1.2 复制与配对

- 所有条件从同一 checkpoint 出发，并使用同一组训练随机种子。
- 条件比较采用共同 seed 配对；不选择最佳 run 后再做显著性检验。
- R=3 仅作为 pilot，用于排错、判断方向和估计 run 间方差，不把其 p 值作为确认性证据。
- accuracy 确认的 seed 数由 pilot 方差和最小关心效应决定；若暂无法做功效估计，R=5 只作为最低起点。

---

## 2. 四个探索特征与必要埋点

### 2.1 本周干预特征

| 特征 | 轴 / 角色 | 本周是否预设方向 |
| --- | --- | --- |
| overlap ratio | 师生兼容性 | 否；可能是门控或非单调关系 |
| teacher confidence | teacher 局部可靠性代理 | 否；高置信也可能自信但错误 |
| student entropy | 学生局部不确定性 | 否；不能单独表示是否掌握 |
| FiRe \(c_t^T\cdot c_t^S\) | teacher × student 组合基线 | 按原方法方向，同时保留取反对照 |

### 2.2 每条 rollout 必须同步记录

- student correctness、teacher correctness（若本任务可直接验证）；
- student entropy、teacher entropy/confidence；
- 逐 token KL、logprob-ratio reward；
- top-k overlap、student mass、teacher mass；
- teacher trajectory normalized log-prob；
- token position、response length、截断标记与重复率；
- 原始特征值、标准化值、最终权重和有效 α 缩放比例。

这些量用于分层和排查混淆，本周不因观察到相关性就直接组成最终统一信号。

---

## 3. Token 统一加权 Harness

目标：所有特征通过同一变换；每条序列的平均权重严格为 1；正常与取反严格中心对称；最终权重非负。序列内归一化用于隔离纯 token 分配效应，避免特征偷偷引入样本级加权。

对每个特征的逐 token 值 \(s_t\)：

1. 对有效 response token 做序列内标准化：

   \[
   z_t=\frac{s_t-\bar{s}}{\max(\operatorname{std}(s),\epsilon)}
   \]

   若序列内方差低于 ε，整条序列使用均匀权重并记录 `degenerate_feature=true`。

2. 构造并中心化有界扰动：

   \[
   d_t=\tanh(\alpha z_t/2),\qquad
   \tilde d_t=d_t-\bar d
   \]

3. 为避免直接 clipping 破坏均值和正反对称，对中心化扰动做整序列等比例安全缩放：

   \[
   \lambda=\min\left(1,\frac{1-\epsilon_w}{\max(\max_t|\tilde d_t|,\epsilon)}\right),\qquad
   d_t^{\mathrm{safe}}=\lambda\tilde d_t
   \]

4. 得到正常与取反权重：

   \[
   w_t^{+}=1+d_t^{\mathrm{safe}},\qquad
   w_t^{-}=1-d_t^{\mathrm{safe}}=2-w_t^{+}
   \]

由此保证：

- 每条序列 \(\operatorname{mean}(w^+)=\operatorname{mean}(w^-)=1\)；
- \(w_t^++w_t^-=2\)；
- \(w_t^+,w_t^-\geq\epsilon_w\)；
- 不通过下限截断改变总权重或破坏对称性。

### 3.1 四个实验臂

- **正常**：使用 \(w^+\)。
- **取反**：使用 \(w^-\)。
- **随机打乱**：在每条 response 内随机置换 \(w^+\)，保持该序列权重分布、均值和计算量不变；置换 RNG 单独记录。
- **均匀**：所有有效 token 权重为 1，即 α=0。

随机打乱用于区分“权重分布形状有效”与“特征–token 对应关系有效”。

### 3.2 α 扫描

- pilot 扫描 α ∈ {0.5, 1, 2, 4}，记录原始 α 与安全缩放后的有效强度。
- validation 上以相对均匀条件的配对效应曲线和预定义 AUC 做探索性排序，不对每个 α 分别挑最小 p 值。
- accuracy 确认前冻结一个主 α；强度 AUC 作为稳健性结果，不用 locked test 选择 α。
- overlap 等方向不明确的特征同时保留正向和取反；方向只能在 validation 上确定一次。

---

## 4. Harness 单元与小步验收

正式 pilot 前先运行极小 batch / 50–100 update 的 smoke test。必须自动检查：

1. α=0 与原始均匀 OPD 的 loss、梯度和更新在数值容差内一致；
2. 每条序列四个实验臂的有效 token 数相同；
3. 正常、取反、随机条件的平均权重均为 1；
4. 正常与取反逐 token 权重之和为 2；
5. 权重均为有限非负值，padding/prompt token 不参与标准化和 loss；
6. 随机打乱保持权重多重集合不变，且跨序列汇总后与原特征的 rank correlation 接近 0；
7. 各条件训练 token、optimizer steps、batch size、学习率和梯度累积完全一致；
8. `degenerate_feature` 比例和安全缩放触发比例被汇总报告。

任一项失败则停止正式训练，先修复 harness。

---

## 5. 因变量与分析

| 阶段 | 数据 | 因变量 | 种子 | 分析定位 |
| --- | --- | --- | --- | --- |
| Smoke test | exploration | loss/gradient 一致性、权重约束 | 1 | 实现验收 |
| 方向 pilot | validation | held-out reverse-KL | R=3，共同 seed | 探索方向、估计方差 |
| 下游 sanity | validation 的独立评测题 | accuracy/reward | pilot checkpoints | 检查 reverse-KL 是否明显误导 |
| 确认（仅胜出 1–2 条件） | locked test | accuracy avg@k / reward | 预先确定，R=5 为最低起点 | 确认性证据 |

### 5.1 主比较

- 可实现增益：正常 vs 均匀，以及正常 vs 随机打乱。
- 方向 sanity：正常 vs 取反；该差值不作为实际收益上界。
- 理想探索序：`取反 < 均匀/随机 < 正常`，但 overlap 的原始正负方向由 validation 决定后再重命名为正常方向。

### 5.2 分层与统计

- 学生 rollout 正确/错误分别报告，不只报告混合均值。
- 同时按长度、位置、难度和 teacher correctness 做描述性分层，检查方向是否由混淆变量主导。
- 训练比较使用 seed 内配对差值和效应量；R=3 不依赖 Mann–Whitney 或双样本 t 检验得出强结论。
- 题目级 uncertainty 使用 seed 内配对 bootstrap；最终汇总采用 seed/题目的层级 bootstrap 或等价混合效应分析。
- 不对“最佳 run”单独检验，不把 token 当作独立统计复制单元。

---

## 6. 本周 Go / No-Go 条件

### Go：进入复合信号冻结阶段

同时满足：

1. harness 的全部数值约束通过；
2. 至少一个特征在 validation 的共同 seed 上相对均匀呈稳定方向；
3. 该特征的正常条件优于随机打乱，或至少给出一致的正向效应迹象；
4. reverse-KL 改善没有伴随明显的 accuracy/reward 反向变化；
5. 有效方向不是完全由 student-correct、长度或位置分布差异解释。

通过后，下一阶段才构造并冻结 compatibility × reliable novelty × non-mastery 的复合信号。

### No-Go：暂停扩展粒度

任一情况触发：

- α=0 无法复现均匀 OPD；
- 权重归一化或随机打乱控制不成立；
- 所有特征方向随 seed 或 α 无规律反转；
- reverse-KL 与下游 performance 系统性反向；
- checkpoint 缺乏 headroom，所有条件均被性能上限压缩。

No-Go 时优先诊断 harness、checkpoint headroom、指标有效性和特征非单调形状，不直接进入样本或 teacher 级。

---

## 7. 本周产出物

1. 每个特征 × α × 实验臂的配对结果表；
2. reverse-KL 与 accuracy/reward 的方向一致性表；
3. student correct/error 分层曲线；
4. 权重均值、最小值、安全缩放率和退化序列率的 harness 报告；
5. 下一阶段保留的 1–2 个原始信号及淘汰理由；
6. 明确的 Go / No-Go 结论。

---
