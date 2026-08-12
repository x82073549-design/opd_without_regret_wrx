# OPD Loss Search 全流程与逐轮实验详解

## 1. 执行结果概览

本次搜索配置计划最多运行 5 轮，即 `round_000`–`round_004`；实际完成了 4 轮，即 `round_000`–`round_003`。`round_004` 没有生成 Python loss，也没有 smoke、probe 或 confirmation 记录。

最终全局 champion 是第一轮的：

```text
r0_reliable_disagreement_router
```

其官方 confirmation score 相对 Fixed OPD baseline 提升：

```text
+0.0072916667 = +0.729 个百分点
```

各轮结果如下：

| 人类计数 | 实际 round | 进入 confirmation 的候选 | Probe delta | Confirmation delta | 是否替换全局 champion |
|---|---:|---|---:|---:|---|
| 第 1 轮 | 000 | Reliable Disagreement Router | +1.667 pp | **+0.729 pp** | 是 |
| 第 2 轮 | 001 | First Error Onset Credit | -4.167 pp | -0.729 pp | 否 |
| 第 3 轮 | 002 | Fixed Pareto Consensus Bridge | -5.000 pp | +0.417 pp | 否 |
| 第 4 轮 | 003 | Outcome-safe Reliable Switch | -2.500 pp | -0.208 pp | 否 |
| 第 5 轮 | 004 | 未运行 | — | — | — |

这里的 `pp` 表示百分点。

---

## 2. 整个搜索系统如何运转

搜索的真实执行链如下：

```text
上一轮结果
   ↓
压缩为新一轮 context
   ↓
盲目 brainstorm 6 个结构性假设
   ↓
检索 4 个论文 soft priors
   ↓
Codex 从中生成 2 个 Python loss
   ├─ evidence_low_risk：有上一轮证据或论文依据
   └─ free_exploration：结构上更激进的新方向
   ↓
静态/合成数据验证
   ↓
真实 recorded batch CPU replay
   ↓
真实双 GPU、1-step smoke test
   ↓
10-step probe + AIME25 pass@4
   ↓
选择该轮 winner
   ↓
100-step confirmation + 2 training seeds + AIME25 n=16
   ↓
更新全局 champion 和下一轮 context
```

总控代码位于：

- `opd_loss_search/search.py`
- `opd_loss_search/orchestration/coordinator.py`

### 2.1 新一轮 context 如何构造

每轮结束后，系统将结果追加到：

```text
opd_loss_search/search_runs/codex_opd_v2/history/experience.jsonl
```

下一轮读取最近 8 轮经验，并压缩为：

- 上一轮 winner 和分数；
- 每个候选的 probe/confirmation 状态；
- paired delta；
- 部分 loss、gradient、entropy、ESS、scale、fallback 等训练指标；
- winner 使用过的 mechanism IDs；
- 当前 champion；
- 简化后的症状描述，例如 `positive` 或 `no validation gain`。

这一设计让下一轮 Python loss 不只是根据论文提出，也会受到前面实验结果影响。

不过目前的 context 压缩有一个重要缺陷：它保留 `SMOKE_FAILED` 状态，但没有把详细的 `failure_reason` 传给下一轮 proposal。下一轮可能知道某条路线 smoke 失败，却不知道原因究竟是 loss 数学问题，还是 CPU/CUDA device-placement bug。这也是后续再次提出 union-support loss 的原因之一。

### 2.2 六个假设如何变成两个 Python loss

Brainstorm 阶段收到：

- 当前 context；
- tensor contract；
- “恰好提出 6 个结构不同的假设”的要求；
- 禁止只调已有 scalar hyperparameter；
- 不允许在这一阶段照抄论文公式。

Proposal 阶段才会收到：

- 6 个 blind hypotheses；
- 当前 context；
- 检索到的 4 个论文 soft priors；
- Python action 接口和安全限制。

Proposal 必须直接返回两个候选：

1. 一个 `evidence_low_risk`；
2. 一个 `free_exploration`。

系统没有额外的独立 ranker 对 6 个假设打分。实际由 proposal 模型直接选择、组合或改写假设。例如第三轮的 `r2_fixed_pareto_consensus_bridge` 不是 brainstorm 中的原始假设名，而是 proposal 阶段结合上一轮结果和 local-support prior 合成出来的。

Prompt 模板位于：

```text
opd_loss_search/prompts/custom_python_prompt.md
```

### 2.3 Proposal repair

Codex 返回候选后，系统会尝试 materialize 并验证。如果验证失败，会把错误附加到 prompt，要求 Codex 只修复 contract 或 safety violation。

最多允许 3 次 proposal 尝试：

```text
proposal
proposal_repair_1
proposal_repair_2
```

候选路径在 repair 间保持稳定，因此 controller 会清空：

```python
clear_candidate_cache()
clear_custom_function_cache()
```

以避免旧的 candidate JSON 或旧 Python function 被路径缓存复用。

---

## 3. Python loss 接口与统一归一化

每个候选只能定义：

```python
def compute_token_objective(inputs, params, ops):
    return {
        "base_advantages": ...,      # [B, T, K]
        "raw_token_weights": ...,    # [B, T]
        "active_mask": ...,          # [B, T]
        "metrics": {...},
    }
```

所有输入都会先 detach。候选只能使用受限的 `SafeOps`，不能进行 import、I/O、随机数调用、反射、模型访问、optimizer 访问或验证集访问。

候选返回结果后，runtime 还会统一执行三层处理。

### 3.1 Token weight 投影

`raw_token_weights` 会按照 candidate normalization 投影，一般包括：

- clip 到 `[0, 2]`；
- 将 active token 的平均 weight 调整到 1；
- 检查 effective sample size，即 ESS；
- 当 weight 异常或 ESS 过低时，完整 fallback 到 Fixed OPD。

### 3.2 Advantage 加权

```text
weighted_advantages
= base_advantages × projected_token_weights
```

### 3.3 与 Fixed OPD 进行 magnitude matching

最终 advantage 的平均绝对值会缩放到与 Fixed OPD reference 相当：

\[
A_{\text{final}}
=
\operatorname{scale\_match}
\left(
A_{\text{candidate}}\times w_{\text{candidate}}
\right).
\]

因此 smoke 中的 `opd_action/scale` 很重要。例如第四轮 outcome switch 将约 90% token neutralize 后，剩余 action 被放大约 11.86 倍，才能匹配 Fixed OPD 的整体幅度。

---

## 4. Smoke test 到底测试什么

系统有三层验证，必须区分开来。

### 4.1 第一层：静态与合成 batch 验证

候选刚生成时会检查：

- JSON schema；
- tensor contract；
- Python AST 安全性；
- FP32 合成 batch；
- BF16 合成 batch；
- forward/backward 是否 finite；
- tensor shape 是否正确；
- empty response mask 是否严格输出 0。

合成 batch 人工包含：

- `trajectory_correctness`；
- `format_valid`；
- history tensors；
- region tensors。

因此“合成验证通过”并不代表在线训练 batch 一定包含这些字段。

### 4.2 第二层：真实 recorded batch replay

Baseline 的真实 smoke 会抓取一份 action batch。所有候选会先在这份 batch 上执行 forward/backward。

这一步可以发现：

- 候选声明了真实 batch 中不存在的输入；
- shape 与真实 response length 或 top-k 不匹配；
- 真实数据上产生 NaN/Inf；
- recorded batch forward/backward 失败。

但 recorded batch 通过：

```python
torch.load(..., map_location="cpu")
```

加载，所以所有 tensor 都在 CPU。它不能发现在线 Ray/FSDP worker 中的 CPU/CUDA 混合放置问题。

### 4.3 第三层：真实 1-step GPU smoke

通过 replay 后，候选才会运行真实的：

- 双 GPU Ray/FSDP；
- 一次 rollout；
- 一次 action objective；
- 一次 actor backward/update。

Smoke 模式下：

- 不运行 validation；
- 不保存 checkpoint；
- 不启动 W&B；
- `TEST_FREQ=0`；
- `VAL_BEFORE_TRAIN=False`。

通过条件为：

- exit code 为 0；
- `actor/grad_norm` finite；
- `actor/pg_loss` finite；
- 至少一个 `opd_action/*` 指标 finite；
- 所有记录指标中不存在 NaN/Inf；
- 对 baseline capture smoke，还要求 recorded batch 文件存在。

所以 smoke 通过只表示“真实训练链能完成一步并产生有限梯度”，不表示 loss 有验证收益，也不保证设计机制真正活跃。

### 4.4 本次 smoke 摘要

| 候选 | 结果 | Grad norm | ESS | Scale | 关键现象 |
|---|---|---:|---:|---:|---|
| R0 Reliable Router | 通过 | 2.046 | 0.9823 | 0.864 | 健康的非均匀 routing |
| R1 First Error Onset | 通过 | 2.030 | 0.99998 | 1.000 | 几乎退化为均匀权重 |
| R1 Union Residual | 失败 | — | — | — | Union support CPU/CUDA mismatch |
| R2 Consensus Bridge | 通过 | 3.017 | 1.000 | 0.593 | Support residual 较强，需向下缩放 |
| R3 History Machine | 失败 | — | — | — | 同样的 union support device mismatch |
| R3 Outcome-safe Switch | 通过 | 2.416 | 0.9979 | 11.856 | 约 90% token 被 neutralize |

---

## 5. 第一轮：round 000

第一轮不是由上一轮实验反馈自动生成，而是来自预先设计的：

```text
opd_loss_search/plans/round_000_action_plan.md
```

核心问题是：

> Teacher/student disagreement 什么时候是可靠的纠错信号，什么时候应当保护 student 已经正确但与 teacher 不同的解法？

### 5.1 候选 A：Reliable Disagreement Router

路径：

```text
round_000/candidates/r0_reliable_disagreement_router/loss.py
```

它不改变 Fixed OPD 的 support-level advantage，只重新分配 token weight。

Teacher reliability 为：

\[
R_t
=
\sqrt{
M_t
(0.25+0.75\,\text{margin}_t)
(0.25+0.75\,\text{overlap}_t)
}.
\]

其中：

- \(M_t\)：teacher top-k probability mass；
- `margin`：teacher top1/top2 margin；
- `overlap`：teacher/student top-k Jaccard。

Learnability 为：

\[
L_t
=
\tanh(\text{JSD}_t/\tau)
\left(
0.5
+0.25\,\text{teacher rejection}
+0.25\,\text{student overconfidence}
\right).
\]

重尾稳定因子：

\[
S_t=\frac{1}{1+|\log r_t|/4}.
\]

最终：

\[
P_t=R_tL_t(0.5+0.5S_t),
\qquad
w_t=1+0.8P_t.
\]

直观上，它只对以下 token 增加 Fixed OPD credit：

- teacher 仍然局部可靠；
- teacher/student 确实存在分歧；
- student 有过度自信或 sampled token 被 teacher 拒绝；
- sampled log-ratio 没有落入极端重尾区。

### 5.2 候选 B：Outcome Correction/Protection

路径：

```text
round_000/candidates/r0_outcome_correction_protection/loss.py
```

它根据最终正确性切换 objective：

- 错误轨迹：加强可靠 teacher disagreement，尤其是数字、算符和 final-answer 区域；
- 正确轨迹：只在低分歧位置 consolidation，保护正确但与 teacher 不同的推理。

近似为：

\[
G_t
=
0.15
+(1-C)R_tD_tO_tL_t
+CR_t(1-D_t),
\]

\[
A'_t=G_tA_t^{\text{Fixed OPD}}.
\]

### 5.3 第一轮验证和 smoke

两个候选都通过了合成验证，因为 synthetic batch 包含 `trajectory_correctness`。

但真实 recorded batch：

```text
recorded_action_batch_9d65b12abb06ab6a.pt
```

没有 `trajectory_correctness`，所以 Outcome candidate 在 replay 阶段被拒绝：

```text
REPLAY_INVALID:
Declared action input is unavailable in this batch: trajectory_correctness
```

它没有进入在线 smoke。

Reliable Router：

- recorded replay 通过；
- 1-step GPU smoke 通过；
- ESS 为 0.982；
- fallback 为 0；
- scale 为 0.864；
- teacher reliability 均值约 0.795；
- routed priority 均值约 0.139。

这说明 routing 确实产生了非均匀变化，但没有变得极端。

### 5.4 第一轮 probe

第一次 10-step 训练的 attempt 0 遇到 Ray infrastructure failure：

```text
Runtime Env Agent timed out
ActorDiedError
Owner's node has crashed
```

协调器将其识别为可重试基础设施错误，自动执行 attempt 1，第二次成功。

Probe 结果：

```text
candidate = 24.17%
baseline  = 22.50%
delta     = +1.667 pp
```

### 5.5 第一轮 confirmation

两个 training seeds：

```text
seed 0: 30.208% vs 29.167% = +1.042 pp
seed 1: 30.000% vs 29.583% = +0.417 pp
```

平均：

```text
+0.729 pp
```

它成为全局 champion。

下一轮 context 得到：

```text
symptom: positive
champion: r0_reliable_disagreement_router
```

---

## 6. 第二轮：round 001

这是第一轮真正由实验反馈驱动生成的新一轮 Python loss。

### 6.1 输入给新一轮 Codex 的信息

Context 明确包含：

- Reliable Router 是 champion；
- probe 为 `+1.667 pp`；
- confirmation 为 `+0.729 pp`；
- 当前 symptom 为 `positive`；
- outcome candidate 没有实验分数；
- 已成功机制是 reliability、disagreement routing 和 tail control。

检索到的 soft priors 是：

- REOPOLD robust clipping；
- Revisiting OPD local support；
- Learning beyond Teacher；
- bounded JSD。

### 6.2 第二轮 brainstorm 的 6 个方向

1. Union support residual rebuild
2. First-error onset credit
3. History hysteresis state machine
4. Region budget conservation
5. Format subspace firewall
6. Prefix-break quarantine

Proposal 最终选择：

- evidence：`r1_union_support_residual_rebuild`
- free：`r1_first_error_onset_credit`

选择逻辑是：

- 第一轮说明 teacher supervision quality 可能重要；
- evidence 路线继续检查 student top-k 是否漏掉 teacher alternative；
- free 路线不再做局部独立 routing，而是尝试沿时间顺序分配 credit。

### 6.3 候选 A：Union Support Residual Rebuild

路径：

```text
round_001/candidates/r1_union_support_residual_rebuild/loss.py
```

Support 从 `student_topk` 扩展到 `union_topk`。

在 union support 上归一化：

\[
p_S(k)
=
\frac{\exp(\ell_S(k))}
{\sum_{j\in U}\exp(\ell_S(j))},
\qquad
p_T(k)
=
\frac{\exp(\ell_T(k))}
{\sum_{j\in U}\exp(\ell_T(j))}.
\]

构造 residual：

\[
r(k)=p_T(k)-p_S(k).
\]

再进行零和中心化：

\[
\tilde r(k)
=
r(k)-\frac{1}{|U|}\sum_{j\in U}r(j).
\]

最终：

```text
base_advantages = centered_residual
raw_token_weights = 1
active_mask = response_mask
```

它不再使用原始 Fixed OPD advantage，目的是纠正 teacher-preferred token 不在 student top-k 时的 support mismatch。

### 6.4 候选 B：First Error Onset Credit

路径：

```text
round_001/candidates/r1_first_error_onset_credit/loss.py
```

它保留 Fixed OPD advantage，只改变时间 credit。

首先检测：

\[
\text{innovation}_t=\max(\Delta\text{NLL}_t,0),
\]

\[
\text{drift}_t
=
\max(
\text{rollingNLL}_{4,t}
-\text{rollingNLL}_{16,t},
0
).
\]

压缩到 `[0,1]`：

\[
g_{\text{innovation}}=1-e^{-\text{innovation}},
\quad
g_{\text{drift}}=1-e^{-\text{drift}}.
\]

再结合：

- sampled token 是否不在 teacher top-k；
- student 是否在 disagreement 下过度自信。

得到 onset hazard \(h_t\)。

累计历史 hazard：

\[
H_{<t}=\sum_{j<t}h_j.
\]

第一次错误优先级：

\[
p_t^{\text{first}}
=
h_t e^{-H_{<t}}.
\]

最终：

\[
w_t=0.5+p_t^{\text{first}}.
\]

越像“第一次错误发生点”的 token，权重越高；前面累计过 hazard 后，后续错误症状会被指数抑制。

### 6.5 第二轮 smoke

两个候选都通过：

- 合成 FP32/BF16；
- empty batch；
- recorded real batch CPU replay。

真实 GPU smoke 中出现分化。

#### Union Support smoke 失败

错误发生在：

```text
build_support()
torch.cat([student_on_student, student_on_teacher], dim=-1)
```

报错：

```text
Expected all tensors to be on the same device,
but found cpu and cuda:0
```

这不是 residual 公式产生 NaN，而是 union-support adapter 的两个 tensor 在在线 worker 上位于不同设备。

Recorded replay 没发现问题，是因为 replay 中所有 tensor 都被加载到 CPU。

因此 Union Support 没有进入 10-step probe。

#### First Error Onset smoke 通过，但机制退化

表面安全指标：

```text
grad norm = 2.030
ESS       = 0.99998
fallback  = 0
scale     = 0.99975
```

但内部指标为：

```text
prior_hazard mean              ≈ 202
first_onset_priority mean      ≈ 0.000171
first_onset_priority zero rate ≈ 96.5%
first_onset_priority p95       ≈ 1.7e-17
```

在长响应中，\(e^{-H_{<t}}\) 很快衰减到接近 0，因此大多数 token：

\[
w_t\approx0.5.
\]

后续 normalizer 再把平均 weight 投影到 1，使它整体接近均匀 Fixed OPD，只有极少量早期 token 得到额外权重。

这说明 smoke 只能证明 loss 能运行，不能证明预期机制有效。

### 6.6 第二轮 probe 与 confirmation

Probe：

```text
candidate = 20.83%
baseline  = 25.00%
delta     = -4.167 pp
```

Confirmation：

```text
seed 0 = -0.625 pp
seed 1 = -0.833 pp
mean   = -0.729 pp
```

First-error 假设没有成功，第一轮 champion 保持不变。

下一轮 context 变成：

```text
positive no validation gain
```

---

## 7. 第三轮：round 002

### 7.1 第三轮如何从前两轮提出新 loss

Codex 现在看到：

- R0 局部 reliability routing：小幅正收益；
- R1 temporal onset：负收益；
- R1 union-support：状态为 `SMOKE_FAILED`，但 context 没有具体 device error；
- champion 仍然是 R0。

Brainstorm 的 6 个方向为：

1. Outcome-conditioned direction split
2. Region-budgeted credit allocation
3. Telescoping prefix potential
4. Support subspace decomposition
5. Batch-matched counterfactual tokens
6. History progress residual

Proposal 合成出：

- evidence：`r2_fixed_pareto_consensus_bridge`
- free：`r2_outcome_conditioned_batch_contrast`

### 7.2 候选 A：Fixed Pareto Consensus Bridge

路径：

```text
round_002/candidates/r2_fixed_pareto_consensus_bridge/loss.py
```

它吸取 Union Residual 直接替换 objective 过于激进的问题，改为：

\[
A'=A^{\text{FixedOPD}}+C.
\]

先计算：

\[
r=p_T-p_S.
\]

再与 local uniform distribution 比较，只保留方向较明确的 support item：

\[
C^+
=
\min\left(
\max(r,0),
\max(p_T-u,0)
\right),
\]

\[
C^-
=
\min\left(
\max(-r,0),
\max(p_S-u,0)
\right),
\]

\[
C=C^+-C^-.
\]

随后将 \(C\) 在有效 support 上中心化为零和，再加到 Fixed OPD 上。

含义是：

- teacher 比 student 更偏好且 teacher 概率高于 uniform，才允许正向 residual；
- student 过度偏好且 student 概率高于 uniform，才允许负向 residual；
- 模糊冲突被 neutralize；
- 完整 Fixed OPD 分支始终保留。

它使用 `student_topk`，因此绕开上一轮 `union_topk` 的 device bug。

### 7.3 候选 B：Outcome-conditioned Batch Contrast

路径：

```text
round_002/candidates/r2_outcome_conditioned_batch_contrast/loss.py
```

它在每个 token position 上把 batch 分成正确和错误轨迹，分别构建 feature prototype：

- JSD；
- student NLL；
- teacher confidence；
- 数学/final-answer region；
- normalized token position。

计算 token 到 correct/incorrect prototype 的距离：

\[
d_C(x),\qquad d_I(x).
\]

Failure affinity：

\[
a(x)=\frac{d_C(x)}{d_C(x)+d_I(x)}.
\]

如果 token 更远离 correct prototype，它就更接近失败状态。

随后：

- 错误轨迹提高 failure-associated token 的 Fixed OPD correction；
- 正确轨迹降低 teacher imitation，保护成功状态；
- 当前位置没有同时出现正确和错误轨迹时，factor 精确回到 1。

这是第一个真正的 batch-relational loss：同一个 token 的 action 会随同 batch 其他轨迹变化。

### 7.4 第三轮 proposal 的三次修复

第一次 proposal 使用：

```python
teacher_normalized_entropy
```

当时 synthetic runtime 没有暴露它，出现：

```text
Declared action input is unavailable:
teacher_normalized_entropy
```

随后：

- repair 1 从 required inputs 中移除该字段；
- repair 2 改为 `teacher_entropy`。

三个 attempt 都记录了同一个旧错误，这与稳定路径上的旧 candidate/function cache 相符。当前 controller 已显式清空 candidate 和 custom function cache。

最终在 contract/runtime 更新后，原始 normalized-entropy 版本被 materialize。

但真实 replay 使用：

```text
recorded_action_batch_81a0a2707c2fd026.pt
```

这份 batch 仍没有 `trajectory_correctness`。因此 Batch Contrast 在 replay 阶段被拒绝，没有进入 smoke。

### 7.5 Consensus Bridge smoke

真实 1-step smoke：

```text
grad norm            = 3.017
ESS                  = 1.000
fallback             = 0
scale                = 0.593
consensus magnitude  ≈ 0.304
valid support sum    ≈ 1e-11
```

Support sum 接近 0，说明零和中心化正确。

Scale 0.593 表示 `Fixed OPD + consensus` 的原始 advantage 比 baseline 更大，runtime 将其缩小到匹配 Fixed OPD 的平均绝对值。

### 7.6 第三轮结果

Probe：

```text
candidate = 22.50%
baseline  = 27.50%
delta     = -5.000 pp
```

该 probe 的 95% CI 不包含 0，是四轮中最明确的负 probe。

但 100-step confirmation：

```text
seed 0 = +0.417 pp
seed 1 = +0.417 pp
mean   = +0.417 pp
```

出现：

```text
10 steps 明显负 → 100 steps 小幅正
```

说明 support-level residual 的短期 dynamics 与长期结果不一致，10-step probe 对这类 loss 的预测能力有限。

由于 `+0.417 pp < +0.729 pp`，它没有替换第一轮 champion。

---

## 8. 第四轮：round 003

### 8.1 第四轮如何提出新 loss

Context 为：

```text
positive no validation gain positive
```

它看到：

- R0 reliability router：小幅正；
- R1 temporal onset：负；
- R2 support consensus：probe 负、confirmation 正，但没有超过 champion。

Brainstorm 产生：

1. Outcome-safe operator switch
2. Prefix debt telescoping
3. History regime state machine
4. Nearest-success witness routing
5. Region-role decomposition
6. Support mass-swap pairs

Proposal 选择：

- evidence：`r3_outcome_safe_reliable_switch`
- free：`r3_history_regime_operator_machine`

### 8.2 候选 A：Outcome-safe Reliable Switch

路径：

```text
round_003/candidates/r3_outcome_safe_reliable_switch/loss.py
```

它组合了前三轮经验：

- 保留 R0 的 reliability router；
- 不再使用 R1 的累计 onset；
- 不在所有 token 上加入 R2 的 dense support residual；
- 根据 correctness/format 做离散 operator switch。

三条路径：

1. 格式正确、答案错误：保留完整 Fixed OPD，并用 R0 reliability priority 增加 token 权重。
2. 格式正确、答案正确：只有 teacher/student top1 agree 时保留 Fixed OPD；disagreement token 置零。
3. 格式错误：只有 format-control region 保留 Fixed OPD。

因此：

\[
A'_t=g_tA_t^{\text{FixedOPD}},
\]

其中 \(g_t\in\{0,1\}\) 是 operator gate。

### 8.3 候选 B：History Regime Operator Machine

路径：

```text
round_003/candidates/r3_history_regime_operator_machine/loss.py
```

它使用历史 NLL EMA，将 token 分成：

- unreliable history；
- improving；
- regressing；
- persistent error。

对应不同 operator：

- unreliable：原始 Fixed OPD；
- improving：仅在 top1 agreement 时 consolidation；
- regressing：放大 Fixed OPD correction；
- persistent：改用 union-support teacher-minus-student residual。

History 在这里不是调节 scalar weight，而是改变 support-level corrective direction。

### 8.4 Recorded batch 升级

第四轮使用：

```text
recorded_action_batch_full_v2_47760c64c9a4f044.pt
```

它已经包含：

- `trajectory_correctness`；
- history tensors；
- region tensors。

因此两个候选都通过真实 batch replay。与第一、三轮 outcome candidate 被拒绝不同，这一轮 outcome candidate 得以进入在线 smoke。

### 8.5 History Machine smoke 再次失败

它在 `union_topk` support 构造处再次触发：

```text
Expected all tensors to be on the same device,
but found cpu and cuda:0
```

这与第二轮 Union Residual 是同一个 adapter/device-placement 问题。

所以 History Machine 没有进入 probe。

### 8.6 Outcome-safe Switch smoke

它数值上通过：

```text
grad norm = 2.416
ESS       = 0.9979
fallback  = 0
```

但内部路径占比为：

```text
failed-valid path       ≈ 4.52%
correct-agreement path  ≈ 5.26%
format-repair path      ≈ 0.25%
neutralized path        ≈ 89.98%
```

约 90% token 的 support objective 被置零。

Global scale matching 为保持与 Fixed OPD 相同的平均绝对值，将剩余约 10% 的 action 放大：

```text
scale ≈ 11.86
```

因此：

- ESS 看起来很好，因为 token weight 没有极端化；
- 但 base advantage 的 operator gate 极度稀疏；
- 剩余 token advantage 被大幅放大。

这是“smoke 通过但机制存在明显风险”的典型情况。

### 8.7 第四轮结果

Probe：

```text
candidate = 22.50%
baseline  = 25.00%
delta     = -2.500 pp
```

Confirmation：

```text
seed 0 = -1.042 pp
seed 1 = +0.625 pp
mean   = -0.208 pp
```

两个训练 seed 方向相反，说明 outcome composition 或稀疏 gate 对训练 seed 较敏感。

第一轮 champion 继续保留。

---

## 9. 第五轮为什么没有 Python loss

最终状态为：

```json
{
  "completed_rounds": [0, 1, 2, 3],
  "no_improvement_rounds": 3,
  "champion": {
    "round": 0,
    "score": 0.007291666666666667
  }
}
```

所以 `round_004` 不存在：

- 没有 context；
- 没有 brainstorm；
- 没有 proposal；
- 没有 Python loss；
- 没有 smoke；
- 没有 probe；
- 没有 confirmation。

这里存在一个运行审计不一致：

- manifest 配置 `no_improvement_patience=2`；
- 当前代码在连续两轮没有超过 champion 后理论上应停止；
- 实际完成了 `round_003`，最终记录 `no_improvement_rounds=3`。

搜索在 8 月 7 日到 11 日间经历了 recorded schema、tensor contract 和 runtime 更新，因此 `round_003` 可能来自显式恢复/续跑，或者运行时使用了不同版本的早停状态。能够确定的是：最终在 `round_003` 后停止，`round_004` 没有执行。

---

## 10. 整个迭代链的核心结论

### 10.1 最成功的是温和 routing，而不是彻底替换 objective

第一轮的成功候选保留了 Fixed OPD support direction，只对可靠 disagreement token 做有边界的 credit reallocation。

相比之下：

- First Error Onset 改变了时间 credit；
- Union Residual 直接替换 support objective；
- Outcome Switch 大规模 neutralize token；
- History Machine 根据历史状态切换 support operator。

这些结构性变化要么未成功运行，要么表现不够稳定。

### 10.2 First Error Onset 的失败原因是可诊断的

它不是简单地“效果不好”，而是 cumulative hazard 在长响应中达到数百，使：

\[
e^{-H_{<t}}\rightarrow 0.
\]

最终约 96.5% token 的 first-onset priority 为 0，loss 几乎退化成均匀 Fixed OPD。

### 10.3 Union-support 路线没有被算法实验否定

第二、第四轮的 union-support 候选都死于同一个 CPU/CUDA device-placement bug。它们没有完成 10-step probe，因此不能据此判断 union support 数学方向无效。

在重新评估这些候选前，应先修复 `build_support()` 中 union tensors 的统一 device placement。

### 10.4 10-step probe 对 support-level loss 预测不足

Consensus Bridge：

```text
Probe        = -5.000 pp
Confirmation = +0.417 pp
```

说明短预算 probe 会错误淘汰一部分起效较慢的 objective。

### 10.5 Outcome Switch 的核心问题是极端稀疏

它 neutralize 约 90% token，再通过 scale matching 放大剩余信号约 11.86 倍。这会提高训练 seed 敏感性，也会让少数 token 承担过多 gradient mass。

### 10.6 跨轮 context 丢失了关键失败原因

下一轮只看到 `SMOKE_FAILED`，看不到 CPU/CUDA stack trace。因此生成器可能将基础设施失败误解为算法失败或未知失败，并在后续轮次重新提出相似结构。

建议将以下信息加入 `_compact_experience`：

- `failure_reason`；
- 最末端异常类型；
- 失败发生阶段：contract、replay、smoke、training、evaluation；
- 是否属于 candidate 数学问题或 runtime infrastructure 问题。

### 10.7 Champion 选择指标与标准 pass@16 存在错位

搜索计划写的是 pass@16，但 `_winner()` 实际使用：

```text
paired.mean_paired_delta
```

这是 480 个 rollout 的平均正确率差，不是标准意义上“每道题 16 次生成至少一次正确”的 best/pass@16。

因此当前 champion 是“逐 rollout mean score 最优”，不一定是标准 pass@16 最优。如果最终研究目标是 pass@16，需要修改 paired report 和 `_winner()` 的 selection metric。

---

## 11. 关键文件索引

### 搜索控制与配置

- `opd_loss_search/search.py`
- `opd_loss_search/orchestration/coordinator.py`
- `opd_loss_search/search_runs/codex_opd_v2/resolved_search_manifest.json`
- `opd_loss_search/search_runs/codex_opd_v2/state.json`
- `opd_loss_search/search_runs/codex_opd_v2/champion.json`

### Prompt 与 contract

- `opd_loss_search/prompts/custom_python_prompt.md`
- `opd_loss_search/contracts/tensor_contract_v2.json`
- `opd_loss_search/contracts/hypotheses.schema.json`
- `opd_loss_search/contracts/proposals.schema.json`
- `opd_loss_search/contracts/candidate.schema.json`

### Runtime 与 validation

- `opd_loss_search/runtime/hook.py`
- `opd_loss_search/runtime/features.py`
- `opd_loss_search/actions/divergences.py`
- `opd_loss_search/actions/normalization.py`
- `opd_loss_search/validation/candidate.py`

### 每轮结果

- `opd_loss_search/search_runs/codex_opd_v2/rounds/round_000/result_summary.json`
- `opd_loss_search/search_runs/codex_opd_v2/rounds/round_001/result_summary.json`
- `opd_loss_search/search_runs/codex_opd_v2/rounds/round_002/result_summary.json`
- `opd_loss_search/search_runs/codex_opd_v2/rounds/round_003/result_summary.json`

### 每轮 Python loss

- `round_000/candidates/r0_reliable_disagreement_router/loss.py`
- `round_000/candidates/r0_outcome_correction_protection/loss.py`
- `round_001/candidates/r1_first_error_onset_credit/loss.py`
- `round_001/candidates/r1_union_support_residual_rebuild/loss.py`
- `round_002/candidates/r2_fixed_pareto_consensus_bridge/loss.py`
- `round_002/candidates/r2_outcome_conditioned_batch_contrast/loss.py`
- `round_003/candidates/r3_outcome_safe_reliable_switch/loss.py`
- `round_003/candidates/r3_history_regime_operator_machine/loss.py`
