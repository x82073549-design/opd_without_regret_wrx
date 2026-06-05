# RG-OPD 完整方案

## 1. 目标

当前 OPD 会在学生 rollout 产生的所有 prefix 上直接蒸馏 teacher 分布。但在数学推理中，学生一旦生成了错误 prefix，teacher 在这个错误 prefix 上给出的分布可能只是“沿着错误上下文补救”，不一定是学生应该学习的目标。

RG-OPD 的目标是：**在 OPD 中加入一个 prefix 可靠性判断机制，只在可信 prefix 上强蒸馏 teacher；在不可信 prefix 上降低 teacher KL 权重，同时对导致 prefix 变差的真实 sampled token 给负反馈。**

---

## 2. 总体方法

RG-OPD 在原 OPD 流程中新增一个数学 PRM 作为 prefix critic。

整个训练流程变成：

```text
student 生成 rollout
→ PRM 判断每个 delimiter-grouped prefix 的可靠性
→ teacher 仍然在 student prefix 上给 token 分布
→ 用 PRM 分数 gate teacher KL
→ 对真实 sampled token 加 prefix reliability delta 纠偏
→ 继续走 VERL 原来的 token-level PPO / OPD 训练
```

这个 PRM 不评价 teacher 的整体能力，而是评价：

```text
当前学生 prefix 是否还值得让 teacher 继续监督。
```

---

## 3. 模型选择

使用一个现成数学 PRM，不重新训练 critic。

默认使用：

```text
Skywork-o1-Open-PRM-Qwen-2.5-1.5B
```

原因是它足够轻，方便先接进 VERL 做在线 gate，后续可以换成Qwen2.5-Math-PRM-7B

但第一版实现固定使用 Skywork 1.5B PRM，先验证方法有效性。

---

## 4. Rollout 处理

student rollout 后，保留原始 response_ids 和 response_mask 不变。

RG-OPD v1 使用 delimiter-grouped segmentation：

1. decode 原始 response_ids 得到 response_text。
2. 在原始 response_ids 的 token 轴上识别两个真实换行字符对应的 token pattern（ASCII 10,10 / LF+LF），切 raw blocks。
3. 去掉空 raw block。
4. 设 raw_block_num = len(raw_blocks)。
5. 设 k = max(1, ceil(raw_block_num / n_prm_blocks))。
6. 每 k 个连续 raw block 合并为一个 PRM block。
7. 因此最终 PRM block 数量 <= n_prm_blocks。
8. 记录每个 PRM block 在原始 response_ids 中的 token span。

两个真实换行字符只用于定位 PRM 分块边界，不参与训练 token ids。
训练 token 轴永远使用原始 response_ids。
不重新 tokenize response_text 来生成训练 ids。

PRM block 到 token 的映射只服务于 reward broadcast。实现时优先在原始 response_ids 上匹配 LF+LF 的 token pattern，包括 tokenizer 把 LF+LF 合成单个 token、拆成两个单 LF token、或合进相邻文本 token 的情况；block 文本由对应 token span decode 得到。

---

## 5. Prefix Gate

PRM 输出每个 PRM block prefix 的可靠性分数后，将其转换为 prefix gate。

gate 的含义是：

```text
gate 高：当前 prefix 可信，teacher KL 保持较高权重。
gate 低：当前 prefix 可能已经偏离正确推理路径，teacher KL 需要降权。
```

Gate 使用当前 prefix 的 PRM 分数，并加入 EMA 平滑。其中 g_0 可以初始化为 V_0，V_0 是 PRM 对 question-only prefix 的分数；如果 PRM 不支持空 prefix，则 g_0 初始化为 1.0。

```text
V_i = PRM(x, y_{\le block_i})

g_i = max(min_gate, λ * g_{i-1} + (1 - λ) * V_i)
```

其中，`V_i` 表示 PRM 对当前 prefix 的可靠性评分，使用 PRM 官方 inference code 输出的 step positive probability；若输出为 logits，则先 sigmoid/softmax 转为 [0,1]，不做反传，只作为 stop-gradient gate。`g_i` 是实际用于 teacher KL 的 gate 权重，`min_gate` 防止 teacher KL 被完全关闭，`λ` 用于平滑相邻 step 之间的 gate 波动。

这样设计的原因是：数学推理确实具有路径依赖，错误 prefix 会影响后续推理；但有些中间错误仍然可能被后续步骤修正。如果使用 cumulative minimum gate，一旦某一步 PRM 分数很低，后续 gate 就无法恢复，容易过度压制 teacher KL。因此，RG-OPD 使用可恢复的 prefix gate：当前 prefix 变差时 gate 下降，当前 prefix 被修复时 gate 可以重新上升。

然后将 block-level gate 广播到 token-level。也就是说，第 i 个 PRM block 完成后得到 V_i，再得到 g_i。

第 i 个 PRM block 覆盖的原始 token_span_i 内，所有 token 使用同一个 g_i。
---

## 6. Candidate Token 集合

当前 OPD 只看 student top-k token。

RG-OPD 增加真实 sampled token。

candidate_ids shape 固定为 [B, T, K+1]
前 K 个是 student top-k，第 K+1 个是 sampled token。
如果 sampled token 已在 top-k 中，sampled slot 仍保留，但 KL mask 置 0，只用于 value delta，避免重复计算 teacher KL。如果 sampled token 不在 top-k 中，sampled slot 正常参与 teacher KL，同时额外叠加 value delta。

候选 token 集合固定为：

```text
student top-k token + 当前真实 sampled token
```

这样做的目的是：如果学生真实生成的 token 不在 top-k 里，它仍然可以收到 token-level credit。

这一步是必须的，否则 value delta 没法真正作用到导致 prefix 变差的 sampled token 上。

---

## 7. Teacher KL 纠偏

teacher 仍然在 student prefix 上 forward，给候选 token 的 logprob。

student 也给这些候选 token 的 logprob。

原 OPD reward 仍然来自 teacher 和 student 的 logprob gap。

RG-OPD 只做一个改动：

```text
每个 token 位置的 teacher KL reward 乘上当前 prefix gate。
```
reward_weight_mode 仍沿用原 OPD 设置。对于 RG-OPD 的 K+1 candidate set，student_p / teacher_p 只在有效 KL candidate 上归一化。若 sampled token 已经出现在 student top-k 中，则 sampled slot 的 KL mask 置 0，不参与 student_p / teacher_p 的归一化，只接收 value delta。

也就是说：

```text
prefix 可信 → teacher KL 正常生效
prefix 不可信 → teacher KL 被大幅降权
```

wrong prefix 后面的续写不直接丢掉，但它们对 OPD loss 的贡献会变小。

---

## 8. Sampled Token Value Delta

除了使用 prefix gate 调整 teacher KL 权重，RG-OPD 还会对真实 sampled token 加入一个额外的过程纠偏信号。

这里使用 PRM 原始分数的变化，对 PRM block i 内所有真实 sampled token 加 value delta，不按 block 覆盖的原始 token 数归一化：

```text
ΔV_i = clip(V_i - V_{i-1}, -delta_clip, delta_clip)

r_delta_t = beta * ΔV_i, t in token_span_i
```

其中 token_span_i 是第 i 个 PRM block 映射回原始 response_ids 后覆盖的 token positions。

如果没有可靠的 V_0，则 g_0 初始化为 1.0，第一块 ΔV_1 设为 0；从第二个 PRM block 开始使用 V_i - V_{i-1}。

---

## 9. VERL 接入位置

RG-OPD 接在 rollout 之后、distillation reward 计算之前。

具体流程是：

```text
1. student rollout 生成原始 responses / response_ids / response_mask
2. student teacher-forcing forward，得到 student top-k
3. 构造 opd_candidate_ids = concat(student_top_k_ids, sampled_token)
4. teacher forward 时 gather opd_candidate_ids 上的 teacher logprob
5. decode response_ids 得到 response_text
6. 在原始 response_ids 上按 LF+LF token pattern 切 raw blocks，并按 k 合并为最多 n_prm_blocks 个 PRM blocks
7. 得到每个 PRM block 覆盖的原始 response token positions
8. PRM 对每个 PRM block prefix 打分，得到 V_i / g_i
9. 将 g_i / ΔV_i broadcast 到对应原始 token_span_i
10. 计算 gated OPD reward 和 sampled-token delta reward
11. 继续走 VERL 原来的 token_reward_direct 和 3D PPO loss
```

PRM 输入构造时使用预先切好的 PRM blocks，不直接让官方代码对原始 response_text 再按单个 LF 重新分段。
每个 PRM block 后追加 PRM 的 step_token 作为打分锚点。

原有 OPD 主流程不改，只替换 reward 构造方式。

---

## 10. 配置项

新增配置如下：

```yaml
algorithm:
  prefix_correction:
    enable: true
    prm_model_path: Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
    segmentation: delimiter_grouped
    # delimiter 不在训练脚本中显式传入；实现默认使用两个真实换行字符（ASCII 10,10 / LF+LF）
    n_prm_blocks: 64
    include_sampled_token: true
    gate_mode: ema
    ema_lambda: 0.6
    min_gate: 0.05
    value_delta_coef: 0.1
    delta_clip: 0.5
    use_prm_gate: true
    use_value_delta: true
```
---

## 11. 训练设置

训练任务使用当前 OPD 的数学任务设置。

teacher、student、rollout 数量、top-k、batch size 等保持当前 OPD baseline 不变。
见 /workspace/s/ddn/shaojw_group/liujiang/OPD/on_policy_distillation.sh。

唯一改动是：

```text
OPD reward → RG-OPD reward
```

这样可以保证实验差异主要来自 prefix gate 和 sampled-token correction。

---

## 12. 实验对比

实验只做四组：

```text
1. 当前 OPD baseline
2. OPD + sampled token
3. OPD + sampled token + PRM gate
4. OPD + sampled token + PRM gate + value delta
```

这四组足够证明方案是否有效。

重点看：

```text
sampled token 是否提升 token-level credit
PRM gate 是否减少坏 prefix 后的 teacher KL 污染
value delta 是否进一步惩罚导致错误的关键 token
```

---

## 13. 主要指标

任务指标：

```text
acc@1 / first@1：第 1 条采样的准确率
avg@16：16 条采样平均正确率，即 mean_score
pass@16：16 条中至少 1 条正确，即 best_score
solve_none
平均 response length
训练稳定性
```

过程指标：

```text
prefix_gate/mean
prefix_gate/low_ratio
prefix_delta/neg_ratio
opd/raw_reward_mean
opd/corrected_reward_mean
opd/sampled_token_in_topk_ratio
opd/gate_active_ratio
```

关键分析指标：

```text
正确轨迹的 prefix gate 是否整体更高
错误轨迹的 prefix gate 是否在错误步骤后下降
value delta 是否集中惩罚 first bad step 附近 token
gate 后 bad prefix 的 teacher KL 是否明显降低
```

---

## 14. 预期结果

如果方案有效，应该看到：

```text
1. 相比 baseline OPD，RG-OPD 的 AIME/AMC mean_score 提升。
2. solve_none 降低。
3. 错误轨迹中，prefix gate 会在关键错误步骤后明显下降。
4. bad prefix 后的 teacher KL reward 被压低。
5. sampled token 不在 top-k 的比例较高，说明加入 sampled token 是必要的。
6. value delta 能把负反馈集中到导致 prefix 变差的位置。
```
