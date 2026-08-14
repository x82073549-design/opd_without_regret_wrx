## 1. 基础 OPD 与 KL 目标修改

| 论文 | Loss 做法 |
|---|---|
| [EOPD: Entropy-Aware On-Policy Distillation](https://arxiv.org/abs/2603.07079) | 根据教师 entropy 自适应组合 FKL 和 RKL。低熵位置主要采用 RKL，集中学习教师的主要模式；高熵位置增加 FKL，避免 RKL 的 mode-seeking 特性丢失教师的多个合理候选。 |
| [Asymmetric On-Policy Distillation（AOPD）](https://arxiv.org/abs/2605.06387) | 按 token advantage 的符号切换目标。正 advantage 区域保留 policy-gradient 强化；非正 advantage 区域改用局部 divergence minimization，减少负 advantage 的高方差、零 advantage 的梯度消失以及错误 token 抑制后缺少替代方向的问题。 |
| [Trust Region On-Policy Distillation（TrOPD）](https://arxiv.org/abs/2606.01249) | 按监督可靠性划分 loss：可信区域使用标准 OPD；异常区域采用 gradient clipping、loss mask 或 FKL；从 teacher prefix 继续生成的 off-policy 轨迹使用 FKL 模仿，将学生引回教师可可靠监督的区域。 |

## 2. 隐式奖励与 Reward Extrapolation

| 论文 | Loss 做法 |
|---|---|
| [Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation（G-OPD/ExOPD）](https://arxiv.org/abs/2602.12125) | 将教师相对 reference model 的 log-ratio 解释为隐式 token reward，并引入外推系数 \(\lambda\) 调整 reward 相对 KL regularization 的强度。ExOPD 使用 \(\lambda>1\) 放大教师的改进方向，使学生不只是拟合教师最终分布。 |
| [Scaling Reasoning Efficiently via Relaxed On-Policy Distillation（REOPOLD）](https://arxiv.org/abs/2603.11137) | 将教师—学生 log-likelihood ratio 作为 token reward，并通过 mixture-based reward clipping 限制极端信号；再结合 entropy-guided token sampling 和 exploration-to-refinement 分阶段策略，放松标准 OPD 的严格模仿约束。 |
| [REOPD: Reliability-Adaptive Reward Extrapolation for On-Policy Distillation](https://arxiv.org/abs/2608.11698) | 将 ExOPD 的全局固定外推系数改为 token-wise 系数 \(\lambda_{b,t}=1+\gamma_bq_t\)。其中 \(q_t\) 衡量该 token 上 teacher-reference 改进方向与学生的兼容性，\(\gamma_b\) 控制 batch 级外推预算，只沿可靠方向放大奖励。 |

## 3. Token 选择、加权、Mask 与 Clip

| 论文 | Loss 做法 |
|---|---|
| [TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084) | 根据学生 entropy 和教师—学生 divergence 选择重要 token。重点保留两类位置：学生高 entropy 的不确定 token，以及学生低 entropy 但 teacher-student divergence 高的“过度自信且错误”token。未选中的 token 不承担直接 OPD loss。 |
| [Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning](https://arxiv.org/abs/2605.07804) | 用教师与学生的局部兼容性（如 top-k overlap）检测 prefix drift；漂移发生后单调降低后续 token reward 权重，并动态截断 rollout，使 loss 只集中在教师仍能提供有效指导的前缀区域。 |
| [Escaping the KL Agreement Trap in On-Policy Distillation（KAT）](https://arxiv.org/abs/2606.09471) | 检测错误前缀中“持续低 KL、但无纠错价值”的 agreement trap。当低 KL 状态超过动态阈值时，终止或 mask 后续 token loss，避免把局部教师—学生一致误认为有效监督。 |
| [On the Position Bias of On-Policy Distillation（IW-OPD）](https://arxiv.org/abs/2606.22600) | 根据当前 token 之前累计的教师—学生分布 discrepancy 计算 importance weight。前缀累计偏差越大，后续 token 权重越低，从而自然提高前段可信 token 的贡献并降低后段漂移 token 的 loss。 |

## 4. 轨迹筛选与 Token 加权联合方法

| 论文 | Loss 做法 |
|---|---|
| [Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models（StableOPD）](https://arxiv.org/abs/2604.08527) | 在标准 OPD 中加入 reference-based divergence constraint，并使用 rollout mixture distillation。reference loss 限制策略漂移，混合轨迹目标抑制长度膨胀、重复饱和和 truncated trajectory 主导训练。 |
| [SCOPE: Signal-Calibrated On-Policy Distillation Enhancement](https://arxiv.org/abs/2604.10688) | 按最终正确性把 rollout 分成两个 loss 分支：错误轨迹使用 teacher-PPL 加权 KL，降低教师无法纠错时的信号；正确轨迹使用 student-PPL 加权 MLE，强化学生尚不熟悉的正确路径。两个分支都进行 group-level normalization。 |
| [Filter, Then Reweight: Rethinking Optimization Granularity in On-Policy Distillation（FiRe-OPD）](https://arxiv.org/abs/2606.02684) | 先在 trajectory 粒度过滤教师监督能力较低的轨迹，再在 token 粒度结合教师 confidence 与学生 uncertainty 对蒸馏 loss 进行软加权，实现“trajectory filtering + token reweighting”。 |
| [Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations](https://arxiv.org/abs/2607.13399) | 对 token-level OPD advantage 进行 clipping 和 log-scale compression，控制教师—学生 mismatch 引起的极端 reward，并抑制由聚合 token 目标产生的长度利用、截断或冗余填充。 |

## 5. Outcome、Ranking 与 RL Loss 融合

| 论文 | Loss 做法 |
|---|---|
| [Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe](https://arxiv.org/abs/2605.03677) | 将 token-level OPD reward 聚合成 trajectory return，并要求同一问题下正确轨迹的 return 高于错误轨迹。通过 margin mask 删除排序不一致的 group，或通过 margin shift 调整正确轨迹 return，恢复蒸馏信号与 outcome reward 的顺序一致性。 |
| [SAF-OPD: Stable Advantage Fusion for On-Policy Distillation](https://arxiv.org/abs/2607.29209) | 将 RLVR 的 response-level advantage 与 OPD 的 token-level advantage 融合。先对 OPD advantage 执行 sparsify-then-compress 控制量级，再对融合系数执行 warmup-then-anneal，避免固定系数导致 entropy collapse 和长期过度模仿教师。 |

