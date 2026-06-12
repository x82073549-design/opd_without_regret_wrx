# OPD（On-Policy Distillation）进展 Survey

截至日期：2026-06-06  
范围：本文将 OPD 狭义定义为 LLM / VLM / agent 后训练中的 **On-Policy Distillation**，即让 student 按照自己的当前策略生成轨迹，再由 teacher / verifier / privileged model 在这些 student-visited states 上提供监督信号。它不同于传统 off-policy KD，也不同于泛化意义上的 RL policy distillation。

---

## 1. 一句话定义

**OPD = student 自己 rollout，teacher 在 student 真实会访问到的 prefix/state 上给反馈，student 再学习这些反馈。**

传统 off-policy KD 通常在 teacher 生成的数据或静态数据集上训练：

\[
y \sim D_{\text{teacher/offline}},\quad
\min_\theta \sum_t D(p_T(\cdot|x,y_{<t}) \Vert p_S(\cdot|x,y_{<t}))
\]

OPD 则改为：

\[
\hat y \sim \pi_S(\cdot|x),\quad
\min_\theta \sum_t D(p_T(\cdot|x,\hat y_{<t}) \Vert p_S(\cdot|x,\hat y_{<t}))
\]

核心变化是：  
teacher 不只在“标准答案前缀”上教学，而是在 student 自己生成的、可能出错的前缀上教学。

---

## 2. OPD 为什么重要？

### 2.1 解决 off-policy KD 的 exposure bias

在 autoregressive LLM 中，训练时常看到的是 teacher-forced prefix，但推理时模型只能看到自己生成的 prefix。  
如果 student 一旦偏离 teacher trajectory，off-policy KD 往往没有告诉它如何从错误状态恢复。

OPD 的主要价值是让 teacher 直接评价 student 自己走到的状态，因此更接近真实推理分布。

### 2.2 对 reasoning model 尤其关键

数学、代码、工具调用、长链推理中，早期 token 的轻微偏差会滚雪球。  
OPD 可以在错误推理链中间给 dense token-level / step-level feedback，比只用 outcome reward 的 RLVR 更“信号密集”。

### 2.3 与 RL 的关系

很多 2026 年论文开始把 OPD 解释为一种特殊的 KL-constrained policy optimization：

- teacher logits / likelihood ratio 可以看成 dense reward；
- RKL / FKL / JSD 等 divergence 可以看成不同 reward-shaping 或 trust-region；
- OPD 更像“有 teacher 的 dense RL”，RLVR 更像“有 verifier 的 sparse RL”。

---

## 3. 发展脉络概览

### 阶段 A：基础思想来源，2011–2020

| 年份 | 论文 | 链接 | 与 OPD 的关系 |
|---|---|---|---|
| 2011 | **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning** / DAgger | https://arxiv.org/abs/1011.0686 | OPD 的 imitation-learning 思想源头：student/policy 访问自己的状态，expert 在这些状态上纠偏。 |
| 2015 | **Policy Distillation** | https://arxiv.org/abs/1511.06295 | 早期 RL policy distillation；还不是现代 LLM OPD，但提出用 teacher policy 压缩 student policy。 |
| 2016 | **Sequence-Level Knowledge Distillation** | https://arxiv.org/abs/1606.07947 | 经典 sequence KD：用 teacher 生成序列训练 student；偏 off-policy。 |
| 2020 | **Autoregressive Knowledge Distillation through Imitation Learning** / ImitKD | https://arxiv.org/abs/2009.07253 | 把 autoregressive KD 和 imitation learning 联系起来，接近“on-policy prefix correction”的思想。 |

---

### 阶段 B：LLM OPD 奠基，2023–2024

| 年份 | 论文 | 链接 | 核心贡献 |
|---|---|---|---|
| 2023 | **MiniLLM: Knowledge Distillation of Large Language Models** / MiniLLM: On-Policy Distillation of Large Language Models | https://arxiv.org/abs/2306.08543 | 提出 reverse KL 导向的 LLM KD，并将训练目标推向 on-policy 优化。RKL 的 mode-seeking 特性被认为更适合保留 teacher 的高质量模式。 |
| 2023 / ICLR 2024 | **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** / GKD | https://arxiv.org/abs/2306.13649 | 现代 LLM OPD 的代表作。提出 Generalized KD：student 生成自己的序列，teacher 在这些 self-generated mistakes 上给反馈；支持多种 divergence，并可结合 RLHF。 |
| 2023 | **f-Divergence Minimization for Sequence-Level Knowledge Distillation** | https://arxiv.org/abs/2307.15190 | 从 f-divergence 角度统一 sequence-level KD，为后续 OPD 的 divergence 设计提供理论工具。 |
| 2024 | **DistiLLM: Towards Streamlined Distillation for Large Language Models** | https://arxiv.org/abs/2402.03898 | 提出 skew KL 和 adaptive off-policy 等技巧，强调 LLM KD 的效率、稳定性和 divergence 选择。虽然不是纯 OPD，但对后续 OPD loss 设计影响很大。 |

---

### 阶段 C：OPD 工业化与 reasoning 扩展，2025

| 年份 | 论文 / 报告 | 链接 | 核心贡献 |
|---|---|---|---|
| 2025 | **Qwen3 Technical Report** | https://arxiv.org/abs/2505.09388 | 在小模型强到弱蒸馏中结合 off-policy 和 on-policy transfer；学生生成序列后对齐强教师 logits，说明 OPD 开始进入工业级模型训练流程。 |
| 2025 | **Token-Level Language Model Alignment as Adaptive Policy Distillation** / AlignDistil | https://arxiv.org/abs/2503.02832 | 将 token-level alignment 解释为 adaptive policy distillation，连接 alignment、KD 和 policy optimization。 |
| 2025 | **Distilling LLMs' Reasoning via Reinforcement Learning** / RLKD | https://arxiv.org/abs/2505.16142 | 将 reasoning distillation 与 RL 结合，代表 KD-RL 融合路线。 |
| 2025 | **Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning** / KDRL | https://arxiv.org/abs/2506.02208 | 统一 KD 与 RL 的 post-training 框架，强调 dense teacher signal 和 sparse reward 的互补。 |
| 2025 | **DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs** | https://openreview.net/forum?id=rc65N9xIrY | 引入 contrastive 思路改进 LLM distillation，对 OPD 中正负样本、偏好样本设计有启发。 |
| 2025 | **A Dual-Space Framework for General Knowledge Distillation of Large Language Models** / DSKD | https://arxiv.org/abs/2504.11426 | 支持 off-policy / on-policy KD，并处理 cross-vocabulary / cross-tokenizer 的 distillation 难题。 |
| 2025 | **Delta Knowledge Distillation for Large Language Models** | https://arxiv.org/abs/2509.14526 | 关注只蒸馏 teacher 与 student 差异部分，降低 KD 噪声和计算成本。 |
| 2025 | **Black-Box On-Policy Distillation of Large Language Models** / GAD | https://arxiv.org/abs/2511.10643 | 代表黑盒 OPD 路线：没有 teacher logits，只能访问 teacher 输出文本，用 adversarial / discriminator 方式做 on-policy distillation。 |

---

### 阶段 D：2026 年爆发期

| 年份 | 论文 / 报告 | 链接 | 方向 |
|---|---|---|---|
| 2026 | **A Survey of On-Policy Distillation for Large Language Models** | https://arxiv.org/abs/2604.00626 | 首篇系统性 OPD survey 之一，提出以 feedback signal、teacher access、loss granularity 等维度组织 OPD。 |
| 2026 | **MiMo-V2-Flash Technical Report** | https://arxiv.org/abs/2601.02780 | 提出 Multi-Teacher On-Policy Distillation，多教师按领域给 dense token-level reward。 |
| 2026 | **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models** / OPSD | https://arxiv.org/abs/2601.18734 | 不依赖外部 teacher，而是让模型利用 privileged information 或自我生成信号进行 on-policy self-distillation。 |
| 2026 | **On-Policy Context Distillation for Language Models** / OPCD | https://arxiv.org/abs/2602.12275 | 将“更强上下文 / 系统提示 / 经验上下文”中的能力蒸馏到无上下文或短上下文 student。 |
| 2026 | **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation** / G-OPD / ExOPD | https://arxiv.org/abs/2602.12125 | 将 OPD 形式化为 dense KL-constrained RL 的特例，并提出 reward extrapolation，使 student 有机会超过 teacher。 |
| 2026 | **Fast and Effective On-policy Distillation from Reasoning Prefixes** | https://arxiv.org/abs/2602.15260 | 利用 reasoning prefix 做更高效的 OPD，减少完整 rollout 成本。 |
| 2026 | **Video-OPD: Efficient Post-Training of Multimodal Large Language Models for Temporal Video Grounding via On-Policy Distillation** | https://arxiv.org/abs/2602.02994 | 将 OPD 扩展到 video temporal grounding。 |
| 2026 | **GLM-5 Technical Report** | https://arxiv.org/abs/2602.15763 | 报告中使用 on-policy cross-stage distillation 作为最后 refinement，缓解能力回退。 |
| 2026 | **Entropy-Aware On-Policy Distillation of Language Models** / EOPD | https://arxiv.org/abs/2603.07079 | 针对 RKL 在高熵状态下过度 mode-seeking 的问题，按 teacher entropy 自适应混合 RKL / FKL。 |
| 2026 | **Scaling Reasoning Efficiently via Relaxed On-Policy Distillation** / REOPOLD | https://arxiv.org/abs/2603.11137 | 把 teacher-student log-likelihood ratio 视为 token reward，引入 reward clipping、动态采样和 exploration-to-refinement curriculum。 |
| 2026 | **X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs** | https://arxiv.org/abs/2603.24596 | 将 OPD 用于 speech LLM 的跨模态能力对齐。 |
| 2026 | **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes** | https://arxiv.org/abs/2603.25562 | 系统分析 OPD 失败模式：one-token signal、teacher 在 drifted prefix 上不可靠、tokenization/special-token mismatch，并给出简单修复。 |
| 2026 | **SODA: Semi On-Policy Black-Box Distillation for Large Language Models** | https://arxiv.org/abs/2604.03873 | 半 on-policy 黑盒蒸馏，用静态 student negatives 和 teacher positives 缓解全量 OPD 成本。 |
| 2026 | **On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning** | https://arxiv.org/abs/2604.07944 | 将 OPD 用于自动驾驶 motion planning。 |
| 2026 | **SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting** | https://arxiv.org/abs/2604.10688 | 通过 signal calibration 和 dual-path adaptive weighting 改进 OPD 的 token/trajectory 权重。 |
| 2026 | **Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe** | https://arxiv.org/abs/2604.13016 | 研究 OPD 成功/失败机制：teacher 与 student 的 thinking pattern 是否兼容，以及 teacher 是否提供真正新增能力。 |
| 2026 | **Rubric-Based On-Policy Distillation** | https://arxiv.org/abs/2605.07396 | 用 rubric / verbal feedback 替代或增强 logits 反馈，代表自然语言教师反馈路线。 |
| 2026 | **SOD: Step-wise On-policy Distillation for Small Language Model Agents** | https://arxiv.org/abs/2605.07725 | 面向 tool-integrated agent，将 OPD 从 token-level 扩展到 step-level，处理工具调用错误级联。 |
| 2026 | **Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning** | https://arxiv.org/abs/2605.07804 | 根据 prefix drift / top-k overlap 剪枝或降权后续 unreliable token，提升长链推理 OPD 的效率和可靠性。 |
| 2026 | **On-Policy Distillation with Best-of-N Teacher Rollout Selection** / BRTS | https://arxiv.org/abs/2605.09725 | 从多个 teacher rollouts 中选择 best-of-N 轨迹，降低 teacher trajectory 方差。 |
| 2026 | **Multi-Rollout On-Policy Distillation via Peer Successes and Failures** | https://arxiv.org/abs/2605.12652 | 对同一 prompt 采样多个 student rollouts，利用 peer successes / failures 提供对比式 on-policy 信号。 |
| 2026 | **Reducing the Safety Tax in LLM Safety Alignment with On-Policy Distillation** | https://arxiv.org/abs/2605.15239 | 将 OPD 用于 safety alignment，目标是在提升安全性的同时减少 helpfulness 能力损失。 |
| 2026 | **Self-Supervised On-Policy Distillation for Enhancing LLM Reasoning** | https://arxiv.org/abs/2605.17497 | 自监督 OPD 路线，减少强外部 teacher 依赖。 |
| 2026 | **Less is More: Early Stopping Rollout for On-Policy Distillation** | https://arxiv.org/abs/2605.27028 | 通过 early stopping rollout 降低无效长 rollout 的计算成本。 |
| 2026 | **Trust-Region Behavior Blending for Policy Optimization** | https://arxiv.org/abs/2605.31159 | 将 trust-region 与 behavior blending 引入 policy optimization，对 OPD/RL 混合训练有启发。 |
| 2026 | **OPRD: On-Policy Representation Distillation** | https://arxiv.org/abs/2606.06021 | 最新方向之一：不只对齐 output logits，而是在 student rollouts 上对齐 hidden representations，降低 full-vocab logits 蒸馏成本。 |

---

## 4. 方法分类

### 4.1 按 teacher access 分类

#### 4.1.1 White-box OPD：teacher logits 可见

代表论文：

- **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**  
  https://arxiv.org/abs/2306.13649

- **MiniLLM: Knowledge Distillation of Large Language Models**  
  https://arxiv.org/abs/2306.08543

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation**  
  https://arxiv.org/abs/2602.12125

- **Entropy-Aware On-Policy Distillation of Language Models**  
  https://arxiv.org/abs/2603.07079

特点：

- teacher 可以输出 full-vocab logits 或 top-k logits；
- token-level dense signal 最强；
- 训练稳定性较好；
- 成本高，尤其是大 vocab、大 teacher、长链推理。

适用场景：

- 同组织内 teacher-student 蒸馏；
- 小模型继承大模型 reasoning；
- teacher 与 student tokenizer 相同或可对齐。

---

#### 4.1.2 Black-box OPD：teacher 只返回文本

代表论文：

- **Black-Box On-Policy Distillation of Large Language Models** / GAD  
  https://arxiv.org/abs/2511.10643

- **SODA: Semi On-Policy Black-Box Distillation for Large Language Models**  
  https://arxiv.org/abs/2604.03873

- **Rubric-Based On-Policy Distillation**  
  https://arxiv.org/abs/2605.07396

特点：

- 无法获取 teacher logits；
- 常用 teacher responses、rubric feedback、preference signal、discriminator signal；
- 更适合 API teacher 或闭源 teacher；
- 信号更稀疏，训练方差更高，容易退化为 preference optimization 或 imitation learning。

适用场景：

- API teacher 蒸馏；
- 跨模型家族蒸馏；
- 无法访问内部 logits 的商业模型。

---

#### 4.1.3 Self-OPD：没有外部 teacher

代表论文：

- **Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models**  
  https://arxiv.org/abs/2601.18734

- **Self-Supervised On-Policy Distillation for Enhancing LLM Reasoning**  
  https://arxiv.org/abs/2605.17497

特点：

- teacher 与 student 可能是同一模型的不同上下文、不同采样、不同阶段或带 privileged information 的版本；
- 降低外部 teacher 成本；
- 风险是自我强化错误、缺乏真正新增能力。

适用场景：

- 无强 teacher；
- 有 verifier 或 privileged context；
- 希望低成本增强 reasoning。

---

#### 4.1.4 Representation-level OPD

代表论文：

- **OPRD: On-Policy Representation Distillation**  
  https://arxiv.org/abs/2606.06021

特点：

- 不只对齐 logits，还对齐 hidden states；
- 可减少 full-vocab logits 传输/存储成本；
- 需要 teacher hidden states，因此仍偏 white-box；
- 对跨架构、跨层映射提出新挑战。

---

### 4.2 按 loss / divergence 分类

#### 4.2.1 Forward KL，FKL

\[
D_{\mathrm{KL}}(p_T \Vert p_S)
\]

性质：

- mode-covering；
- 鼓励 student 覆盖 teacher 的多种可能输出；
- 对 open-ended generation、creative writing、多答案任务更友好；
- 可能把 teacher 的低质量长尾也学进去。

相关论文：

- **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**  
  https://arxiv.org/abs/2306.13649

- **Entropy-Aware On-Policy Distillation of Language Models**  
  https://arxiv.org/abs/2603.07079

---

#### 4.2.2 Reverse KL，RKL

\[
D_{\mathrm{KL}}(p_S \Vert p_T)
\]

性质：

- mode-seeking；
- student 倾向选择 teacher 分布中最有把握的模式；
- 对数学、代码、单答案 reasoning 常见有效；
- 但可能降低 diversity，且在 teacher high-entropy 状态下不稳定。

相关论文：

- **MiniLLM: Knowledge Distillation of Large Language Models**  
  https://arxiv.org/abs/2306.08543

- **Entropy-Aware On-Policy Distillation of Language Models**  
  https://arxiv.org/abs/2603.07079

- **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes**  
  https://arxiv.org/abs/2603.25562

---

#### 4.2.3 Jensen-Shannon / generalized f-divergence

性质：

- 在 FKL 和 RKL 之间折中；
- 稳定性通常比单纯 RKL 好；
- 可配合 student sampling ratio、teacher mixing ratio。

相关论文：

- **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**  
  https://arxiv.org/abs/2306.13649

- **f-Divergence Minimization for Sequence-Level Knowledge Distillation**  
  https://arxiv.org/abs/2307.15190

---

#### 4.2.4 Reward-extrapolated OPD

核心思想：

- 标准 OPD 可看成 reward 与 KL penalty 等权的特殊形式；
- 通过 reward scaling / extrapolation，让 student 不只是模仿 teacher，而是可能超过 teacher。

相关论文：

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation**  
  https://arxiv.org/abs/2602.12125

---

#### 4.2.5 Entropy-aware adaptive divergence

核心思想：

- teacher 低熵：teacher 很确定，RKL / mode-seeking 更合理；
- teacher 高熵：teacher 自己也不确定，FKL / mode-covering 更稳；
- 按 token-level teacher entropy 自适应切换或加权。

相关论文：

- **Entropy-Aware On-Policy Distillation of Language Models**  
  https://arxiv.org/abs/2603.07079

---

### 4.3 按监督粒度分类

| 粒度 | 说明 | 代表论文 |
|---|---|---|
| Token-level logits | 每个 student prefix 上蒸馏 teacher next-token distribution | GKD, MiniLLM, G-OPD, EOPD |
| Sequence-level reward | teacher / verifier 对整条回答打分 | GAD, SODA, RLKD |
| Step-level feedback | 对 reasoning step、tool call、agent action 打分 | SOD, Prune-OPD, Multi-Rollout OPD |
| Prefix-level feedback | 只蒸馏关键 reasoning prefix 或 early prefix | Fast and Effective OPD from Reasoning Prefixes, Prune-OPD |
| Representation-level | 对齐 hidden states 而非 logits | OPRD |
| Rubric / natural-language feedback | teacher 给标准、解释、评分理由 | Rubric-Based OPD |

---

## 5. 代表性论文详解

### 5.1 GKD：On-Policy Distillation of Language Models

论文：**On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**  
链接：https://arxiv.org/abs/2306.13649

贡献：

1. 明确提出 LLM on-policy distillation：student 生成自己的样本，teacher 在这些样本上反馈；
2. 提出 Generalized KD，允许使用多种 divergence；
3. 可在 summarization、translation、arithmetic reasoning、instruction tuning 中使用；
4. 说明 OPD 可与 RLHF 类方法结合。

局限：

- 需要 teacher logits，成本较高；
- teacher 在 student 错误 prefix 上的 logits 未必可靠；
- 对长链推理中的 prefix drift 仍处理不足。

---

### 5.2 MiniLLM

论文：**MiniLLM: Knowledge Distillation of Large Language Models**  
链接：https://arxiv.org/abs/2306.08543

贡献：

1. 强调 reverse KL 对 LLM KD 的价值；
2. 将 token-level KD 推向 on-policy 学习；
3. 对小模型蒸馏大模型具有重要实践意义。

关键点：

- FKL 倾向覆盖 teacher 分布；
- RKL 倾向选择高概率模式；
- reasoning task 中，mode-seeking 往往更有利。

局限：

- RKL 容易导致输出多样性下降；
- teacher high-entropy token 上可能训练不稳定。

---

### 5.3 Qwen3 Technical Report

论文：**Qwen3 Technical Report**  
链接：https://arxiv.org/abs/2505.09388

贡献：

1. 说明 OPD 已经进入工业级 LLM 后训练；
2. 对小模型使用 strong-to-weak distillation；
3. 结合 off-policy 和 on-policy 阶段：
   - off-policy：先用高质量 teacher 数据初始化；
   - on-policy：student 自己生成，teacher 给 logits / dense feedback。

意义：

- OPD 不再只是学术实验，而是大模型家族训练 pipeline 的组成部分；
- 说明 OPD 适合在小模型 reasoning、instruction following、alignment 中做最后能力对齐。

---

### 5.4 GAD：Black-Box On-Policy Distillation

论文：**Black-Box On-Policy Distillation of Large Language Models**  
链接：https://arxiv.org/abs/2511.10643

贡献：

1. 解决 teacher logits 不可见的问题；
2. 将 student 视作 generator，用 discriminator 区分 teacher response 与 student response；
3. 使 OPD 可用于 API teacher / closed-source teacher。

意义：

- 将 OPD 从 white-box distillation 推向 black-box distillation；
- 对商业模型蒸馏、跨组织蒸馏有现实意义。

局限：

- discriminator signal 比 token logits 更稀疏；
- 训练稳定性和 reward hacking 风险更高；
- 法务、服务条款、模型输出使用授权需要额外注意。

---

### 5.5 G-OPD / ExOPD

论文：**Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation**  
链接：https://arxiv.org/abs/2602.12125

贡献：

1. 将 OPD 统一为 dense KL-constrained RL；
2. 说明标准 OPD 是其中一个特殊点；
3. 提出 ExOPD：通过 reward scaling / extrapolation，让 student 有机会超过 teacher；
4. 支持多 teacher / expert merging。

核心洞见：

- 传统 OPD 容易被 teacher 上限限制；
- 如果把 teacher signal 视为 reward 而不是绝对真理，student 可以通过 reward extrapolation 做超越式学习。

---

### 5.6 Rethinking OPD

论文：**Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe**  
链接：https://arxiv.org/abs/2604.13016

贡献：

1. 系统研究 OPD 什么时候成功、什么时候失败；
2. 提出两个关键条件：
   - teacher 与 student 的 thinking pattern 兼容；
   - teacher 确实提供 student 缺失的新能力；
3. 观察到成功 OPD 往往集中在一个 shared small token set 上进行概率质量对齐；
4. 给出实践 recipe：off-policy cold start、teacher-aligned prompt selection 等。

意义：

- 将 OPD 从“方法有效”推进到“机制解释”；
- 对实际训练很有指导意义。

---

### 5.7 Revisiting OPD

论文：**Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes**  
链接：https://arxiv.org/abs/2603.25562

贡献：

1. 指出 token-level OPD 具有低方差但有偏的特点；
2. 总结常见失败模式：
   - one-token signal 过弱；
   - teacher 在 drifted student prefix 上不可靠；
   - tokenization mismatch；
   - special-token mismatch；
3. 提出简单修复：
   - top-k support matching；
   - truncated RKL；
   - top-p filtering；
   - special-token masking。

意义：

- 对工程落地非常重要；
- 说明 OPD 不是“student 采样 + teacher logits”这么简单，细节会显著影响效果。

---

### 5.8 EOPD

论文：**Entropy-Aware On-Policy Distillation of Language Models**  
链接：https://arxiv.org/abs/2603.07079

贡献：

1. 分析 RKL 在 high-entropy teacher distribution 上的问题；
2. 根据 teacher entropy 自适应切换 / 混合 RKL 与 FKL；
3. 在数学 benchmark 上提升 pass@k / diversity。

适用场景：

- reasoning 中既有确定 token，也有开放 token；
- 需要同时保留准确率和多样性。

---

### 5.9 REOPOLD

论文：**Scaling Reasoning Efficiently via Relaxed On-Policy Distillation**  
链接：https://arxiv.org/abs/2603.11137

贡献：

1. 将 teacher-student log-likelihood ratio 解释为 token reward；
2. 引入 mixture-based reward clipping；
3. 使用 entropy-based dynamic sampling；
4. 采用 exploration-to-refinement curriculum。

意义：

- 试图解决长链 reasoning 中 OPD 的 sample efficiency 问题；
- 更接近 OPD 与 RL 的统一训练范式。

---

### 5.10 Prune-OPD

论文：**Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning**  
链接：https://arxiv.org/abs/2605.07804

贡献：

1. 关注长链推理中的 prefix drift；
2. 用 top-k overlap 检测 student prefix 与 teacher 可信区域的偏离；
3. 对后续 unreliable token 降权或截断；
4. 降低训练时间，同时保持或提升性能。

意义：

- 长链 reasoning 中，越往后 teacher 对 student 错误前缀的反馈越可能不可靠；
- Prune-OPD 提供了一个实用的“不要盲目蒸馏所有 token”的思路。

---

### 5.11 SOD：Step-wise OPD for Agents

论文：**SOD: Step-wise On-policy Distillation for Small Language Model Agents**  
链接：https://arxiv.org/abs/2605.07725

贡献：

1. 将 OPD 用于 tool-integrated reasoning / small LM agents；
2. 不只按 token 蒸馏，而是按 step / tool-call 粒度重加权；
3. 处理 agent 中工具调用错误级联问题。

意义：

- OPD 正从纯文本 generation 扩展到 agent action distillation；
- 对小模型 agent 的工具使用能力提升很关键。

---

### 5.12 OPRD

论文：**OPRD: On-Policy Representation Distillation**  
链接：https://arxiv.org/abs/2606.06021

贡献：

1. 质疑只做 output logits OPD 的效率；
2. 在 student rollouts 上对齐 teacher hidden representations；
3. 试图减少 full-vocab logits 存储与计算成本；
4. 为 representation-level OPD 打开新方向。

意义：

- 如果 teacher vocabulary 很大，logits 蒸馏极贵；
- hidden-state distillation 可能成为大模型 OPD 的更高效替代。

局限：

- 需要 white-box teacher；
- 不同模型架构、层数、hidden size 的映射并不简单；
- 跨 tokenizer 仍需额外处理。

---

## 6. OPD 与其他训练范式对比

| 方法 | 数据分布 | 监督信号 | 优点 | 缺点 |
|---|---|---|---|---|
| SFT | 人类/teacher 数据 | hard label | 简单稳定 | 不处理 student 自己犯错后的状态 |
| Off-policy KD | teacher/static 数据 | logits / soft label | 稳定、便宜 | exposure bias，distribution mismatch |
| SeqKD | teacher 生成序列 | sequence hard label | 简洁有效 | 缺少 token-level teacher uncertainty |
| OPD | student 当前策略生成 | teacher logits / reward / feedback | 解决 self-generated mistake，信号 dense | teacher serving 成本高，prefix drift 风险 |
| RLHF / DPO | 偏好数据 | preference | alignment 有效 | 通常不是 on-policy，且 token-level 信号弱 |
| RLVR / GRPO | student rollout | outcome reward / verifier | 可超过 teacher，适合可验证任务 | sparse reward，sample cost 高 |
| OPD + RL | student rollout | dense teacher + sparse reward | 兼具 dense guidance 和 exploration | 实现复杂，权重难调 |

---

## 7. 典型 OPD 训练流程

### 7.1 基础 white-box OPD

1. 准备 prompt batch：  
   \[
   x \sim D_{\text{prompt}}
   \]

2. student rollout：  
   \[
   \hat y \sim \pi_S(\cdot|x)
   \]

3. 对每个 prefix 查询 teacher：  
   \[
   p_T(\cdot|x,\hat y_{<t})
   \]

4. 计算 distillation loss：  
   可选 FKL、RKL、JSD、skew-KL、entropy-aware mixture 等。

5. 更新 student。

6. 重复以上步骤，使数据分布跟随 student 当前策略变化。

---

### 7.2 更稳的工程 recipe

推荐实践：

1. 先做 off-policy cold start  
   - 用 teacher high-quality data 或 SFT 数据初始化 student；
   - 避免 student 一开始生成太差，teacher 在离谱 prefix 上给无意义反馈。

2. 控制 student rollout 分布  
   - temperature 不宜过高；
   - top-p / top-k 采样避免过多噪声；
   - 对数学、代码可使用较低温度。

3. 不盲目蒸馏 full vocabulary  
   - 可用 top-k teacher logits；
   - 可用 support matching；
   - 可过滤 special tokens；
   - 可对 low-confidence teacher token 降权。

4. 检测 prefix drift  
   - teacher top-k 与 student token overlap 过低时，后续 token 的 teacher feedback 可能不可靠；
   - 可截断 rollout 或降低 loss weight。

5. 按 teacher entropy 调整 loss  
   - teacher 低熵：RKL / mode-seeking；
   - teacher 高熵：FKL / mode-covering；
   - 混合 loss 通常更稳。

6. 结合 verifier / outcome reward  
   - OPD 提供 dense token-level guidance；
   - RLVR / verifier 提供最终正确性约束；
   - 对数学、代码、工具任务尤其有用。

---

## 8. OPD 的关键设计选择

### 8.1 Student rollout 采样策略

影响因素：

- temperature；
- top-p / top-k；
- rollout length；
- 是否 early stop；
- 是否 best-of-N；
- 是否多 rollout 对比。

相关论文：

- **Less is More: Early Stopping Rollout for On-Policy Distillation**  
  https://arxiv.org/abs/2605.27028

- **On-Policy Distillation with Best-of-N Teacher Rollout Selection**  
  https://arxiv.org/abs/2605.09725

- **Multi-Rollout On-Policy Distillation via Peer Successes and Failures**  
  https://arxiv.org/abs/2605.12652

---

### 8.2 Teacher signal 类型

| Signal | 优点 | 缺点 | 代表 |
|---|---|---|---|
| Full logits | 信息最完整 | 成本极高 | GKD, MiniLLM |
| Top-k logits | 成本较低 | 丢失长尾概率 | Revisiting OPD |
| Sampled-token likelihood | 成本低 | 信号稀疏、有偏 | REOPOLD-style |
| Sequence reward | 可黑盒 | 方差高 | GAD, SODA |
| Verifier outcome | 可超越 teacher | sparse | RLKD, KDRL |
| Rubric feedback | 可解释 | 难优化 | Rubric-Based OPD |
| Hidden states | 避免 full vocab | 需 white-box | OPRD |

---

### 8.3 Loss 权重与 token weighting

常见策略：

- uniform token weighting；
- teacher confidence weighting；
- entropy-aware weighting；
- advantage weighting；
- prefix-drift-aware pruning；
- step-level divergence weighting；
- peer success/failure contrastive weighting。

相关论文：

- **SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting**  
  https://arxiv.org/abs/2604.10688

- **SOD: Step-wise On-policy Distillation for Small Language Model Agents**  
  https://arxiv.org/abs/2605.07725

- **Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning**  
  https://arxiv.org/abs/2605.07804

---

## 9. OPD 的主要应用方向

### 9.1 小模型继承大模型 reasoning

代表：

- MiniLLM  
  https://arxiv.org/abs/2306.08543

- Qwen3 Technical Report  
  https://arxiv.org/abs/2505.09388

- MiMo-V2-Flash Technical Report  
  https://arxiv.org/abs/2601.02780

优势：

- 用强 teacher 的 dense signal 提升小模型；
- 比纯 RLVR 更 sample-efficient；
- 比 off-policy KD 更贴近 student inference distribution。

---

### 9.2 数学 / 代码 / 可验证推理

代表：

- GKD  
  https://arxiv.org/abs/2306.13649

- REOPOLD  
  https://arxiv.org/abs/2603.11137

- EOPD  
  https://arxiv.org/abs/2603.07079

- Prune-OPD  
  https://arxiv.org/abs/2605.07804

关键点：

- reasoning 任务存在强 prefix dependency；
- OPD 能在中间推理状态纠偏；
- 但长链错误 prefix 会使 teacher feedback 失真，需要 pruning / entropy-aware / verifier 结合。

---

### 9.3 Agent 和工具调用

代表：

- SOD: Step-wise On-policy Distillation for Small Language Model Agents  
  https://arxiv.org/abs/2605.07725

- Multi-Rollout On-Policy Distillation via Peer Successes and Failures  
  https://arxiv.org/abs/2605.12652

关键点：

- agent 的错误不是单 token，而是 action / tool call / observation chain；
- step-wise OPD 更自然；
- 需要把 token-level loss 与 trajectory-level success 结合。

---

### 9.4 多模态 OPD

代表：

- Video-OPD  
  https://arxiv.org/abs/2602.02994

- X-OPD  
  https://arxiv.org/abs/2603.24596

关键点：

- OPD 正从纯语言扩展到 video、speech、多模态 grounding；
- 挑战在于跨模态 state 表示、alignment signal 和长上下文成本。

---

### 9.5 Safety alignment

代表：

- Reducing the Safety Tax in LLM Safety Alignment with On-Policy Distillation  
  https://arxiv.org/abs/2605.15239

关键点：

- 传统安全对齐可能牺牲 helpfulness；
- OPD 可在 student 自己可能产生风险回答的状态上给安全 teacher feedback；
- 目标是减少 safety tax。

---

## 10. 主要问题与开放挑战

### 10.1 Teacher 在 student 错误 prefix 上是否可靠？

OPD 的优势恰恰是 teacher 能看到 student 错误状态。  
但问题是：teacher 未必擅长评价“离 distribution 很远”的错误 prefix。

解决方向：

- prefix drift detection；
- top-k overlap；
- teacher confidence / entropy filtering；
- truncated rollout；
- verifier 校验。

相关论文：

- **Revisiting On-Policy Distillation**  
  https://arxiv.org/abs/2603.25562

- **Prune-OPD**  
  https://arxiv.org/abs/2605.07804

---

### 10.2 OPD 是否会被 teacher 上限限制？

标准 OPD 容易让 student 模仿 teacher，而不是超越 teacher。

解决方向：

- reward extrapolation；
- verifier-guided OPD；
- multi-teacher merging；
- self-play / peer contrastive rollout；
- RL + OPD hybrid。

相关论文：

- **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation**  
  https://arxiv.org/abs/2602.12125

- **Multi-Rollout On-Policy Distillation via Peer Successes and Failures**  
  https://arxiv.org/abs/2605.12652

---

### 10.3 计算成本过高

主要成本：

- student rollout；
- teacher forward；
- full-vocab logits 存储；
- 长链 reasoning 的 token 数暴涨；
- 多 teacher / multi-rollout 进一步放大成本。

解决方向：

- top-k logits；
- sampled-token logits；
- prefix-only OPD；
- early stopping rollout；
- prune unreliable suffix；
- representation-level OPD；
- semi-on-policy black-box distillation。

相关论文：

- **Fast and Effective On-policy Distillation from Reasoning Prefixes**  
  https://arxiv.org/abs/2602.15260

- **Less is More: Early Stopping Rollout for On-Policy Distillation**  
  https://arxiv.org/abs/2605.27028

- **Prune-OPD**  
  https://arxiv.org/abs/2605.07804

- **OPRD**  
  https://arxiv.org/abs/2606.06021

---

### 10.4 Divergence 选择没有统一答案

经验规律：

- reasoning / math / code：RKL 往往有效；
- open-ended generation：FKL 或 JSD 更稳；
- high-entropy teacher token：不宜强行 RKL；
- long-horizon task：需要 prefix-aware 或 step-aware weighting。

相关论文：

- **MiniLLM**  
  https://arxiv.org/abs/2306.08543

- **GKD**  
  https://arxiv.org/abs/2306.13649

- **EOPD**  
  https://arxiv.org/abs/2603.07079

---

### 10.5 Teacher-student thinking pattern mismatch

即使 teacher 更强，如果 teacher 的 reasoning style 与 student 差异过大，student 可能学不好。

表现：

- token-level logits 看似正确，但 student 无法内化完整推理模式；
- student 复制表面格式而非能力；
- teacher 的 chain-of-thought 太长或太跳跃，小 student 难以吸收。

解决方向：

- off-policy cold start；
- teacher-aligned prompt selection；
- intermediate teacher；
- curriculum distillation；
- multi-stage OPD。

相关论文：

- **Rethinking On-Policy Distillation of Large Language Models**  
  https://arxiv.org/abs/2604.13016

---

### 10.6 Cross-tokenizer / cross-architecture 难题

问题：

- teacher 和 student vocabulary 不同；
- token-level logits 无法直接对齐；
- hidden states 维度、层数、架构不同。

相关论文：

- **A Dual-Space Framework for General Knowledge Distillation of Large Language Models**  
  https://arxiv.org/abs/2504.11426

- **OPRD**  
  https://arxiv.org/abs/2606.06021

---

## 11. 实践建议

### 11.1 如果你要训练一个 1B–8B reasoning student

推荐流程：

1. 用高质量 reasoning traces 做 SFT / off-policy KD cold start；
2. 用 student rollout 采样数学、代码、工具 prompt；
3. teacher 在 student prefix 上提供 top-k logits；
4. loss 以 RKL 或 JSD 为主；
5. 对 high-entropy token 混合 FKL；
6. 用 verifier 过滤最终错误样本或加 outcome reward；
7. 对 prefix drift 严重的 suffix 做 prune；
8. 最后用 RLVR / GRPO 做少量强化。

参考论文：

- MiniLLM  
  https://arxiv.org/abs/2306.08543

- GKD  
  https://arxiv.org/abs/2306.13649

- REOPOLD  
  https://arxiv.org/abs/2603.11137

- Prune-OPD  
  https://arxiv.org/abs/2605.07804

---

### 11.2 如果 teacher 是 API 黑盒模型

推荐流程：

1. student on-policy 生成多个候选；
2. teacher 生成 reference 或对 student answers 打 rubric；
3. 用 discriminator / preference / reward model 转换成训练信号；
4. 避免只做 hard imitation，尽量引入 negative samples；
5. 控制查询成本，可采用 semi-on-policy。

参考论文：

- GAD  
  https://arxiv.org/abs/2511.10643

- SODA  
  https://arxiv.org/abs/2604.03873

- Rubric-Based On-Policy Distillation  
  https://arxiv.org/abs/2605.07396

---

### 11.3 如果目标是 long-horizon agent

推荐流程：

1. 不要只按 token 蒸馏；
2. 按 step / action / tool-call 分段；
3. 对每一步计算 divergence 或 success/failure signal；
4. 对错误早期 step 加大权重；
5. 对错误后严重漂移的 suffix 降权；
6. 引入 multi-rollout peer comparison。

参考论文：

- SOD  
  https://arxiv.org/abs/2605.07725

- Prune-OPD  
  https://arxiv.org/abs/2605.07804

- Multi-Rollout OPD  
  https://arxiv.org/abs/2605.12652

---

## 12. 阅读路线推荐

### 入门必读

1. **DAgger**  
   https://arxiv.org/abs/1011.0686

2. **Policy Distillation**  
   https://arxiv.org/abs/1511.06295

3. **Sequence-Level Knowledge Distillation**  
   https://arxiv.org/abs/1606.07947

4. **Autoregressive Knowledge Distillation through Imitation Learning**  
   https://arxiv.org/abs/2009.07253

---

### LLM OPD 核心必读

1. **MiniLLM: Knowledge Distillation of Large Language Models**  
   https://arxiv.org/abs/2306.08543

2. **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**  
   https://arxiv.org/abs/2306.13649

3. **A Survey of On-Policy Distillation for Large Language Models**  
   https://arxiv.org/abs/2604.00626

---

### 机制与失败模式必读

1. **Rethinking On-Policy Distillation of Large Language Models**  
   https://arxiv.org/abs/2604.13016

2. **Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes**  
   https://arxiv.org/abs/2603.25562

3. **Entropy-Aware On-Policy Distillation of Language Models**  
   https://arxiv.org/abs/2603.07079

---

### Scaling / efficiency 必读

1. **Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation**  
   https://arxiv.org/abs/2602.12125

2. **Scaling Reasoning Efficiently via Relaxed On-Policy Distillation**  
   https://arxiv.org/abs/2603.11137

3. **Prune-OPD**  
   https://arxiv.org/abs/2605.07804

4. **OPRD**  
   https://arxiv.org/abs/2606.06021

---

### 黑盒 / API teacher 必读

1. **Black-Box On-Policy Distillation of Large Language Models**  
   https://arxiv.org/abs/2511.10643

2. **SODA: Semi On-Policy Black-Box Distillation for Large Language Models**  
   https://arxiv.org/abs/2604.03873

---

### Agent / 多模态方向

1. **SOD: Step-wise On-policy Distillation for Small Language Model Agents**  
   https://arxiv.org/abs/2605.07725

2. **Video-OPD**  
   https://arxiv.org/abs/2602.02994

3. **X-OPD**  
   https://arxiv.org/abs/2603.24596

---

## 13. 总结

OPD 的发展可以概括为三条主线：

1. **从 off-policy KD 到 student-on-policy KD**  
   - 代表：GKD、MiniLLM  
   - 关键词：self-generated mistakes、student distribution、RKL/FKL/JSD

2. **从单纯模仿 teacher 到 OPD-RL 统一**  
   - 代表：G-OPD、REOPOLD、KDRL、RLKD  
   - 关键词：dense reward、KL-constrained RL、reward extrapolation、verifier

3. **从 token logits 蒸馏到更复杂的反馈形式**  
   - 代表：GAD、SODA、SOD、Prune-OPD、OPRD  
   - 关键词：black-box teacher、step-level feedback、prefix pruning、hidden-state alignment

当前 OPD 的核心矛盾是：

- 它比 off-policy KD 更接近真实推理分布；
- 它比 RLVR 提供更密集的训练信号；
- 但它也带来 teacher serving 成本、prefix drift、teacher-student mismatch、loss 选择和长链稳定性问题。

因此，未来最可能继续发展的方向是：

1. OPD + RLVR / verifier 的统一训练；
2. 长链 reasoning 的 prefix-aware / step-aware OPD；
3. 黑盒 teacher 的低成本 OPD；
4. representation-level OPD；
5. 多模态和 agentic OPD；
6. 可超过 teacher 的 reward-extrapolated OPD。