# OPD without regret · Learning-Utility Verifier Formalization Handoff

> 作者：Xiao  
> 日期：2026-07-30  
> 用途：供 Xiao、RA 及后续 GPT 共同 formalize 研究问题。  
> 状态：Token-level M0 formalization 已冻结；早期探索过程保留作设计依据。实施时以 `OPD_verifier_M0_spec_2026-07-30.md` 为准，不得把尚未通过实验门槛的候选结论写成结果。

## 1. 一句话研究目标

我们希望从“人工挑选若干 feature 并设计固定启发式权重”，升级为学习一个条件化决策器：它根据当前 student、候选 teacher、question、token/prefix 和训练阶段，预测某个蒸馏动作对 student 未来能力的学习价值，并据此动态决定：

1. **Question 级**：现阶段应该采样哪些问题；
2. **Teacher 级**：现阶段 student 最适合向哪个 teacher 学习；
3. **Token 级**：哪些 token 应该多学、少学或跳过，以及使用 FKL、RKL 或二者的组合；在当前语义中，OPD 是 RKL 的 on-policy 实现/基线，不作为与 RKL 并列的概念动作。

这里的核心对象不是某个固定 feature，而是一个能把状态和候选学习动作映射到“预期学习收益”的 verifier/controller。

## 2. 为什么要转向这个问题

此前实验分别尝试了 student entropy、teacher confidence/entropy、top-k overlap、FiRe、symmetric KL、EOPD gate 等 feature 或固定规则，但没有得到稳定、统一的方向：

- OPD baseline 仍是最可靠的锚点；
- 低强度 overlap 只有很小的探索性信号；
- student entropy 的最佳方向和强度随测试集变化；
- 强 FiRe 加权在多个测试集上明显变差；
- teacher entropy 的较好信号来自 partial/crashed run；
- EOPD 尚未完成严格、clean 的复现；
- 多数比较只有单 seed，缺少独立 dev 选 checkpoint、逐题预测和完整统计证据。

这些结果不能证明“feature 没用”。更合理的解释是：同一个 feature 的含义取决于 student 当前能力、问题难度、teacher 适配性、token 位置和训练阶段，因此不应被直接写成全局固定规则。Feature 更适合作为决策器的**状态输入**。

相关材料：

- `OPD_feature_list_README.md`
- `OPD_实验结果整理_2026-07-22.md`
- `evidence/2026-07-22/ruxin_experiment_report.md`
- `evidence/2026-07-22/student_entropy_fire_report.md`

## 3. 初始想法与需要修正的地方

初始直觉是定义：

\[
S(\text{question},\text{token},p_S,p_T)\rightarrow R,
\]

其中 \(p_S\) 和 \(p_T\) 是 student/teacher 在当前位置的预测分布，\(R\) 表示该配置下的学习质量。

这个式子表达了“学习一个 verifier”的方向，但还缺少一个关键变量：**候选学习动作**。如果 action 不进入 \(S\)，同一个输入状态下便无法区分：

- 应该跳过还是学习；
- 应该使用 FKL、RKL，还是二者的 mixture；
- token 权重应该是多少；
- 应该选哪个 teacher；
- question 是否值得进入当前训练 batch。

因此，当前更完整的候选定义是：

\[
S_\phi(c,a)\rightarrow \widehat{\Delta J},
\]

其中 \(c\) 是当前学习状态，\(a\) 是候选学习动作，输出是该动作带来的预期未来学习收益。

## 4. 当前候选 formalization

### 4.1 状态

在 student 训练第 \(k\) 阶段，记 student 参数为 \(\theta_k\)。对问题 \(x\)、候选 teacher \(T\)、student rollout prefix \(h_t=(x,y_{<t})\)，定义候选状态：

\[
c_{k,t}
=
\left(
\psi_k,\,
x,\,
T,\,
h_t,\,
p_{S,k,t},\,
p_{T,k,t},\,
z_{k,t}
\right).
\]

其中：

- \(\psi_k\)：训练阶段摘要，例如 global step、当前能力、最近验证表现、历史学习动态；
- \(x\)：question 及其可计算属性；
- \(T\)：teacher 身份、能力/成本摘要或 teacher pool 信息；
- \(h_t\)：当前 token 的生成前缀和位置；
- \(p_{S,k,t}\)、\(p_{T,k,t}\)：师生分布；
- \(z_{k,t}\)：由上述对象计算的 feature，例如 entropy、KL、top-k overlap、top-1 agreement、margin、sampled-token log-prob gap、正确性/可验证 reward、长度、截断或时序变化。

以上是完整研究方向中的候选变量；token-level M0 的实际输入、checkpoint-level stage 表示和可观测性已在下文冻结。

#### M0 冻结的 student-stage 表示

M0 不只使用 normalized training step，也不允许把 search-validation 或 locked-test 表现作为 verifier 输入。使用一组与 inner-train、search-validation 和 test 均互斥的固定 calibration/probe questions，在预定 checkpoint 上形成：

\[
u_k
=
\left[
\frac{k}{K},\,
\bar R_k^{\mathrm{probe}},\,
\Delta\bar R_k^{\mathrm{probe}},\,
\bar H_{S,k}^{\mathrm{probe}},\,
\bar D_{R,k}^{\mathrm{probe}}
\right].
\]

其中：

- \(k/K\)：归一化训练进度；
- \(\bar R_k^{\mathrm{probe}}\)：student 在固定 probe questions 上的平均可验证 reward；
- \(\Delta\bar R_k^{\mathrm{probe}}\)：相对上一次 probe evaluation 的 reward 变化；
- \(\bar H_{S,k}^{\mathrm{probe}}\)：student 在 probe rollout 上的平均 token entropy；
- \(\bar D_{R,k}^{\mathrm{probe}}\)：student 与固定 teacher 在 probe rollout 上的平均 RKL/OPD divergence。

这些量只在预注册的 checkpoint/evaluation interval 更新，在两次 probe 之间保持上一份 snapshot，不按单 batch 的噪声即时更新。所有 stage statistics 对 student inner update 使用 stop-gradient。

数据隔离要求：

- probe/calibration questions 不进入 inner training；
- probe statistics 可以作为 \(S\) 的输入，但不能兼作 outer search reward；
- search-validation 只用于评价 controller/policy 的 learning utility；
- locked test 只在 controller、checkpoint 和所有超参数冻结后使用。

第一版消融至少比较：

1. `step-only`：只使用 \(k/K\)；
2. `capability-aware`：使用完整 \(u_k\)。

这样可以验证 verifier 的 stage dependence 是否真的来自当前 student 能力，而不只是记住训练步数。

#### M0 冻结的 rollout-level 表示

对 question \(x_q\) 的第 \(i\) 条 student rollout，设同题共有 \(n\) 条 rollout，定义：

\[
r_{q,i}
=
\left[
R_{q,i},\,
\widehat p_{q,-i},\,
R_{q,i}-\widehat p_{q,-i},\,
\frac{L_{q,i}}{L_{\max}},\,
\mathbb 1_{\mathrm{trunc}},\,
\mathbb 1_{\mathrm{format}}
\right],
\]

其中：

\[
\widehat p_{q,-i}
=
\frac{1}{n-1}
\sum_{j\ne i}R_{q,j}.
\]

各字段含义：

- \(R_{q,i}\)：当前完整 response 的 verifier reward；
- \(\widehat p_{q,-i}\)：排除当前 rollout 后，同题其他 rollout 的平均 reward，用作当前 student 对该 question 的局部掌握度/难度估计；
- \(R_{q,i}-\widehat p_{q,-i}\)：当前 rollout 相对同题其他采样的结果差异；
- \(L_{q,i}/L_{\max}\)：归一化 response 长度；
- \(\mathbb 1_{\mathrm{trunc}}\)：是否被最大长度截断；
- \(\mathbb 1_{\mathrm{format}}\)：是否通过格式检查。

使用 leave-one-out group reward，是为了不让当前 rollout 自己同时决定“问题难度”和自身 outcome。若 reward 不是二值 exact reward，则 \(\widehat p_{q,-i}\) 表示 leave-one-out mean reward，而不是严格的 pass rate。

M0 不在 rollout-level state 中加入：

- teacher 自己生成答案后的正确性，除非实验明确增加 teacher rollout；
- search-validation 或 locked-test 的任何信息；
- 本次 student update 之后才会出现的指标；
- 无法在部署时同样获得的人工标注。

当 \(n=1\) 时，leave-one-out 量不可定义；M0 要么要求 \(n\ge2\)，要么将对应字段设为缺失值并增加显式 missingness mask，不能默认填 0 并混同“同题其他 rollout 全错”。

#### M0 冻结的 token-level distribution 表示

M0 同时输入 **top-k 压缩师生分布** 与一组显式关系 feature。固定 \(K=16\)，与当前 OPD student top-k 支持对齐。

对位置 \(t\)，定义联合支持：

\[
\mathcal U_t
=
\operatorname{TopK}_{16}(p_{S,t})
\cup
\operatorname{TopK}_{16}(p_{T,t})
\cup
\{y_t\},
\]

其中 \(y_t\) 是 student rollout 实际采样的 token。显式加入 \(y_t\)，避免 sampled token 不在任一 top-k 时被压缩表示丢失。

对每个 \(v\in\mathcal U_t\)，构造：

\[
e_t(v)
=
\left[
\log p_{S,t}(v),\,
\log p_{T,t}(v),\,
\operatorname{rank}_{S,t}(v),\,
\operatorname{rank}_{T,t}(v),\,
\mathbb 1[v=y_t]
\right].
\]

若 \(v\) 不在某一方 top-k 中，其真实 log-prob 仍从该方完整 logits 中 gather；对应 rank 使用显式 out-of-top-k 值 \(K+1\)，不能把概率填 0。Rank 在输入模型前除以 \(K+1\) 归一化。

联合支持使用 permutation-invariant encoder（例如 DeepSets 或小型 set attention）聚合：

\[
g_t^{\mathrm{dist}}
=
\operatorname{SetEnc}
\left(
\{e_t(v):v\in\mathcal U_t\}
\right).
\]

同时保留 tail mass：

\[
m_{S,t}^{\mathrm{tail}}
=
1-\sum_{v\in\mathcal U_t}p_{S,t}(v),
\qquad
m_{T,t}^{\mathrm{tail}}
=
1-\sum_{v\in\mathcal U_t}p_{T,t}(v).
\]

这避免两个 top-k 内部形状相同、但未覆盖概率质量完全不同的分布被视为相同。

显式 token-level features 冻结为：

\[
f_t
=
\left[
\frac{t}{L},\,
H(p_{S,t}),\,
H(p_{T,t}),\,
D_{\mathrm{FKL},t},\,
D_{\mathrm{RKL},t},\,
\operatorname{overlap@16}_t,\,
\mathbb 1[\operatorname{top1}_S=\operatorname{top1}_T],\,
\operatorname{margin}_{S,t},\,
\operatorname{margin}_{T,t},\,
\log p_{S,t}(y_t),\,
\log p_{T,t}(y_t),\,
\log p_{T,t}(y_t)-\log p_{S,t}(y_t),\,
m_{S,t}^{\mathrm{tail}},\,
m_{T,t}^{\mathrm{tail}}
\right].
\]

这里 FKL/RKL 的具体支持、重归一化和数值实现必须与对应 candidate loss 的审计定义一致；不能让 feature 中的“RKL”与实际 OPD surrogate 使用不同语义而不注明。

M0 的完整 token state 因而写成：

\[
c_{k,q,i,t}
=
\left[
u_k,\,
r_{q,i},\,
g_t^{\mathrm{dist}},\,
f_t
\right].
\]

M0 暂不加入 raw question text、完整 prefix、student hidden state、teacher hidden state或可训练 token-ID embedding。这样第一版仍能利用分布形状，但不会把结果归因复杂化为文本语义/表征学习问题。后续可通过 ablation 比较：

1. explicit features only；
2. top-k set representation only；
3. top-k representation + explicit features。

### 4.2 动作

可以将动作分为三个层次：

#### Question-level action

\[
a_k^Q(x)\in
\{\text{skip},\text{sample}\}
\quad\text{或}\quad
a_k^Q(x)=w_x\ge 0.
\]

它决定问题是否被采样，或问题的采样权重。

#### Teacher-level action

\[
a_k^T(x)\in\mathcal T,
\]

即从 teacher pool 中选择 teacher，也可以包含“不使用 teacher”或 teacher mixture。

#### Token-level action

当前更合适的定义是直接输出 FKL/RKL 两个非负权重：

\[
a_{k,t}^{\mathrm{tok}}
=
\left(w_{k,t}^{F},w_{k,t}^{R}\right),
\qquad
w_{k,t}^{F},w_{k,t}^{R}\ge 0,
\]

\[
\ell_{k,t}
=
w_{k,t}^{F}\ell_{k,t}^{\mathrm{FKL}}
+
w_{k,t}^{R}\ell_{k,t}^{\mathrm{RKL/OPD}}.
\]

这里：

- \((w^F,w^R)=(0,0)\) 等价于 skip；
- \(w^F>0,w^R=0\) 表示只使用 FKL；
- \(w^F=0,w^R>0\) 表示只使用 RKL/OPD；
- 二者同时为正表示 token-level mixture；
- \(w^F+w^R\) 表示该 token 的总学习强度，二者比例表示方向。

权重必须有上界、归一化或固定 batch-level 总预算，否则策略可能只通过放大整体梯度获得短期优势。当前默认 OPD baseline 可表示为所有有效 token 上 \((w^F,w^R)=(0,1)\)。M0 的具体上界与 batch-level 总预算已在下文冻结；更稀疏或连续的后续版本尚未决定。

一个等价且更便于解释的参数化是：

\[
a_{k,t}^{\mathrm{tok}}=(w_{k,t},\lambda_{k,t}),
\qquad
w_{k,t}\in[0,w_{\max}],
\quad
\lambda_{k,t}\in[0,1],
\]

\[
\ell_{k,t}
=
w_{k,t}
\left[
\lambda_{k,t}\ell_{k,t}^{\mathrm{FKL}}
+
(1-\lambda_{k,t})\ell_{k,t}^{\mathrm{RKL/OPD}}
\right].
\]

其中 \(w_t\) 单独表示“学多少”，\(\lambda_t\) 表示“沿哪个 KL 方向学”：\(w_t=0\) 为 skip，\(\lambda_t=1\) 为纯 FKL，\(\lambda_t=0\) 为纯 RKL/OPD。该参数化与 \((w_t^F,w_t^R)\) 等价，但更清楚地区分学习强度与方向。

#### M0 冻结的 token-weight 约束

对一个 global optimizer batch 中所有有效 OPD token 的集合 \(\mathcal M\)，M0 冻结：

\[
0\le w_t\le 2,
\qquad
\sum_{t\in\mathcal M}w_t=|\mathcal M|.
\]

等价地，有效 token 的平均权重严格为 1：

\[
\frac{1}{|\mathcal M|}
\sum_{t\in\mathcal M}w_t=1.
\]

含义如下：

- baseline 为所有有效 token \(w_t=1\)；
- \(w_t=0\) 允许 skip；
- \(0<w_t<1\) 表示少学；
- \(1<w_t\le2\) 表示多学；
- controller 只能在 token 之间重新分配固定学习预算，不能放大整个 batch 的总权重；
- \(w_{\max}=2\) 意味着即使采用最极端的 `0/2` 分配，也至少保留约一半的有效 token mass，避免第一版出现过度稀疏和梯度集中。

若 controller 先输出 unconstrained raw weights \(\widetilde w_t\)，实际训练权重定义为其在 capped simplex 上的投影：

\[
\mathbf w
=
\arg\min_{\mathbf v}
\|\mathbf v-\widetilde{\mathbf w}\|_2^2
\quad
\text{s.t.}\quad
0\le v_t\le2,\quad
\sum_{t\in\mathcal M}v_t=|\mathcal M|.
\]

该投影必须按 **global optimizer batch** 计算；分布式训练需要对有效 token 数和权重和做同步，不能让每个 micro-batch 独立改变总预算。若没有有效 OPD token，则该 batch 的 distillation loss 为 0；若 raw controller 输出退化或不可用，则回退到 \(w_t=1\) 的 OPD baseline。

训练时还必须记录：

- mean/min/max/std 与权重直方图；
- \(w_t=0\)、\(w_t<1\)、\(w_t>1\) 的 token 比例；
- effective sample size：

  \[
  \operatorname{ESS}
  =
  \frac{\left(\sum_t w_t\right)^2}
  {\sum_t w_t^2};
  \]

- 权重与 token 位置、entropy、KL、overlap 和正确性的关系；
- 投影前后权重差异及 baseline fallback 次数。

FKL/RKL 两个 basis 在进入 mixture 前仍需独立做尺度对齐；固定总 token weight 不能替代 loss/advantage RMS matching。Student inner update 使用 stop-gradient 后的 \(w_t,\lambda_t\)，避免 controller 通过 student-loss 反向路径引入未定义梯度。

#### M0 冻结的 hard action space 与联合分配

M0 不允许同一个 token 同时使用连续 FKL/RKL mixture，而让每个 token 从以下七个动作中 hard-select 一个：

\[
\mathcal A_{\mathrm{tok}}
=
\{\mathrm{skip}\}
\cup
\{(\mathrm{FKL},w):w\in\{0.5,1,2\}\}
\cup
\{(\mathrm{RKL/OPD},w):w\in\{0.5,1,2\}\}.
\]

其中固定 OPD baseline 是每个有效 token 选择 \((\mathrm{RKL/OPD},1)\)。不同 token 可以选择不同动作；不是整个 batch 只能选择同一种 action。

为同时保留离散动作语义和精确总预算，formal target 写成 batch-level constrained allocation：

\[
(a_1^*,\ldots,a_N^*)
=
\arg\max_{a_t\in\mathcal A_{\mathrm{tok}}}
\sum_{t=1}^{N}S_\phi(c_t,a_t)
\quad
\text{s.t.}\quad
\sum_{t=1}^{N}w(a_t)=N.
\]

这里 \(w(\mathrm{skip})=0\)，\(N=|\mathcal M|\) 是 global optimizer batch 的有效 OPD token 数。该约束意味着 controller 在 token 间联合分配固定预算；例如四个 token 可以选择权重 `(0, 2, 1, 1)`。实现采用 deterministic Lagrangian selection + exact-budget residual repair：必须精确满足总预算，但在没有证明前不得声称得到全局最优整数解。Capped-simplex projection 只保留为未来连续 action space 的备选，不属于 hard-action M0。

### 4.3 输出：什么叫“学习质量”

当前首选候选是短训练窗口后的 held-out downstream reward 增量：

\[
S_\phi(c,a)
\approx
\mathbb E\left[
J(\theta_{k+H}^{(a)})-J(\theta_k)
\mid c,a
\right],
\]

其中：

- \(U_H(\theta_k;a)\) 表示在状态 \(c\) 下执行动作 \(a\)，经过 \(H\) 个受控更新得到 \(\theta_{k+H}^{(a)}\)；
- \(J(\theta)\) 是独立 held-out questions 上的可验证任务 reward；
- \(H\) 是短训练 horizon，而不是只观察当前 token loss。

如需考虑效率，可定义：

\[
\Delta J_{\mathrm{net}}
=
J(\theta_{k+H}^{(a)})-J(\theta_k)
-\lambda C(a),
\]

其中 \(C(a)\) 是 teacher inference、额外 rollout、显存、wall-clock 或 token budget 成本。

这仍是候选定义，尚未决定：

- 使用一步、短 horizon 还是更长 horizon；
- 预测绝对 reward、reward 增量、相对 baseline 增量还是 risk-adjusted gain；
- 是否扣除成本；
- 是否需要同时预测均值、不确定性和失败风险。

#### M0 冻结的 verifier 输出语义

M0 不把 \(S_\phi(c_t,a_t)\) 宣称为可直接观测的“单 token 真实 reward”。它表示：在当前状态 \(c_t\) 下，把 token action 从固定 OPD baseline 改为 \(a_t\) 的**相对学习价值/advantage**。

定义 baseline action：

\[
a^0=(\mathrm{RKL/OPD},1).
\]

令一个未中心化 action-value model 为 \(Q_\phi(c,a)\)，实际 verifier 输出定义为：

\[
A_\phi(c,a)
=
Q_\phi(c,a)-Q_\phi(c,a^0).
\]

因此对任意状态都有：

\[
A_\phi(c,a^0)=0.
\]

这消除了任意 intercept，并保证“所有 token 都使用固定 OPD”时预测的相对收益为 0。

对 intervention/controller policy \(\pi\) 在一批有效 token 上产生的 actions \(a_{1:N}^{\pi}\)，定义预测的 policy-level utility：

\[
\widehat Y_\phi(\pi)
=
\frac{1}{N}
\sum_{t=1}^{N}
A_\phi(c_t,a_t^\pi).
\]

这里采用 additive contribution 作为 M0 的可检验建模假设，而不是已知事实。若真实收益存在强 token–token、sequence 或 optimizer interaction，该分解可能失败，必须由 held-out intervention 检验。

#### M0 冻结的监督标签

从同一个 student checkpoint \(\theta_k\) 出发，对 controller policy \(\pi\) 和固定 OPD baseline \(\pi_0\) 使用：

- 相同 inner-train prompts；
- 相同 rollout/decoding seeds；
- 相同 optimizer、学习率和 gradient clipping；
- 相同有效 training-token budget；
- 相同 pilot-candidate \(H=20\) optimizer steps；正式 horizon 由 reliability pilot 冻结；
- 相同 teacher 和 evaluation prompts。

分别得到：

\[
\theta_{k+H}^{\pi}
=
U_H(\theta_k;\pi),
\qquad
\theta_{k+H}^{0}
=
U_H(\theta_k;\pi_0).
\]

真实的 policy-level learning-utility label 定义为：

\[
Y(\pi)
=
J_{\mathrm{search\text{-}val}}
\left(\theta_{k+H}^{\pi}\right)
-
J_{\mathrm{search\text{-}val}}
\left(\theta_{k+H}^{0}\right).
\]

\(J\) 使用同一批独立 search-validation questions 和相同 decoding seed list 上的 verifier exact-reward `mean@4`。优先记录逐 question paired difference，再求均值，以降低 evaluation sampling noise。

M0 首版不从 \(Y(\pi)\) 中扣除计算成本，而是通过固定 optimizer steps、有效 training tokens、rollouts 和 evaluation budget保证比较 matched；wall-clock、显存和失败率作为单独诊断。

#### M0 的基础拟合目标

对一组受控 intervention policies \(\{\pi_m\}\)，用 policy-level regression 与 ranking 联合训练：

\[
\mathcal L_S
=
\sum_m
\left(
\widehat Y_\phi(\pi_m)-Y(\pi_m)
\right)^2
+
\gamma\mathcal L_{\mathrm{rank}}.
\]

\(\mathcal L_{\mathrm{rank}}\) 要求当 \(Y(\pi_i)>Y(\pi_j)\) 时，模型也倾向于预测
\(\widehat Y_\phi(\pi_i)>\widehat Y_\phi(\pi_j)\)。

必须明确：

- 监督标签是 **policy-level**，不是直接观测的 token label；
- token advantage 来自跨大量受控 intervention 的弱监督/credit assignment；
- 需要在未用于拟合 \(S\) 的新 intervention policies 上检验排序和 utility prediction；
- 如果不同 token allocations 得到相似的整体 label，单 token attribution 可能不可识别；
- 若 additive model 无法预测 held-out policy 排序，应升级 aggregation/interaction model，而不是把任意 token 分数解释为真实贡献。

#### M0 候选的 verifier 架构

以下 DeepSets + MLP 是 M0 的候选扩展架构，不使用 LLM、raw-text encoder 或大型 Transformer。但由于独立监督单位是 intervention run/bag，而不是 bag 内 token，正式启用前还必须通过下文的 linear-first 可识别性门。

首先对 top-k union support 中每个元素编码：

\[
\xi_t(v)=\phi_{\mathrm{item}}(e_t(v)),
\]

再使用 permutation-invariant pooling：

\[
g_t^{\mathrm{dist}}
=
\rho_{\mathrm{set}}
\left(
\operatorname{mean}_{v\in\mathcal U_t}\xi_t(v),\,
\operatorname{max}_{v\in\mathcal U_t}\xi_t(v),\,
m_{S,t}^{\mathrm{tail}},\,
m_{T,t}^{\mathrm{tail}}
\right).
\]

Stage、rollout 和显式 token features 由另一个 MLP 编码：

\[
g_t^{\mathrm{scalar}}
=
\phi_{\mathrm{scalar}}
\left(
[u_k,r_{q,i},f_t]
\right).
\]

两部分合并为 token-state representation：

\[
h_t
=
\phi_{\mathrm{state}}
\left(
[g_t^{\mathrm{dist}},g_t^{\mathrm{scalar}}]
\right).
\]

七个 hard actions 不只用无结构 action ID，而使用：

\[
e(a)
=
\left[
\mathbb 1_{\mathrm{skip}},\,
\mathbb 1_{\mathrm{FKL}},\,
\mathbb 1_{\mathrm{RKL}},\,
w(a),\,
w(a)^2
\right].
\]

其中 skip 的 weight 为 0。加入 \(w\) 和 \(w^2\) 允许模型表达学习强度的线性与简单非线性效应，同时保留 FKL/RKL 方向。

Action value 使用带 state–action interaction 的小型 MLP：

\[
Q_\phi(c_t,a)
=
\phi_Q
\left(
[h_t,e(a),h_t\odot P e(a)]
\right),
\]

其中 \(P\) 将 action embedding 投影到与 \(h_t\) 相同维度。实际输出仍按 fixed OPD baseline 中心化：

\[
A_\phi(c_t,a)
=
Q_\phi(c_t,a)
-
Q_\phi(c_t,a^0).
\]

该 Level-3 候选网络约束：

- 总参数量保持小型（目标不超过约 \(10^5\) 参数，最终按输入维度调整）；
- 所有 scalar normalization statistics 只由 calibration/probe split 估计并冻结；
- missingness 使用显式 mask；
- 不使用 action 后才产生的信息；
- 不给模型 token-level pseudo-label；
- 不加入 raw token ID embedding，避免第一版变成词汇记忆器；
- student inner update 使用 stop-gradient 后的 action allocation。

#### Bag/policy-level credit assignment

每条 intervention run 是一个 bag：

\[
\mathcal B_m
=
\{(c_t,a_t)\}_{t=1}^{N_m},
\]

只有一个整体标签 \(Y_m\)。训练时对 bag 内 token advantages 求均值：

\[
\widehat Y_m
=
\frac{1}{N_m}
\sum_{t=1}^{N_m}
A_\phi(c_t,a_t).
\]

再用前文的 regression + ranking loss 拟合 \(Y_m\)。若一个 bag 的 token 数过大，可对 `(checkpoint, outcome, action, entropy quartile, overlap quartile)` 分层抽样，并使用 inverse-probability weights 保持均值无偏；不能简单均匀抽少量 token 而丢失稀有 action/state。

为防止“一个 policy 一个 label”造成的任意 token attribution：

- baseline centering 强制 \(A(c,a^0)=0\)；
- 使用随机与 sign-flipped interventions 增加 action/state 变化；
- 对 \(Q\) 使用 weight decay，并限制网络规模；
- 主要验收是 unseen intervention policy 的整体 utility 与排序；
- token-level explanation 只作诊断，除非后续获得更细粒度干预证据；
- 如果 unseen-policy ranking 不成立，则视为 credit assignment 失败。

#### 部署时的闭环

对当前 global optimizer batch 的每个有效 token，模型计算七个 actions 的
\(A_\phi(c_t,a)\)，再用前文的 exact-budget constrained allocator 选择：

\[
(a_1^*,\ldots,a_N^*)
=
\arg\max
\sum_t A_\phi(c_t,a_t)
\quad
\text{s.t.}\quad
\sum_t w(a_t)=N.
\]

选出的 actions 使用 stop-gradient 进入 FKL/RKL/OPD loss，更新 student；\(S\) 本身不通过这条 inner student-loss 路径更新。

#### Formalization consistency audit

##### Audit 1：Additive token utility

当前：

\[
\widehat Y(\pi)
=
\frac1N\sum_t A(c_t,a_t)
\]

是假设 token contributions 在给定 state/action 后可以一阶相加。真实 optimizer update 可能包含同一 sequence 内的 token interaction、gradient interference、action concentration 与 batch normalization 交互，以及 FKL/RKL 在共享参数上的非线性组合。

M0 可以保留 additive model 作为最低复杂度假设，但必须加入专门的 falsification policies：

1. 保持 action counts/总 weight 相同，把非-baseline actions 集中在少数 sequences；
2. 保持 action counts/总 weight 相同，把它们均匀分散到多条 sequences；
3. 保持 marginal feature/action 统计近似一致，只改变同一 sequence 内的 action 共现。

若 additive model 在这些 matched policies 上出现系统性 residual，或不能预测其排序，则先升级为 sequence-level pooling/interaction model，而不是继续把 token 分数解释成独立真实贡献。

##### Audit 2：\(H=20\) label reliability

\(H=20\) 是当前预算候选，不应在 pilot 前视为已证明可靠。Phase 0 需要把 label noise 分成：

- evaluation sampling noise：同一 trained checkpoint 使用不同 evaluation seeds；
- optimization noise：同一 policy 从同一起点使用不同 training seeds；
- checkpoint/stage variation：同一 policy 在不同 student stages 的真实异质性。

Pilot 至少检查：

- 同一 policy 的 seed-replicate correlation/rank stability；
- paired per-question difference 的置信区间；
- policy effect 方差与 seed/evaluation noise 的比例；
- \(H=5,10,20\) 的方向是否一致。

如果 \(H=20\) 的 policy 排序在重复实验中不稳定，则不能用它训练 \(S\)。应在读取方向性结论前预注册选择：增加 matched seeds、增大 \(H\)、使用 \(H=5,10,20\) learning-curve AUC，或在短期指标与长期 reward 不一致时停止该目标。因此 `H=20` 的当前状态是 **pilot candidate**。

##### Audit 3：Policy-level labels 对 token action value 的可识别性

一条 intervention run 即使包含百万 token，也只提供一个独立 \(Y_m\)。独立监督样本量近似为：

\[
N_{\mathrm{supervision}}
\approx
\#\{\text{checkpoint}\times\text{seed}\times\text{policy runs}\},
\]

而不是 token 数。Bag 内 token 共享同一个 label，不能当作独立样本扩增。

因此 M0 冻结为 **linear-first capacity ladder**：

###### Level 0：global-action baseline

只学习每个 action 的全局平均效果，不看 state：

\[
A(c,a)=\beta_a-\beta_{a^0}.
\]

###### Level 1：regularized linear state–action model

使用预注册的低维显式 state features \(\widetilde c\)：

\[
Q(c,a)
=
\beta_a+\widetilde c^\top W e(a),
\]

\[
A(c,a)=Q(c,a)-Q(c,a^0).
\]

采用 ridge/elastic-net regularization；feature 集和正则搜索范围在读取 held-out labels 前冻结。

###### Level 2：small nonlinear feature model

只有当 Level 1 在 unseen policies 上稳定优于 Level 0，且 bag-level learning curve 与 validation variance 支持增加容量时，才使用小型 MLP 处理显式 features。

###### Level 3：top-k DeepSets + MLP

只有当 Level 2 已通过，并且 top-k/combined input 在严格 unseen-policy validation 上提供可重复增益时，才启用前文的 learnable set encoder。参数量不能仅凭“低于 \(10^5\)”自动视为安全，必须由 bag-level learning curve 与 regularization sensitivity 支持。

每升一级都必须满足：

- unseen-policy ranking 改善；
- held-out-stage 表现不恶化；
- calibration 不明显变差；
- 不依赖单一 policy family；
- 多个 random initializations/regularization settings 结论一致。

若 Level 1 无法超过 Level 0，结论是当前 intervention labels 不足以识别 state-dependent action value；不能直接跳到更复杂网络。

#### M0 冻结的训练、冻结与闭环流程

M0 采用 **offline-train, freeze, then control**。在第一次闭环验证完成前，不在线更新 \(S_\phi\)，也不让 closed-loop student run 的 search-validation 结果回流到 \(S\)。

##### Phase 0：label/noise pilot

目的不是训练 verifier，而是估计候选 20-step learning-utility label 的可重复性、成本和最小可辨别效应。

从一个预注册的中期 student checkpoint 出发，至少选择：

- fixed OPD baseline；
- fixed FKL；
- state-independent random direction；
- random budget-balanced weight；
- 一对 feature policy / sign-flipped counter-policy；
- 一对 random-linear policy / counter-policy。

每个 policy 使用至少两个 matched training seeds。记录每个 policy 的：

- \(Y(\pi)\) 及 paired per-question differences；
- seed 间方差；
- 训练/评测失败率；
- action coverage；
- wall-clock 和显存；
- policy effect 相对 label noise 的比例。

Pilot 后才根据观测到的 \(\operatorname{Var}[Y]\)、运行成本和目标最小效应决定正式 intervention policy 数与 seeds。Pilot 本身不用于报告 verifier 已学会 token utility。

###### Phase 0 最小 run matrix

Phase 0 起始 checkpoint 冻结为现有、具有明确 snapshot hash 的初始 student：

- `DeepSeek-R1-Distill-Qwen-1.5B`
- snapshot `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`

不使用旧 baseline 的 step-150 checkpoint：旧实验报告能证明它存在，但当前材料不足以证明其 optimizer state、clean code provenance 和完整 hash 均满足新 M0。Phase 0 的目标是测量 label/implementation reliability，不负责验证 stage generalization；正式 Phase 1 再从同一 clean OPD baseline 生成并冻结 early/mid/late checkpoints。

Teacher 冻结为：

- `JustRL-DeepSeek-1.5B`
- snapshot `0637e4096c789c67f9eecbe8355e0bdeddede1c2`

Student/teacher tokenizer hash 均为
`8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5`。

每个 policy 使用两个 matched training seeds \(s_0,s_1\)，共：

\[
9\ \text{policies}\times2\ \text{seeds}=18\ \text{short-training runs}.
\]

Phase 0 冻结：

- training seeds：`0, 1`；
- evaluation seed list \(E_0\)：`0, 1, 2, 3`；
- evaluation seed list \(E_1\)：`4, 5, 6, 7`；
- calibration prompts：64；
- inner-train prompts：128；
- search-validation prompts：64；
- 每个 evaluation prompt 每个 seed list 生成 4 个 rollouts；
- AIME24/25/26 保持 locked，不进入 Phase 0。

Phase 0 的训练预算为 18 条 run、最多 `18 × 20 = 360` 个 optimizer-run steps。若三组 horizon checkpoints 全部完成，search-validation evaluation 预算为：

\[
18\times3\times2\times64\times4
=
27{,}648
\]

条 response。Pilot 开始后不能因看到某个 policy 的方向而删减其 counter-policy 或额外增加有利候选。

| ID | Policy | Token actions | 主要目的 |
|---|---|---|---|
| P0 | Fixed OPD control | 所有 token `(RKL/OPD, 1)` | 共同 baseline |
| P1 | Fixed FKL control | 所有 token `(FKL, 1)` | 全局方向效应 |
| P2 | Random direction | FKL/RKL 各 1/2，全部 \(w=1\) | state-independent 方向对照 |
| P3 | Random weight | RKL；`skip/2` 各 1/2 | skip/upweight 与极端预算重分配 |
| P4 | Entropy-conditioned | 高 student entropy → FKL1；低 entropy → RKL1 | 一个 state-conditioned hypothesis |
| P5 | Entropy counter-policy | 高 entropy → RKL1；低 entropy → FKL1 | P4 的 sign-flipped 对照 |
| P6 | Sequence-concentrated | 一半 sequences 全 FKL1；一半全 RKL1 | interaction/additivity 测试 |
| P7 | Sequence-dispersed | 每条 sequence 内 FKL1/RKL1 各一半 | 与 P6 保持总体 action counts matched |
| P8 | Full-action random | 七个 hard actions 各约 1/7 | 七动作 coverage 与 allocator smoke test |

七个 hard actions 的 weights 恰好满足：

\[
\frac{0+0.5+1+2+0.5+1+2}{7}=1,
\]

因此 P8 可在 action 数量平衡时满足 global mean-weight=1。实际 token 数不能被 7 整除时，由 exact-budget allocator 处理余数。

随机 assignment 使用稳定 hash：

\[
\operatorname{hash}
\left(
\text{prompt ID},
\text{rollout ID},
\text{token position},
\text{policy seed}
\right),
\]

不能依赖进程顺序或非记录 RNG。P4/P5 使用每个 global optimizer batch 内 student entropy 的 matched quantile split；P6/P7 必须记录并核对实际 FKL/RKL counts、总 weight 和 outcome/length 分布。

###### Horizon 与 evaluation 设计

同一条 run 在 optimizer steps \(5,10,20\) 保存 checkpoint：

\[
H\in\{5,10,20\}.
\]

每个 checkpoint 使用两套预注册、互斥的 evaluation decoding-seed lists
\(E_0,E_1\) 运行相同 search-validation prompts。这样：

- \(E_0\) vs \(E_1\) 估计 evaluation sampling noise；
- \(s_0\) vs \(s_1\) 估计 optimization/training-seed noise；
- step 5/10/20 比较 horizon reliability；
- 不增加短训练 run 数，只增加 checkpoint evaluation。

所有 policies 共用相同 prompt IDs、rollout count、temperature、top-p、token budget 和 evaluation seed lists。

###### Phase 0 preflight

在 18 条 runs 前必须完成：

1. FKL 与 RKL/OPD basis 的数值审计及 RMS/gradient-scale matching；
2. P0 one-step 与当前 OPD baseline 完全一致；
3. P8 one-step 覆盖七动作且 global total weight 精确；
4. P6/P7 one-step action counts 与总 weight matched；
5. 单 run 的 step 5/10/20 checkpoint 和双 evaluation-seed-list 流程跑通；
6. 完成下述 gradient intervention audit，确认动作分配确实改变了 student update；
7. 所有 manifests、hash assignment、seeds 和 failure rules 冻结。

###### Gradient intervention audit：动作是否真正改变更新

一个关键风险是：token actions、权重直方图和 ESS 虽然发生变化，但不同 token
的梯度高度同向，或归一化后的增减彼此抵消，导致 aggregate student gradient
几乎仍等于 fixed OPD。此时“intervention 最终性能没有改善”不能用于判断
verifier、feature 或 action selection 无效，因为实验实际上没有产生足够强的
optimization intervention。

在相同冻结 batch、相同 rollout/token mask、相同模型参数、相同 loss
normalization 下，关闭 dropout 等非必要随机性，并在 FKL/RKL RMS matching
之后、optimizer step 与 gradient clipping 之前，分别计算：

\[
g_0=\nabla_\theta L_{\mathrm{OPD}},\qquad
g_\pi=\nabla_\theta L_\pi .
\]

至少对 P0 重算 control、一个差异最大的健康 intervention，以及一个当前最有希望
的 weighting policy，记录：

\[
C_\pi=
\cos(g_\pi,g_0)
=
\frac{\langle g_\pi,g_0\rangle}
{\lVert g_\pi\rVert\lVert g_0\rVert},
\qquad
D_\pi=
\frac{\lVert g_\pi-g_0\rVert}{\lVert g_0\rVert}.
\]

同时记录全模型及 layer/module-level 的 cosine、relative difference 和 norm
ratio，并与 action/weight histogram、skip/down/baseline/up 比例、ESS、
FKL/RKL raw/scaled RMS、gradient clipping 状态、投影变化和 fallback 次数关联。
正确/错误 rollout 与 student-support coverage 分层也需保留，以判断变化是否只
集中在少量样本。

不要将所有参数梯度拼接成单一向量。逐参数张量累计 dot product 与 squared norm
即可；FSDP/ZeRO 下每个 rank 对本地 shard 累计，最后只 all-reduce 标量。

P0 在相同输入上独立重算一次，得到数值/实现噪声基线
\((1-C_{\mathrm{repeat}},D_{\mathrm{repeat}})\)。正式运行前预注册相对于该噪声基线
的最小 intervention-separation 条件，而不是事后选择任意 cosine 阈值。若所有
健康 intervention 均表现为 \(C_\pi\approx1\) 且
\(D_\pi\) 与 repeat noise 同量级，则 Phase 0 暂停：先增强 action contrast、
检查 token-gradient 共线性与 normalization，或更换能提供更大可学习差异的
teacher；不得把后续 null result 解释为 verifier 已被否证。若方向/幅度已明显
改变但 downstream reward 不变，则优先检查 horizon、evaluation noise 与
credit assignment。

###### Phase 0 reliability gate

对每个 \(H\in\{5,10,20\}\)，计算：

- \(E_0/E_1\) 的 cross-policy Spearman ranking；
- \(s_0/s_1\) 的 cross-policy Spearman ranking；
- paired per-question effect 与 confidence interval；
- non-baseline policies 相对 P0 的方向一致率；
- policy-effect variance / noise variance；
- P6/P7 的 matched difference。

预注册选择最小满足以下工程门的 \(H\)：

1. evaluation-seed-list ranking Spearman \(\rho\ge0.8\)；
2. training-seed ranking Spearman \(\rho\ge0.5\)；
3. 没有由 crash、长度、训练 token 数或 gradient scale 驱动的 policy 排序；
4. 至少存在若干 effect 大于 paired noise floor 的 policies，能够支持后续拟合。

如果 step 5 已通过，则选择 \(H=5\) 以降低正式数据成本；否则依次检查 10、20。若 \(H=20\) 仍未通过，不进入 verifier fitting，应增加 seeds、延长 horizon 或修改 utility target。

P6/P7 若在 matched counts 下差异超过 paired noise floor，说明 sequence-level action interaction 不可忽略；正式模型至少需要 sequence-level aggregation，不能使用纯 token additive verifier。

##### Phase 1：正式 intervention bags

至少覆盖 early/mid/late student stages。对每个 `(checkpoint, training seed)`：

1. 运行一条 matched fixed-OPD baseline；
2. 运行预注册的 fixed/random/single-feature/random-linear policies；
3. 保存每条 run 的 token states、executed actions、coverage table、训练 provenance 与 \(Y(\pi)\)；
4. 对失败 run 记录失败类型，不用事后替换 policy。

正式样本量不在看到 labels 后随意增加特定方向的 policies；若使用分批扩充，下一批的生成规则和停止条件必须在读取其 labels 前冻结。

##### Phase 2：offline fit

按 intervention policy 和 checkpoint 拆分：

- meta-train：拟合 \(S_\phi\)；
- meta-validation：选择网络规模、regularization、ranking-loss weight 和训练轮数；
- held-out stage：检验 student-stage generalization。

训练期间只能访问 meta-train bags。Early stopping 和所有模型选择只看 meta-validation，不能看 held-out stage 或 locked downstream test。

至少报告：

- policy utility RMSE/MAE；
- Spearman/Kendall policy ranking；
- pairwise ranking accuracy；
- calibration plot；
- 不同 action/state strata 的 coverage；
- 与简单 baselines 的比较：
  - 预测所有 policies 与 OPD 相同；
  - 只按全局平均 action 效果排序；
  - 只用 step；
  - explicit-features only；
  - top-k only；
  - combined input。

##### Phase 3：freeze

在闭环前冻结：

- \(S_\phi\) checkpoint/hash；
- input normalization；
- action set；
- exact-budget allocator；
- missing-value policy；
- student starting checkpoint；
- inner-training config、prompts 和 seeds；
- search-validation 与 locked-test manifests；
- failure/continuation rules。

冻结后不得根据 closed-loop 结果重新选择 verifier checkpoint。

##### Phase 4：unseen closed-loop test

从一个未参与拟合 \(S\) 的 student checkpoint/stage 出发，比较：

1. fixed OPD；
2. fixed FKL；
3. 最佳预注册 fixed/heuristic policy；
4. frozen \(S\) + constrained allocator。

所有方法使用 matched training/evaluation seeds、Phase 0 后冻结的相同 horizon \(H\) 和相同有效 token budget。先只使用 search-validation 判断 M0 是否通过 continuation gate；locked downstream test 只在所有选择完全冻结后评测一次。

##### M0 continuation gate

只有同时满足以下条件才继续扩展：

1. unseen-policy ranking 明显优于 constant/global-action baselines；
2. held-out-stage closed-loop policy 的 paired learning utility 为正，并超过 Phase 0 预注册的最小可辨别效应；
3. action allocation 不塌缩为所有 token 同一个 action；
4. global weight budget、loss-scale matching 和训练健康检查全部通过；
5. 改善不能仅由 response length、梯度尺度、训练 token 数或失败 run 选择解释。

若 ranking 有效但 closed-loop 不增益，保留 \(S\) 作为诊断模型并重新检查 allocator/additive assumption。若 ranking 本身无效，不扩大模型或进入 question/teacher level，优先检查 label noise、coverage、credit assignment 和 state definition。

### 4.4 为什么不能直接把现有 outcome verifier 当作 \(S\)

答案正确性 verifier 回答的是“当前 response 是否正确”；这里需要回答的是“在当前 student 状态下，执行这个学习动作是否会改善未来 student”。

Teacher 正确、teacher confidence 高、当前 KL 下降或单步 loss 下降，都不必然意味着下游能力提高。因此这里的 \(S\) 更接近：

- learning-utility verifier；
- meta-critic；
- treatment-effect predictor；
- contextual bandit value function。

命名尚未决定，但应避免与普通 answer verifier 混淆。

## 5. 三层决策如何统一

### 5.1 不同层级的决策时间

三个层级不能共享完全相同的可观测信息：

- **Question selection**：发生在采样/rollout 之前，只能使用当前 student stage、question 本身、历史统计、既有 difficulty/pass-rate 或低成本预估信息；不能使用本次尚未生成的 rollout outcome。
- **Teacher selection**：可根据具体设计发生在 rollout 前，或在 student rollout 完成后、teacher forward 前；其可见信息必须按实际调用位置单独冻结。
- **Token action**：M0 冻结在 student rollout 完成、outcome verifier 已返回结果、teacher 在 student trajectory 上完成 forward 之后，student update 之前。

因此 token-level M0 的执行顺序是：

\[
\text{question}
\rightarrow
\text{student rollout}
\rightarrow
\text{outcome reward}
\rightarrow
p_S,p_T
\rightarrow
S_\phi(c_t,a_t)
\rightarrow
\text{token action allocation}
\rightarrow
\text{student update}.
\]

Token-level \(S\) 可以合法使用完整 rollout 的最终正确性、response-level verifier reward、token/prefix 信息和师生分布。该定义不暗示未来的 question-level selector 也能使用这些信息。

一个概念上的层次化分解是：

\[
x^*
=
\arg\max_x S_Q(c_k,x),
\]

\[
T^*
=
\arg\max_T S_T(c_k,x^*,T),
\]

\[
(m_t^*,w_t^*)
=
\arg\max_{m,w}
S_{\mathrm{tok}}(c_{k,t},m,w).
\]

但尚未决定最终应采用：

1. 一个共享的 \(S_\phi(c,a)\)；
2. 三个分别训练的 \(S_Q,S_T,S_{\mathrm{tok}}\)；
3. 共享 encoder + 三个 action head；
4. 自上而下的 hierarchical policy；
5. 先 token、再 question、最后 teacher 的逐阶段扩展。

单一 verifier 形式统一，但存在数据归因和尺度不一致问题；三个 verifier 更易实现，却可能丢失 question–teacher–token 的交互。

## 6. 与现有 M0 的关系

现有 `OPD_dynamic_loss_search_M0.md` 已经定义了一个较窄的 token-level 起点：

- 固定 question distribution 和 teacher；
- 输入只使用由 student/teacher 分布计算的 feature；
- 原 M0 的实现级 search space 区分 OPD surrogate、FKL、top-k RKL、JS 及其 mixture；在新的概念层 formalization 中，OPD 归入 RKL action，是否保留不同 RKL surrogate 作为实现变体需单独审计；
- 每个候选从同一 checkpoint 进行相同预算的短训练；
- 使用独立 search-validation 上的 exact-reward mean@4 排序；
- locked AIME test 不参与搜索；
- 不搜索任意 gradient，而搜索 PPO-compatible scalar surrogate。

它可以视为新的三层构想中的 **token-level M0**，但不是完整答案。尤其需要重新讨论：

- M0 当前是直接搜索 loss gate \(F_\phi\)，还是应该显式学习 action-value verifier \(S_\phi(c,a)\)；
- \(S\) 是只负责评估候选 action，还是也直接输出 action；
- 短训练产生的整体 \(\Delta J\) 如何归因给 batch 内的 question/token/action；
- 黑盒候选搜索、监督学习、meta-gradient 和在线 bandit 应如何衔接。

## 7. 这个决策器可能如何学到

以下是待比较的路线，不是已决定方案。

### 路线 A：受控干预数据 + 监督学习

从同一 student checkpoint 出发，随机或均衡分配不同 action，保持 prompts、seed、预算和 optimizer 一致，运行短训练并测量 held-out \(\Delta J\)。用：

\[
(c,a,\Delta J)
\]

训练 \(S_\phi\)。

优点：定义清楚、容易审计。  
难点：每个 label 需要真实训练，成本高；batch-level reward 很难归因到 token。

#### M0 冻结的 intervention-data protocol

这里的 **intervention data（受控动作实验数据）** 不是一个现成的数据集。它是从同一个 student checkpoint 出发，在其他训练条件尽量相同的情况下，主动改变 token actions，并记录短训练后的 held-out learning gain 所得到的实验记录。每一条 policy-level 记录至少包含：

\[
\left(
\text{start checkpoint},\,
\{c_t\}_{t=1}^{N},\,
\{a_t\}_{t=1}^{N},\,
\text{training config/seeds},\,
Y(\pi)
\right).
\]

例如，从同一 checkpoint 分出四条 matched 短训练：

1. 所有 token 使用固定 OPD；
2. 高 entropy token 使用 FKL，其余使用 OPD；
3. 把第 2 条的方向反过来；
4. 在满足总预算的条件下随机分配 FKL/RKL。

比较四条 run 的 held-out reward 变化，才能判断“高 entropy 时使用 FKL”是否带来学习收益。只读取历史日志中“高 entropy token 往往出现在哪里”属于 observational correlation，不能区分 feature 本身是否真的指导了更好的 action。

这里不要求四条 run 两两比较，也不要求为每个候选重复运行一份新的 baseline。对同一个起始 checkpoint、training seed 和 matched config，一条固定 OPD Run A 可以作为多个 intervention policies 共用的 control：

\[
Y(\pi_B)=J(\pi_B)-J(\pi_A),\quad
Y(\pi_C)=J(\pi_C)-J(\pi_A),\quad
Y(\pi_D)=J(\pi_D)-J(\pi_A).
\]

若研究目标只是检验一个固定 heuristic，Run A + Run B 已经足够。只有当目标是学习能跨 state/action 泛化的 \(S(c,a)\) 时，才需要多个不同的 B/C/D policies 提供输入变化；否则所有非-baseline actions 都没有足够监督，或只能学到一个固定 policy 的整体好坏。

M0 的 intervention policy 不由待训练的 \(S_\phi\) 自适应生成，而是在观察 learning-utility labels 前预注册并冻结。这样避免 controller 一边看 search-validation 结果一边改变数据分布。

从多个预注册 student checkpoints \(\theta_{k_1},\ldots,\theta_{k_s}\) 出发，每个 checkpoint 使用相同的数据拆分、teacher、optimizer、训练预算和 evaluation seed list，运行以下 policy families：

1. **Fixed controls**
   - 所有有效 token 使用 `(RKL/OPD, 1)`；
   - 所有有效 token 使用 `(FKL, 1)`。

2. **State-independent randomized controls**
   - 在 FKL/RKL 方向间做与 state 无关的平衡随机分配，weight 固定为 1；
   - 在满足 mean-weight=1 的前提下随机分配离散 weights，例如：
     - `0/2` 各占 1/2；
     - `0.5/2` 分别占 2/3 与 1/3；
     - `0/0.5/2` 分别占 0.2/0.4/0.4；
   - direction 与 weight 的随机种子预注册。

3. **Single-feature interventions**
   - 分别按 student entropy、teacher entropy、FKL/RKL、overlap、outcome、position 等单一 feature 排序；
   - 对高/低区间交换 FKL/RKL 或 high/low weights；
   - 每个 policy 同时保留其 sign-flipped/counter policy，避免只测试人为相信的方向。

4. **Random linear policies**
   - 对标准化 state features 使用预注册随机参数：

     \[
     s_a(c_t)=W_a^\top c_t+b_a;
     \]

   - 用 batch-level constrained allocator 从七个 hard actions 中选择；
   - 参数文件、随机 seed 和 action coverage 在运行前冻结。

每个 intervention policy \(\pi_m\) 必须从同一个 checkpoint 分叉，并与该 checkpoint 对应的 fixed-OPD baseline 形成 matched pair。Policy-level label 使用前文定义的 \(Y(\pi_m)\)。

#### Coverage/positivity 要求

为了使 \(S(c,a)\) 能比较 action，而不是记住某个 action 只在某类状态出现，数据生成必须检查：

- 七个 actions 在总体数据中都有非零支持；
- FKL/RKL 在 correct/incorrect rollout、entropy/overlap quartiles、early/mid/late checkpoints 中都出现；
- 每个主要 state stratum 内至少有两个以上 actions；
- action 与 policy family、checkpoint、seed 的列联表可审计；
- 若某 action 在某状态区域从未执行，则不得声称 \(S\) 能估计该区域的反事实价值。

#### 数据拆分

拆分单位必须是 **intervention policy 和 checkpoint**，而不是随机拆 token：

- `meta-train`：拟合 \(S\) 的 intervention policies；
- `meta-validation`：未参与拟合的新 intervention policies；
- `stage-generalization test`：至少一个未用于拟合的 student checkpoint/stage；
- `locked downstream test`：仅在 verifier、allocator 和所有超参数冻结后使用。

随机拆 token 会让同一个 policy-level label 同时出现在 train/validation，构成严重泄漏。

#### Pilot 与正式数据的边界

第一轮 pilot 只需要证明：

- 七个 hard actions 和 global budget allocator 可运行；
- fixed/random/feature-conditioned policies 能产生不同但健康的 action distributions；
- matched OPD baseline 与候选 \(H=20\) label pipeline 可复现；
- provenance、coverage 和数据拆分能完整记录。

少量 pilot policies 不能支持“已经学会 token utility verifier”的结论。正式拟合 \(S\) 前，应根据一次短训练的 reward noise、运行成本和 policy-level effect variance 决定 intervention policy 数与 seeds，而不是先拍脑袋固定样本量。

### 路线 B：黑盒搜索 / contextual bandit

把 action-conditioned gate 当作 policy，根据短训练后的 reward 更新候选分布，先验证能否在受约束 action space 中找到优于 fixed OPD 的策略。

优点：不需要精确 token label。  
难点：样本效率低，容易被 seed、loss scale、长度或 checkpoint selection 噪声误导。

### 路线 C：双层优化 / meta-gradient

通过内层 student update 对外层 held-out reward 求 meta-gradient：

\[
\phi^*
=
\arg\max_\phi
J\bigl(U_H(\theta_k;S_\phi)\bigr).
\]

优点：目标与最终 reward 直接对齐。  
难点：长 horizon、LLM 内层更新和离散 sampling 使梯度昂贵且不稳定。

### 路线 D：低成本 proxy 预训练 + 真实 reward 校准

先用 gradient alignment、held-out KL reduction、student correctness transition 等 proxy 预训练 \(S\)，再用少量真实短训练 \(\Delta J\) 校准。

优点：可能降低数据成本。  
难点：proxy 与真正下游能力提升是否一致必须验证，不能直接当最终目标。

## 8. 最重要的可识别性问题

后续 formalization 必须正面处理以下问题：

1. **反事实不可见**：同一 student 状态下，执行动作 \(a\) 后就看不到其他动作的真实结果；如何构造 matched interventions？
2. **credit assignment**：短训练后的整体 reward 改变，如何归因到某个 question、teacher 或 token？
3. **non-stationarity**：student 在持续变化，同一 feature/action 在不同训练阶段可能有不同作用。
4. **action scale confound**：不同 loss/operator 的梯度尺度不同，不能让某个 action 仅靠更新更大而获胜。
5. **sampling confound**：question、teacher 和 token action 同时变化时，无法知道收益来自哪一层。
6. **selection leakage**：search-validation、checkpoint selection 和最终 test 必须隔离。
7. **cost–quality tradeoff**：更强 teacher、更多 rollout 或更长训练带来的收益是否值得额外成本？
8. **uncertainty**：当 \(S\) 不确定时，是探索、回退到 OPD baseline，还是跳过该样本？
9. **support/generalization**：训练 \(S\) 时没见过的 student checkpoint、teacher 或问题类型能否泛化？
10. **stability/safety**：动态策略是否导致长度膨胀、模式坍缩、梯度爆炸或只优化短期 reward？

## 9. 早期共识（历史快照）

> 本节记录 formalization 过程中的早期共识。其后已经冻结的 M0 决策以本文前面的 “M0 冻结” 小节和
> `OPD_verifier_M0_spec_2026-07-30.md` 为准；二者冲突时，specification 是唯一实施依据。

1. 固定 feature heuristic 不是最终目标；feature 是候选状态输入。
2. 研究对象应覆盖 question、teacher 和 token 三个层级。
3. Verifier 至少在概念上应当 action-conditioned，否则无法比较 FKL/RKL/skip 等动作。
4. “学习质量”不能简单等同于当前答案正确率、teacher confidence、KL 或训练 loss。
5. Held-out downstream reward 的未来增量是首选目标；当时 horizon、成本和归因方式未定，现已冻结为 matched fixed-OPD policy label，并在 \(H\in\{5,10,20\}\) 中仅按 reliability 选择。
6. 第一版应限制 action space，保留 OPD baseline，并避免同时搜索所有三层。
7. 现有 token-level M0 是可复用起点，不是最终 formalization。

## 10. 早期开放问题及其当前处置

> 以下问题保留用于说明设计空间，不是 RA 可以重新默认为“未决定”的 implementation options。
> Token M0 已在 specification 中冻结输入、七动作、固定权重预算、policy-level label、离线拟合和
> Phase 0 gates。Question/teacher-level 问题仍延后；只有 Phase 0/1 证据要求修改时才重新打开已冻结项。

### A. 输入/state

1. 最小充分 state 是什么？
2. 如何表示“当前 student 所处训练阶段”？
3. Question 原文是否直接输入，还是只输入 difficulty/pass-rate/embedding 等摘要？
4. Teacher identity 是否必要，还是 teacher distribution 已足够？
5. Token-level \(S\) 是否允许看到 outcome、future tokens 或最终正确性？若训练时可看但部署时不可看，应如何处理？
6. 哪些 feature 是在线免费、哪些需要额外 teacher forward 或多次 rollout？

### B. action/output

1. \(S\) 输出 action value、ranking score、mixture weights，还是直接输出 policy？
2. Token action 是否采用 \((w_t^F,w_t^R)\)，并以 \((0,0)\) 表示 skip？
3. \(w_t^F,w_t^R\) 是连续值还是少量离散档位；其上界、稀疏性和 batch-level 总权重如何约束？
4. Question/teacher/token 是否共享同一量纲的 utility？
5. 是否输出 uncertainty、cost 和 failure risk？

### C. target/measurement

1. \(J\) 应该是 exact reward mean、pass rate、majority accuracy，还是多任务综合？
2. \(H\) 取 1 step、20 steps、一个 checkpoint interval，还是训练终点？
3. 使用 \(\Delta J\)、相对 OPD 的 \(\Delta J\)，还是单位训练成本收益？
4. 如何降低 evaluation sampling noise？
5. 如何避免短期 reward 改善损害长期训练？

### D. learning algorithm

1. 首版选择受控干预监督学习、bandit、黑盒搜索、meta-gradient，还是 proxy→校准？
2. 如何生成足够覆盖、同时成本可控的 action intervention 数据？
3. Batch reward 如何分配给 question/token？
4. 是否先只验证排序能力，而不让 \(S\) 直接控制在线训练？
5. 何时从离线 \(S\) 切换到在线更新？

### E. 实验与证伪

1. 最小实验怎样证明“学到的是条件化决策”，而不是平均偏好某个 loss？
2. 应选择哪些 fixed baselines：OPD、FKL、RKL、JS、uniform mixture、旧 feature heuristics？
3. 怎样验证同一 feature 在不同 student stage 上需要不同 action？
4. 哪些结果会证伪当前 action space、state design 或短 horizon 目标？
5. 怎样做 train/search-validation/locked-test 隔离和 matched-seed 比较？

## 11. 已完成的最小产出

下列原定讨论产出已经由 handoff、M0 specification、RA checklist 和 red-team audit 覆盖：

1. 一张变量表：state、action、output、target、部署时是否可见、计算成本；
2. 一个严格的 \(S_\phi(c,a)\) 数学定义；
3. 一个明确的 held-out learning-utility label；
4. 三种可行学习方法的比较和首选理由；
5. 一个只做 token-level 的最小可证伪实验；
6. 从 token 层扩展到 question/teacher 层的接口，而非立即联合搜索。

## 12. 可直接发给 GPT 的提示词

```text
我们正在 formalize 一个用于 On-Policy Distillation (OPD) 的动态学习决策器。

背景：
- 过去我们分别尝试 student entropy、teacher confidence/entropy、top-k overlap、
  FiRe、symmetric KL、EOPD gate 等固定 feature/heuristic。
- 结果不稳定：overlap 只有很小信号；student entropy 的最佳方向随测试集变化；
  强 FiRe 明显变差；很多实验只有单 seed。
- 因此我们不再假设存在一个全局固定 feature rule，而希望把这些 feature 作为
  当前学习状态的输入。

目标包括三个层级：
1. question-level：当前 student 应该采样什么问题；
2. teacher-level：当前 student 最适合向哪个 teacher 学；
3. token-level：哪些 token 多学/少学/跳过，以及如何分配 FKL/RKL 权重；当前把 OPD 视为 RKL 的 on-policy 实现/基线。

初始想法是：
S(question, token, student distribution, teacher distribution) -> R

当前讨论认为这个定义缺少 action。候选形式为：
S_phi(c, a) -> expected learning utility

其中 c 包含当前训练阶段、question、teacher、prefix/token、student/teacher
distribution 以及 entropy、KL、overlap、correctness 等 feature；a 包含
question sampling、teacher choice，以及 token-level 非负权重
(w_FKL, w_RKL)；二者都为 0 表示 skip，当前 OPD 归入 RKL。

当前 token M0 已冻结的监督目标是 policy 相对 matched fixed-OPD baseline 的
search-validation learning-utility：
Y(pi;H) = J(theta_{k+H}^{pi}) - J(theta_{k+H}^{OPD})。
H 在 {5,10,20} 中只按 Phase 0 reliability 选择，不按效果方向选择；M0 不从 Y
中扣除成本，而通过 matched budget 控制。普通 answer verifier 只判断当前答案
对错；这里的 S 预测学习动作对未来 student 能力的相对价值。

现有 token-level M0 已有以下起点：
- 固定 question distribution 和 teacher；
- hard action space 为 skip、FKL×{0.5,1,2}、
  RKL/OPD×{0.5,1,2}，并使用固定 global token-weight budget；
- 每个候选从同一 checkpoint 做相同预算的短训练；
- 用独立 search-validation reward 排序；
- locked test 不参与搜索；
- 不搜索任意 gradient，只搜索受约束 PPO-compatible scalar surrogate。

请以研究合作者身份 red-team 当前 token M0，而不是重新发明一套未对齐方案：
1. 检查 state/action/output/label 是否存在数学矛盾；
2. 检查 policy-level bag labels 是否足以识别 linear state-action value；
3. 检查 exact-budget approximate allocator 是否引入系统偏差；
4. 检查 FKL/RKL scale matching、evaluation noise 和数据泄漏；
5. 检查 18-run Phase 0 是否足以完成 reliability 目标；
6. 给出最小必要修正、证伪标准和应保持冻结的部分；
7. 只有 token M0 通过后，再讨论 question/teacher-level 接口。

输出应包含：
- 数学定义；
- 变量与可观测性表；
- 至少三个候选方案的比较；
- 推荐的第一版方案及理由；
- 最小实验、baseline、指标、数据隔离与停止条件；
- 主要失败模式和证伪标准。
```
