# OPD 数学优先的 Token Action-Pair 选择计划

> 日期：2026-07-31  
> 状态：研究计划草案，不代表已经得到实验结论  
> 当前范围：只研究 token-level action；question distribution 与 teacher 暂时固定  
> 核心原则：先从目标函数和模型更新推导可解释的 action score，再用实验验证；不把希望首先寄托在黑盒神经网络上

---

## 1. 研究目标

在 On-Policy Distillation（OPD）训练中，对 student rollout 的每一个有效 token \(t\)，决定：

1. 是否学习该 token；
2. 使用多少 FKL；
3. 使用多少 RKL/OPD；
4. 是否同时组合 FKL 与 RKL；
5. 在整个 optimizer batch 的学习预算固定时，如何联合分配所有 token 的动作。

每个 token 的 action 定义为一个非负 pair：

\[
a_t=(w_{F,t},w_{R,t}),
\qquad
w_{F,t},w_{R,t}\ge 0.
\]

其中：

- \(w_{F,t}\)：该 token 的 FKL 权重；
- \(w_{R,t}\)：该 token 的 RKL/OPD 权重；
- \((0,0)\)：skip；
- \((1,0)\)：标准强度的纯 FKL；
- \((0,1)\)：标准强度的纯 RKL/OPD，也是默认 baseline；
- \((0.5,0.5)\)：总强度为 1 的均匀 mixture；
- \((1,1)\)：总强度为 2 的强 mixture。

目标不是声称能够观测“每个 token 的真实长期最优 action”，而是首先构造一个有数学依据、可计算、可解释、可证伪的局部 utility proxy，并检查它是否能带来真实的未来学习收益。

---

## 2. 为什么不直接训练黑盒 learning-utility verifier

原始设想是把大量 token feature 和候选 action 输入神经网络：

\[
S_\phi(c_t,a_t)
\rightarrow
\text{expected future learning utility}.
\]

其中 \(c_t\) 可以包含：

- student/teacher entropy；
- FKL、RKL；
- top-k overlap；
- student/teacher log-prob；
- token position；
- rollout correctness；
- student training stage；
- 其他分布或训练状态特征。

### 2.1 真正的监督标签不可直接观测

一次短训练可能包含几十万或几百万个 token，但训练结束后只能得到一个整体 label：

\[
Y(\pi)
=
J(\theta_{k+H}^{\pi})
-
J(\theta_{k+H}^{\mathrm{OPD}}).
\]

它表示整套 token action policy \(\pi\) 相对固定 OPD 的整体收益，但不能告诉我们：

- 哪个 token 贡献了收益；
- 哪个 token 有害；
- 收益来自 FKL/RKL 方向还是权重大小；
- 多个 token 的梯度是否相互抵消或增强。

因此，一条 training run 本质上只提供一个独立监督点：

\[
\left(
\{(c_t,a_t)\}_{t=1}^{N},
Y(\pi)
\right).
\]

独立样本数近似为：

\[
\#\{
\text{checkpoint}
\times
\text{policy}
\times
\text{training seed}
\},
\]

而不是 token 数。

例如，9 个 policies、2 个 seeds 只有约 18 个独立 label。它足够用于 pipeline/noise pilot，但不足以识别几百个线性参数，更不足以训练一个大型神经网络。

### 2.2 “训练 loss 收敛”不等于学会 token utility

神经网络可能轻易记住少量 intervention runs，使训练 loss 很低，但它可能实际利用的是：

- policy family；
- checkpoint identity；
- action 的全局平均偏好；
- response length；
- gradient scale；
- seed noise；
- failure-run selection。

换一个 checkpoint、policy family 或数据分布后，预测可能失效。

因此，本计划不把黑盒 verifier 作为第一步。

---

## 3. 数学优先的核心思想

不根据 entropy、KL 等表面 feature 猜测 action 是否有益，而是直接研究：

> 某个 token action 产生的参数更新方向，是否与我们真正想改善的参考目标方向一致。

---

## 4. Reference objective 与数据隔离

准备一组独立的 reference/calibration questions，定义：

\[
L_{\mathrm{ref}}(\theta).
\]

它表示当前 student 在参考任务上的 loss。

计算 reference gradient：

\[
g_{\mathrm{ref}}
=
\nabla_\theta L_{\mathrm{ref}}(\theta).
\]

数据至少拆分为：

1. **inner-train**：真正用于 FKL/RKL/OPD student update；
2. **reference/calibration**：只用于计算 \(g_{\mathrm{ref}}\) 并指导 action；
3. **search-validation**：评价 action policy 的真实短训练 learning gain；
4. **locked test**：所有方法、超参数、checkpoint 和规则冻结后只使用一次。

不能使用 search-validation gradient 选择动作，再在同一 search-validation 上宣称泛化提升。

---

## 5. 一阶 Gradient Alignment 推导

### 5.1 单个动作产生的梯度

对 token \(t\)，定义：

\[
g_{F,t}
=
\nabla_\theta \ell_{F,t}(\theta),
\]

\[
g_{R,t}
=
\nabla_\theta \ell_{R,t}(\theta).
\]

当 action pair 为 \((w_F,w_R)\) 时，token loss 为：

\[
\ell_t(w_F,w_R)
=
w_F\ell_{F,t}
+
w_R\ell_{R,t}.
\]

只要 \(w_F,w_R\) 在 student update 时使用 stop-gradient，组合梯度就是：

\[
g_t(w_F,w_R)
=
w_Fg_{F,t}
+
w_Rg_{R,t}.
\]

### 5.2 一步 student update

若训练梯度为 \(G\)，student 更新为：

\[
\theta'
=
\theta-\eta G.
\]

对 reference loss 做一阶 Taylor 展开：

\[
L_{\mathrm{ref}}(\theta')
\approx
L_{\mathrm{ref}}(\theta)
-
\eta
g_{\mathrm{ref}}^\top G.
\]

因此：

\[
g_{\mathrm{ref}}^\top G
\]

越大，reference loss 预计下降越多。

这里不是判断“梯度为正还是负”。梯度是高维向量，需要用内积判断训练梯度与 reference gradient 是否同向。

### 5.3 Action pair 的一阶分数

定义：

\[
s_t(w_F,w_R)
=
g_{\mathrm{ref}}^\top
\left(
w_Fg_{F,t}
+
w_Rg_{R,t}
\right).
\]

以固定 OPD action

\[
a^0=(0,1)
\]

为 baseline，相对 advantage proxy 定义为：

\[
A_t(w_F,w_R)
=
g_{\mathrm{ref}}^\top
\left[
w_Fg_{F,t}
+
w_Rg_{R,t}
-
g_{R,t}
\right].
\]

解释：

- \(A_t>0\)：局部一阶近似下，该 pair 比固定 OPD 更好；
- \(A_t<0\)：该 pair 比固定 OPD 更差；
- \(A_t=0\)：与固定 OPD 的一阶预测效果相同。

---

## 6. 学习强度与 FKL/RKL 比例

也可以将 pair 参数化为：

\[
w_t=w_{F,t}+w_{R,t},
\]

\[
\lambda_t
=
\frac{w_{F,t}}{w_{F,t}+w_{R,t}},
\qquad
w_t>0.
\]

于是：

\[
w_{F,t}=w_t\lambda_t,
\qquad
w_{R,t}=w_t(1-\lambda_t).
\]

其中：

- \(w_t\)：该 token 总共学多少；
- \(\lambda_t\)：FKL 占比；
- \(1-\lambda_t\)：RKL 占比；
- \(w_t=0\)：skip。

组合 loss 为：

\[
\ell_t
=
w_t
\left[
\lambda_t\ell_{F,t}
+
(1-\lambda_t)\ell_{R,t}
\right].
\]

---

## 7. 重要结论：纯一阶目标通常不会选择 mixture

定义：

\[
b_{F,t}
=
g_{\mathrm{ref}}^\top g_{F,t},
\]

\[
b_{R,t}
=
g_{\mathrm{ref}}^\top g_{R,t}.
\]

固定总强度 \(w_t\) 时，一阶分数为：

\[
s_t(w_t,\lambda_t)
=
w_t
\left[
\lambda_tb_{F,t}
+
(1-\lambda_t)b_{R,t}
\right].
\]

这是关于 \(\lambda_t\) 的线性函数，因此最优值通常在边界：

\[
\lambda_t\in\{0,1\}.
\]

也就是说：

- 若 \(b_{F,t}>b_{R,t}\)，选择纯 FKL；
- 若 \(b_{R,t}>b_{F,t}\)，选择纯 RKL；
- 中间 mixture 通常不会严格优于两个端点。

这是线性优化的数学性质，不是实现错误。

因此，如果希望同一个 token 上的 FKL/RKL mixture 真正可能最优，必须考虑一阶方向之外的因素。

---

## 8. 使 mixture 有意义：稳定性、Trust Region 与二阶信息

### 8.1 带梯度大小惩罚的局部 utility

定义：

\[
U_t(w_F,w_R)
=
g_{\mathrm{ref}}^\top
(w_Fg_{F,t}+w_Rg_{R,t})
-
\frac{\mu}{2}
\left\|
w_Fg_{F,t}+w_Rg_{R,t}
\right\|_M^2.
\]

其中：

- 第一项奖励对 reference objective 有帮助的更新方向；
- 第二项惩罚过大或不稳定的参数更新；
- \(M=I\) 时是普通梯度范数；
- 后续可以考虑 Fisher/Gauss–Newton 等正半定 metric。

Mixture 可能通过组合两种梯度：

- 保留共同的有益方向；
- 抵消各自的有害方向；
- 降低总体更新范数；
- 提高局部稳定性。

### 8.2 一个 mixture 有益的二维例子

若：

\[
g_{\mathrm{ref}}=(1,0),
\]

\[
g_F=(1,1),
\qquad
g_R=(1,-1),
\]

则均匀 mixture：

\[
0.5g_F+0.5g_R=(1,0).
\]

它保留了 reference 需要的第一个方向，同时抵消了第二个方向的扰动。因此在带范数或 trust-region 惩罚的目标下，\((0.5,0.5)\) 可能优于纯 FKL 或纯 RKL。

### 8.3 二阶 Taylor 形式

更完整的局部展开为：

\[
L_{\mathrm{ref}}(\theta-\eta G)
\approx
L_{\mathrm{ref}}(\theta)
-
\eta g_{\mathrm{ref}}^\top G
+
\frac{\eta^2}{2}
G^\top H_{\mathrm{ref}}G.
\]

其中 \(H_{\mathrm{ref}}\) 是 reference loss 的 Hessian。

- 一阶项描述更新方向；
- 二阶项描述曲率与步长风险；
- 二阶项会产生 FKL/RKL 以及不同 token 之间的 interaction。

直接使用完整 Hessian 成本很高，因此第一版优先使用：

1. 普通 gradient-norm penalty；
2. Fisher/Gauss–Newton 正半定近似；
3. 只有前两者有稳定证据后，再考虑 Hessian-vector product 或多步 meta-gradient。

---

## 9. 不能严格地为每个 token 独立求“真实最佳 pair”

所有 token 的实际 batch gradient 是：

\[
G
=
\sum_t
\left(
w_{F,t}g_{F,t}
+
w_{R,t}g_{R,t}
\right).
\]

不同 token 的梯度可能：

- 相互增强；
- 相互抵消；
- 共同产生过大的更新；
- 在 Hessian/Fisher metric 下产生 interaction。

因此，更准确的问题是：

> 联合选择一个 optimizer batch 中所有 token 的 action pairs，使总更新在固定预算下最有利。

最终仍然会给每个 token 一个 pair，但这些 pairs 是联合决定的，不应被解释成彼此完全独立的真实最优处理。

---

## 10. Global Token-Weight Budget

对 global optimizer batch 中的有效 token 集合 \(\mathcal M\)，定义：

\[
w_t=w_{F,t}+w_{R,t}.
\]

冻结约束：

\[
0\le w_t\le 2,
\]

\[
\sum_{t\in\mathcal M}w_t
=
|\mathcal M|.
\]

即平均 token weight 严格为 1。

该约束保证 controller 只能重新分配固定学习预算，不能通过放大整个 batch 的梯度获得优势。

联合优化可以写为：

\[
\max_{\{w_{F,t},w_{R,t}\}}
\mathcal U
\left(
\sum_t
w_{F,t}g_{F,t}
+
w_{R,t}g_{R,t}
\right)
\]

满足：

\[
w_{F,t},w_{R,t}\ge0,
\]

\[
w_{F,t}+w_{R,t}\le2,
\]

\[
\sum_t(w_{F,t}+w_{R,t})=|\mathcal M|.
\]

若采用一阶 additive score，问题较容易分解；若加入完整二阶 interaction，则需要 batch-level quadratic optimization 或其近似。

---

## 11. 第一版 Action-Pair Space

第一版不直接搜索无限多的连续 pair，而使用一个可审计的离散集合：

\[
\mathcal A=
\{
(0,0),
(0.5,0),
(0,0.5),
(1,0),
(0,1),
(0.5,0.5),
(2,0),
(0,2),
(1,1)
\}.
\]

| Pair | 含义 |
|---|---|
| \((0,0)\) | skip |
| \((0.5,0)\) | 少量 FKL |
| \((0,0.5)\) | 少量 RKL |
| \((1,0)\) | 标准 FKL |
| \((0,1)\) | 标准 OPD baseline |
| \((0.5,0.5)\) | 标准强度的均匀 mixture |
| \((2,0)\) | 强 FKL |
| \((0,2)\) | 强 RKL |
| \((1,1)\) | 强 mixture |

该集合仍是候选，需要在实现前审计：

- FKL/RKL loss scale；
- gradient norm；
- 总强度定义；
- \((1,1)\) 与其他权重的可比性；
- 是否需要额外加入 \((0.25,0.75)\)、\((0.75,0.25)\)。

只有离散 mixture 在独立实验中表现出可重复收益后，才扩展至连续 pair。

---

## 12. 解析式 Controller 的推荐流程

### 12.1 直接计算版本

1. 在 reference/calibration questions 上计算 \(g_{\mathrm{ref}}\)；
2. 对 inner-training batch 计算 \(g_{F,t}\) 与 \(g_{R,t}\)；
3. 对每个 token、每个候选 pair 计算：

   \[
   A_t(w_F,w_R)
   \]

   或带稳定性惩罚的：

   \[
   U_t(w_F,w_R);
   \]

4. 在 global mean-weight \(=1\) 的条件下联合选择所有 token pairs；
5. 对选择结果使用 stop-gradient；
6. 更新 student；
7. 在独立 search-validation 上评价真实 learning gain。

### 12.2 计算成本控制

不需要显式保存每个 token 的完整参数梯度。内积：

\[
g_{\mathrm{ref}}^\top g_t(a)
\]

等价于 token loss 沿 \(g_{\mathrm{ref}}\) 的方向导数：

\[
\left.
\frac{d}{d\epsilon}
\ell_t^a(\theta+\epsilon g_{\mathrm{ref}})
\right|_{\epsilon=0}.
\]

可以使用 JVP/VJP、gradient sketch 或受限参数子空间近似。

建议逐级实现：

1. sequence-level；
2. token-group level；
3. per-token discrete pairs；
4. per-token continuous pairs。

若 sequence-level 都没有稳定信号，不进入更昂贵的 per-token 实现。

---

## 13. 可选的神经网络角色：近似数学分数，而非直接猜未来收益

如果在线计算 gradient alignment 太昂贵，可以在部分 checkpoint/batch 上计算解析式 label，收集：

\[
(c_t,a,s_t(a)).
\]

更完整地，可以记录：

\[
(c_t,a,b_{F,t},b_{R,t},n_{F,t},n_{R,t},q_{FR,t}),
\]

其中：

\[
b_{F,t}=g_{\mathrm{ref}}^\top g_{F,t},
\]

\[
b_{R,t}=g_{\mathrm{ref}}^\top g_{R,t},
\]

\[
n_{F,t}=\|g_{F,t}\|^2,
\qquad
n_{R,t}=\|g_{R,t}\|^2,
\]

\[
q_{FR,t}=g_{F,t}^\top g_{R,t}.
\]

因为：

\[
\|w_Fg_F+w_Rg_R\|^2
=
w_F^2n_F
+
w_R^2n_R
+
2w_Fw_Rq_{FR}.
\]

推荐让小模型预测这些可解释量，再由确定性的 constrained optimizer 输出最终 pair：

\[
c_t
\rightarrow
(\widehat b_F,\widehat b_R,\widehat n_F,\widehat n_R,\widehat q_{FR})
\rightarrow
(w_F^*,w_R^*).
\]

不推荐直接训练：

\[
c_t\rightarrow(w_F,w_R),
\]

因为直接输出 pair：

- 不容易保证全局预算；
- 容易利用 loss scale；
- 难以解释 mixture 原因；
- 失败时难以定位问题。

### 13.1 Feature 可能不足

只使用 entropy、KL、overlap 和 log-prob，未必能预测参数空间中的 gradient alignment。

两个 token 可以具有相似的输出分布 feature，但由于 hidden state、上下文和参数 Jacobian 不同，产生完全不同的梯度方向。

因此需要验证：

\[
\text{cheap token features}
\quad\text{能否预测}\quad
\text{gradient statistics}.
\]

如果不能，可能需要加入：

- hidden-state summary；
- gradient norm；
- 低维 gradient sketch；
- sequence representation；
- student stage。

如果低维线性模型无法预测这些量，不应直接扩大为黑盒网络。

---

## 14. 分阶段实验路线

### Phase A：定义与数值审计

目标：确认比较对象本身正确。

必须检查：

1. FKL 与 RKL/OPD 的精确定义；
2. support、top-k、tail mass 和重归一化语义；
3. FKL/RKL loss scale；
4. gradient norm 与 gradient clipping 前后差异；
5. 固定 \((0,1)\) 是否严格复现 OPD baseline；
6. 所有 pair 的总强度是否符合定义；
7. global batch weight 是否精确等于有效 token 数；
8. 分布式训练与 gradient accumulation 是否保持同一预算。

若本阶段未通过，不进入后续 action selection。

### Phase B：Sequence-Level Gradient Alignment

先在整条 response 上比较：

- fixed OPD；
- fixed FKL；
- FKL/RKL mixture；
- alignment-selected sequence action；
- anti-alignment action；
- random matched action。

目的：判断 gradient alignment 是否至少在 sequence 粒度存在信号。

### Phase C：Token-Group Level

按少数预注册分组测试：

- high/low student entropy；
- high/low teacher entropy；
- high/low overlap；
- correct/incorrect rollout；
- early/late token position。

对每组计算 FKL/RKL/pair alignment，并进行 group-level budget allocation。

目的：检查信号是否具有稳定的条件差异，同时控制 per-token 工程成本。

### Phase D：Per-Token Discrete Action Pairs

在 Phase B/C 通过后：

1. 对每个 token 计算离散 pair scores；
2. 使用 global constrained allocator；
3. 比较纯一阶 score 与带 norm/trust penalty 的 score；
4. 检查 mixture 是否被选择；
5. 检查 mixture 的收益是否超过随机与纯方向控制。

### Phase E：Continuous Pair Optimization

只有离散 mixture 显示可重复收益后，扩展到：

\[
w_F,w_R\in[0,2],
\qquad
w_F+w_R\le2.
\]

使用正半定 trust-region/Fisher 近似，使优化问题尽量保持稳定、可解和可审计。

### Phase F：可选的 Learned Approximation

只有满足以下条件时，才训练小模型近似解析式 gradient statistics：

1. 解析式 alignment 与真实 learning gain 稳定相关；
2. 直接计算成本确实过高；
3. cheap features 对 alignment/statistics 有可重复预测能力；
4. linear/ridge model 明显优于 constant/global-action baseline；
5. held-out checkpoint/stage 上仍能泛化。

容量顺序：

1. constant/global-action baseline；
2. ridge linear model；
3. small MLP；
4. 只有严格证据支持时才加入更复杂表示。

---

## 15. 最小 Baselines

至少比较：

1. **Fixed OPD**：所有 token 使用 \((0,1)\)；
2. **Fixed FKL**：所有 token 使用 \((1,0)\)；
3. **Fixed uniform mixture**：所有 token 使用 \((0.5,0.5)\)；
4. **Random budget-matched allocation**；
5. **Norm-only allocation**：只根据梯度大小选，不看 reference alignment；
6. **First-order alignment allocation**；
7. **Anti-alignment allocation**：故意选择低分动作；
8. **Alignment + trust/norm penalty**；
9. **旧 entropy/overlap heuristic**；
10. 若进入学习阶段，再比较 learned approximation。

---

## 16. 主要指标

### 16.1 真实学习结果

- search-validation 上的 paired learning gain；
- exact reward mean；
- per-question paired difference；
- 多个 training/evaluation seeds 的均值和置信区间；
- 不同 horizon \(H\) 下的方向稳定性；
- held-out student stage 的泛化。

### 16.2 Proxy 有效性

- predicted alignment 与 one-step reference-loss change 的相关性；
- predicted alignment 与 \(H\)-step downstream gain 的相关性；
- policy ranking 的 Spearman/Kendall；
- alignment、random、anti-alignment 的有序关系。

期望至少观察到：

\[
Y_{\mathrm{alignment}}
>
Y_{\mathrm{random}}
>
Y_{\mathrm{anti\text{-}alignment}}.
\]

### 16.3 优化健康度

- FKL/RKL gradient norm；
- clipping rate；
- batch total weight；
- action-pair histogram；
- skip、pure FKL、pure RKL、mixture 比例；
- effective sample size；
- response length；
- training loss；
- KL；
- wall-clock、显存和失败率。

---

## 17. 证伪与停止条件

出现以下情况时，不应继续扩大 controller 或神经网络：

1. sequence-level alignment 无法稳定优于 random；
2. alignment 与 anti-alignment 没有可重复差异；
3. alignment 只能改善当前 reference loss，不能改善独立 search-validation；
4. 一步预测与 5/10/20-step learning gain 方向不一致；
5. 结果主要由 gradient norm、clipping 或训练强度解释；
6. 不同 seeds 下 policy ranking 不稳定；
7. mixture 的收益不能超过 pure FKL/RKL 与 matched random；
8. per-token 方法不能超过更简单的 sequence/group-level 方法；
9. cheap features 无法预测 gradient statistics；
10. learned approximation 在 held-out checkpoint/stage 上退化。

若一阶 alignment 失败，可依次考虑：

1. 检查 reference objective；
2. 检查 FKL/RKL 数值定义与尺度；
3. 加入 norm/trust-region；
4. 检查 sequence/token interaction；
5. 考虑二阶或短程 meta-gradient；
6. 若仍失败，接受当前 action space 或局部 proxy 不适合该问题。

不能直接通过扩大网络来掩盖失败。

---

## 18. 主要风险与限制

1. **Reference mismatch**：reference questions 可能不能代表真实目标；
2. **Short-horizon bias**：一步有利不代表长期有利；
3. **Non-stationarity**：student 更新后，最佳 action 会变化；
4. **Gradient interaction**：token 之间并非严格可加；
5. **Teacher error**：teacher 不一定是正确目标；
6. **Scale confound**：FKL/RKL 可能因数值尺度不同产生假优势；
7. **Compute cost**：per-token/action directional derivative 可能昂贵；
8. **Feature insufficiency**：分布 feature 未必包含参数 Jacobian 信息；
9. **Allocator approximation**：大规模离散/二次约束优化可能只能近似；
10. **Validation overfitting**：reference、search-validation 和 locked test 必须严格隔离。

---

## 19. 推荐的当前第一版

当前最推荐的最小版本是：

1. 固定 question distribution 和 teacher；
2. 暂停训练黑盒 learning-utility verifier；
3. 明确定义 FKL 与 RKL/OPD token gradients；
4. 使用独立 reference/calibration set；
5. 先做 sequence-level gradient alignment；
6. 使用离散 action pairs；
7. 同时比较纯一阶 score 与 alignment + norm penalty；
8. 保持 global mean token weight \(=1\)；
9. 设置 random、anti-alignment、norm-only、fixed OPD/FKL/mixture controls；
10. 只有通过 sequence/group-level falsification gates 后，才进入 per-token；
11. 只有解析式方法有效但计算昂贵时，才训练小模型近似 gradient statistics；
12. 所有最终结论必须来自独立 search-validation 与冻结后的 locked test。

---

## 20. 一句话总结

本计划不是让黑盒网络从少量 policy-level reward 中猜出每个 token 的长期价值，而是：

> 先根据 reference objective 的梯度，计算每个 token 上不同 FKL/RKL action pair 所产生的更新方向、大小与稳定性；再在固定全局学习预算下联合选择所有 token 的 pairs，并通过独立短训练实验验证这一局部数学 proxy 是否真的带来未来能力提升。神经网络只在解析式信号已被验证且在线计算过贵时，作为该数学量的可选近似器。
