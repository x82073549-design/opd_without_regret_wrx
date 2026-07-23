# On-Policy Distillation 有效性相关特征清单

本文件整理 On-Policy Distillation (OPD) 训练过程中可探测、且可能影响或预测 OPD 收益的特征，
用于「样本选择 / teacher 选择 / token 加权」三类策略的设计与实验规划。

## 术语与约定

**特征粒度**：token 级、样本/轨迹级、问题级、teacher 级。

**成本标记**：
- *无额外开销*：标准 OPD 前向中已计算的量（如 log-prob、熵、KL）。
- *额外 teacher 前向*：需一次 teacher 推理但无需重复采样。
- *额外采样*：需对同一问题多次采样（如 pass-rate、teacher 正确率）。
- *需验证器*：需 ground-truth 或可验证奖励。
- *高（离线）*：开销大，仅适合离线诊断而非进入训练循环。

**证据等级**（用于「效应方向」列）：
- **[实证]**：在所引文献的设定下有直接实验观察。
- **[推断]**：从某项实证观察推导出的设计含义，未被该来源直接验证。
- **[假设]**：尚无直接 OPD 证据，仅来自类比或理论动机。

**「跨粒度」**：该特征能否从 token 聚合至样本、再至 teacher 选择三层通用：
- ✓✓✓ = 三层均可天然定义；✓ = 适用其中一两层；△ = 基本仅限单一粒度；— = 不按粒度区分。

**「自带门控」**：该特征是否内含「学生是否已掌握」的可靠性判断：
- ✓ = 自带或本身即门控；✗ = 不带，单独使用会在「学生已掌握」区误分配；— = 不适用。

**评估目标的定义**：后续相关性/消融分析中，因变量「OPD 收益」需事先明确操作化，候选包括：
(a) 单步对 held-out 集 reverse-KL（学生→teacher）的下降量；
(b) 通过留一/影响函数估计的、归因于某样本或 token 子集的下游准确率边际增益。
本清单不预设其中之一，但要求实验前固定一种。

---

## 1. 特征优先级与实验规划

排序依据：证据强度 × 计算成本 × 跨粒度能力 × 是否自带门控。
建议第一阶段对 Tier 1 全部埋点，并以 Tier 2 的「学生正确性」作为门控，先做相关性/消融分析，再确定组合。

| 优先级 | 特征                                | 排序理由                                                                        | 成本            | 跨粒度           | 自带门控     | 第一步实验                               |
| --- | --------------------------------- | --------------------------------------------------------------------------- | ------------- | ------------- | -------- | ----------------------------------- |
| T1  | top-k overlap ratio               | RethinkingOPD 中同时用作 teacher 兼容性判据与 token 级训练成败的观测指标；verl 已内置为日志指标（不进入 loss） | 无额外开销         | ✓✓✓           | 弱（需配正确性） | 读取 verl 现有指标，作为主信号候选                |
| T1  | student 熵                         | TIP 报告其为 token 重要性的强一阶代理，但结构上不完备（不能区分「自信且对」与「自信且错」）                         | 无额外开销         | △（偏 token）    | ✗        | token 加权基线，对标 TIP                   |
| T1  | teacher 熵 / 置信度                   | 可作可靠性门控；EOPD 据此在高 teacher 熵位置切换为 forward KL                                 | 额外 teacher 前向 | △             | ✓（作门控）   | 与 student 熵相乘，复现 FiRe 的 c_t^T·c_t^S |
| T1  | logprob-ratio reward / 逐 token KL | OPD 优势项本体，作为所有加权方案的对照基线                                                     | 无额外开销         | △             | ✗        | 作为基线，其余方案相对其比较                      |
| T2  | 可约散度（散度 × 可靠性门控）                  | 本工作拟验证的复合信号假设；与 RHO-LOSS 的「可约损失」思想类比，尚无系统性 OPD 证据                           | 无额外开销         | ✓✓✓           | ✓        | 由 T1 各量构造，三粒度各实例化一版并消融              |
| T2  | 「自信且错」指示（低 student 熵 + 高散度）       | TIP 第二象限；熵阈值规则无法识别的高价值 token                                                | 无额外开销         | △             | ✓        | 并入 token 加权，测增量贡献                   |
| T2  | teacher 轨迹归一化 log-prob            | 样本/轨迹级最有效的免费过滤量                                                             | 无额外开销         | ✓（样本+teacher） | ✓        | 复现 FiRe 的底部 20% 硬过滤                 |
| T2  | entropy gap（teacher − student）    | 训练健康度指标，与 overlap 互补                                                        | 无额外开销         | ✓             | ✗        | 作监控量及辅助加权                           |
| T2  | 学生 rollout 正确性                    | 全清单的核心门控，决定各特征的有效方向                                                         | 需验证器          | —             | ✓（门控本体）  | 必埋；用于按「已掌握/未掌握」分层各特征                |
| T3  | 组 pass-rate / 难度                  | 问题级可学带定位（与 PACED、CoDaPO 同源）                                                 | 额外采样          | △（问题级）        | ✓        | 样本选择基线                              |
| T3  | 新能力指示（teacher 正确 ∧ student 错误）    | teacher 选择的关键判据（RethinkingOPD 条件 ii）                                        | 需双方正确性        | ✓（teacher 级）  | ✓        | teacher 路由实验阶段引入                    |
| T3  | 长度 / 重复膨胀率                        | 训练稳定性监控（Demystifying OPD 的失败模式）                                             | 无额外开销         | △             | ✗        | 作 guardrail，不作主加权信号                 |
| T4  | 梯度对齐分数                            | 最接近理论理想信号（理想梯度的余弦相似度），但计算昂贵，宜作离线验证基准                                        | 高（离线）         | ✓             | ✓        | 仅在小集上离线验证 T1/T2 的有效性                |
| T4  | 隐状态 / 表征逐层对齐                      | 信息更丰富但需 hidden states                                                       | 额外 teacher 前向 | ✓             | —        | 进阶方向，非起步项                           |
| T4  | 跨 tokenizer / 词表重叠                | 仅在 teacher 与 student tokenizer 不一致时适用                                       | 中             | △             | —        | 视场景启用                               |
| T4  | 任一特征的时序导数                         | 可更早预测收益，但需先有静态日志                                                            | 无额外开销         | ✓             | —        | 静态版本跑通后再加                           |

### 起步实验协议

1. 对 Tier 1 四项全部埋点（均无额外开销，verl 大部分已内置），并采集 Tier 2 的「学生正确性」作为门控变量。
2. 固定一种「OPD 收益」操作化定义（见上「评估目标的定义」），统计各特征（及其复合）与该因变量的相关性。
3. 以 overlap 为主信号、可约散度为待验证假设，分别向 token 加权与 teacher 路由两个方向扩展，每一步配独立消融。
4. 梯度对齐分数不进入训练循环，仅在小规模集合上作为「所选特征方向是否正确」的离线判据。

门控原则贯穿全程：任一特征在「学生已正确」的区域应降权。该原则由 Unmasking OPD 的实证观察推断而来（见下节），其跨设定的普适性仍属待验证假设。

---

## 2. 总体判据

Unmasking OPD 报告：在其自蒸馏及部分外部 teacher 设定下，蒸馏信号与「理想梯度」（定义为最大化学生成功概率的参数更新）的余弦对齐度，在学生答错的 rollout 上显著高于答对的 rollout；并观察到最优 teacher 随学生容量与目标任务而变化，不存在单一普适配置，且按 student–teacher 散度门控 teacher 信号与对齐度正相关。

由此**推断**（非该来源直接验证）：多数特征的有效方向应附带「学生是否已掌握」的门控；统一信号宜采用复合形式 `分歧/可学量 × 可靠性门控`。此推断的跨设定普适性是本工作的待验证假设之一。

---

## 3. 特征全表（按测量来源分组）

字段：测什么 / 天然粒度 / 效应方向（含证据等级）/ 计算成本 / 来源。
「来源」列出文献者，表示在该文献设定下的实证观察；标注「(假设)」者尚无直接 OPD 证据。

### A. 学生侧内禀（student 在自身 rollout 上计算）

| 特征 | 测什么 | 粒度 | 效应方向 | 成本 | 来源 |
|---|---|---|---|---|---|
| student token 熵 | 不确定性 / 分叉点 | token | [实证] 高熵对应决策点、信息量大；过高时噪声风险上升 | 无额外开销 | TIP 主轴、FiRe c_t^S |
| student 置信度 / sampled-token 概率 | 承诺程度 | token | [推断] 低置信=需引导；高置信且错误=过度自信，纠错价值高 | 无额外开销 | TIP；CoDaPO 类比 |
| student 校准（置信 vs 正确率） | 自我认知质量 | 样本/全局 | [假设] 校准越差越应由外部信号修正 | 需验证器 | (假设)；CoDaPO confidence 同源 |
| 多样性坍缩率 | 高熵 token 占比的变化 | 全局/批 | [实证] OPD 下高熵 token 保留比例下降（约 6.8%，teacher 约 18.5%），反映多样性退化 | 无额外开销 | EOPD |
| 重复率 / n-gram 重复 | 退化生成 | 样本/token | [实证] 重复 token 获得偏大优势，经 on-policy 放大致长度膨胀 | 无额外开销 | Demystifying OPD |
| rollout 长度 / 截断标记 | 是否失控 | 样本 | [实证] 长度突增伴随截断坍缩，为训练病理征兆 | 无额外开销 | Demystifying OPD |
| 轨迹位置 / 深度 | 前缀漂移程度 | token/段 | [实证] 越靠后局部兼容性越脆弱；权重宜随结构变化 | 无额外开销 | Prune-OPD、FiRe |
| token 类型（连接词 vs 数值/算符） | 推理 vs 执行 | token | [实证] 推理连接词（如 Therefore、Since）获较高权重，数值/算符较低 | 无额外开销 | FiRe 案例分析 |

### B. Teacher 侧内禀（teacher 在 student 轨迹上计算）

| 特征 | 测什么 | 粒度 | 效应方向 | 成本 | 来源 |
|---|---|---|---|---|---|
| teacher token 熵 / 置信度 | 监督可靠性 | token | [实证] 高 teacher 熵处信号不稳定，宜降权或改用 forward KL | 额外 teacher 前向 | EOPD、FiRe c_t^T |
| teacher 对学生 token 的 NLL | 局部认同度 | token | [实证] teacher 赋予低概率=局部分歧大 | 无额外开销 | OPD 优势项组成 |
| teacher 轨迹归一化 log-prob | 对该路径的胜任度 | 样本/轨迹 | [实证] 偏低表明 teacher 处于陌生路径，监督不可靠，宜过滤 | 无额外开销 | FiRe（底部 20% 过滤） |
| teacher 在该问题的正确率 | 能否提供有效监督 | 样本 | [实证/推断] teacher 错误则信号有害 | 额外采样 | BRTS、RethinkingOPD 条件 ii |
| teacher top-k 质量覆盖 | top-k 近似是否充分 | token | [实证] 覆盖率低表明截断了较多 teacher 概率质量 | 无额外开销 | verl teacher_mass |

### C. 师生关系型（统一信号的主要来源）

| 特征 | 测什么 | 粒度 | 效应方向 | 成本 | 来源 |
|---|---|---|---|---|---|
| 逐 token KL / reverse-KL | 分歧=可学量 | token | [实证] 即 OPD loss/优势本体；不加门控地追求最大值会引入噪声 | 无额外开销 | Thinking Machines（reward = −reverse KL） |
| top-k overlap ratio | 师生高概率区重合度 | token→样本→teacher | [实证] 训练中上升与成功相关、停滞与失败相关 | 无额外开销（verl 日志，不进 loss） | RethinkingOPD、Prune-OPD |
| overlap-token advantage | 重叠区有效梯度 | token | [实证] 趋零对应收敛 | 无额外开销 | RethinkingOPD / verl |
| entropy gap（teacher − student） | 局部置信度差 | token/样本 | [实证] 收窄与学到 teacher 局部置信相关 | 无额外开销 | RethinkingOPD |
| logprob-ratio reward a_t | OPD 优势 | token | [实证] 低熵 token 上该奖励高度集中于 0，梯度趋于消失 | 无额外开销 | FiRe Eq.2、Relaxed OPD |
| 「自信且错」指示（低 student 熵 + 高散度） | 密集纠错信号 | token | [实证] 高价值且为熵阈值规则所忽略 | 无额外开销 | TIP 第二象限 |
| student-mass on teacher 偏好 token | 学生赋予对方偏好 token 的概率 | token | [实证] 偏低表明尚未习得 | 无额外开销 | verl student_mass |
| 梯度对齐分数（蒸馏梯度与理想梯度的余弦相似度） | 信号方向正确性 | token/样本/teacher | [实证] 越高越好；在答错 rollout 上更高 | 高（离线） | Unmasking OPD |
| 思维模式兼容性（格式/结构/风格匹配） | 选择条件 i | teacher/样本 | [实证] 不兼容时 OPD 可能失败甚至倒退 | 中 | RethinkingOPD |
| 隐状态/表征逐层对齐 | LM head 之外的结构信息 | token/层 | [实证] 对齐高与可迁移相关，并消除采样方差 | 额外 teacher 前向 | OPRD |
| 漂移信号（累积位置兼容失败） | 前缀已偏离程度 | 段/轨迹 | [实证] 超阈值时宜截断或降权后续 | 无额外开销 | Prune-OPD |

### D. Outcome / 问题级

| 特征 | 测什么 | 粒度 | 效应方向 | 成本 | 来源 |
|---|---|---|---|---|---|
| 学生 rollout 正确性 | 是否已掌握 | 样本 | [推断] 答对处 teacher 信号偏噪，宜降权 | 需验证器 | Unmasking OPD、BRTS |
| 组 pass-rate / 难度 | 可学带定位 | 问题 | [实证/假设] 中等难度信息量最大（U 型） | 额外采样 | PACED、GFPO |
| pass@k / 发现率 | 是否可达 | 问题 | [假设] 过难=不可达，宜降权 | 额外采样 | CoDaPO 发现概率同源 |
| 新能力指示（teacher 正确 ∧ student 错误） | 是否含可迁移新知识 | 问题/teacher | [实证] teacher 选择关键判据 | 需双方正确性 | RethinkingOPD 条件 ii |
| 可约散度（散度 × 可靠性门控） | 高分歧且可学 | token→样本→teacher | [假设] 本工作拟验证的统一信号 | 无额外开销 | (假设)；RHO-LOSS 类比 |

### E. 跨 tokenizer / 词表（teacher 与 student tokenizer 不一致时）

| 特征 | 测什么 | 粒度 | 来源 |
|---|---|---|---|
| 词表重叠 / token 对齐质量 | 能否逐 token 对比 | token/样本 | GOLD、DWA-KD、byte-level interface |
| 对齐 span 占比 | 可监督位置比例 | 段 | SimCT、CTPD |

### F. 动态 / 时序（任一特征的变化量）

将静态特征替换为其时序导数，可更早预测某样本或 teacher 的剩余价值。RethinkingOPD 即以指标「上升 vs 停滞」区分成功与失败的训练。

| 特征 | 说明 | 来源 |
|---|---|---|
| overlap ratio 上升斜率 | 上升与健康相关，停滞与失败相关 | RethinkingOPD |
| entropy gap 收窄速率 | 收窄与习得 teacher 置信相关 | RethinkingOPD |
| 长度 / 重复膨胀率 | 突增为病理前兆 | Demystifying OPD |
| 梯度范数 / 优势量衰减 | 衰减对应信号枯竭 | 通用 |

---

## 4. 筛选标尺

**标尺一：成本。** 无额外开销项（student/teacher 熵、置信度、KL、overlap、logprob-ratio、长度、重复、位置）应优先纳入；高成本项（梯度对齐、teacher 正确率、pass@k）宜作离线验证而非进入训练循环。统一信号应尽量建立在无额外开销量之上。

**标尺二：跨粒度能力。** 可从 token 聚合至样本再至 teacher 的特征仅少数：overlap、可约散度（假设）、梯度对齐、teacher 轨迹 log-prob。仅限单一粒度者（如 token 类型、位置）不适合作为统一信号候选。

**标尺三：门控。** 不含正确性/可达性门控的特征（如纯熵、纯散度）在「学生已掌握」区会误分配。统一信号宜为复合形式 `分歧/可学量 × 可靠性门控`，其中门控在 teacher 级取兼容性（overlap）、token 级取 teacher 置信、问题级取新能力指示。

---

## 5. 文献索引

| 简称 | 标题 | arXiv |
|---|---|---|
| RethinkingOPD | Rethinking On-Policy Distillation: Phenomenology, Mechanism, and Recipe | 2604.13016 |
| TIP | Token Importance in On-Policy Distillation | 2604.14084 |
| FiRe-OPD | Filter, Then Reweight: Rethinking Optimization Granularity in OPD | 2606.02684 |
| PACED | Distillation and On-Policy Self-Distillation at the Frontier of Student Competence | 2603.11178 |
| EOPD | Entropy-Aware On-Policy Distillation of Language Models | 2603.07079 |
| Relaxed OPD (REOPOLD) | Scaling Reasoning Efficiently via Relaxed On-Policy Distillation | 2603.11137 |
| Demystifying OPD (Stable-OPD) | Length Inflation and Stabilization Strategies | 2604.08527 |
| Prune-OPD | Efficient and Reliable OPD for Long-Horizon Reasoning | 2605.07804 |
| Unmasking OPD | Where It Helps, Where It Hurts, and Why | 2605.10889 |
| BRTS | OPD with Best-of-N Teacher Rollout Selection | 2605.09725 |
| SCOPE | Signal-Calibrated OPD with Dual-Path Adaptive Weighting | 2604.10688 |
| Uni-OPD | Unifying On-Policy Distillation with a Dual-Perspective Recipe | 2605.03677 |
| ExOPD | Learning beyond Teacher: Generalized OPD with Reward Extrapolation | 2602.12125 |
| GFPO | Sample More to Think Less: Group Filtered Policy Optimization | 2508.09726 |
| Thinking Machines | On-Policy Distillation (技术博客) | thinkingmachines.ai/blog/on-policy-distillation |
| verl OPD | verl 官方文档（overlap_ratio / overlap_token_advantage / teacher_mass / student_mass） | verl.readthedocs.io |
| OPRD | On-Policy Representation Distillation（隐状态对齐） | github.com/ShenzhiYang2000/OPRD |
| 跨 tokenizer | GOLD / DWA-KD / SimCT / CTPD / byte-level interface | 见 awesome-on-policy-distillation |

策展列表：
- github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation
- github.com/chrisliu298/awesome-on-policy-distillation
