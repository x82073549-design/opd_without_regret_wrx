# OPD Loss State 0 选择与复现实验计划

## 1. 目标

State 0 用于建立一组可执行、可比较的现有 OPD loss，作为后续 loss 演化研究的初始基础。

本阶段只完成以下工作：

1. 选择具有代表性的论文 loss；
2. 按论文定义完成实现；
3. 在本项目的统一设置下重新运行完整实验；
4. 保存代码、配置和实验结果，形成 State 0。

本阶段不设计后续 evolve 流程，不加入额外论文 memory，也不把项目内部产生的 loss 作为初始父节点。

## 2. 固定对照

### Fixed OPD

Fixed OPD 是所有实验的统一 baseline，不作为待演化的论文 loss。

所有 State 0 方法必须与 Fixed OPD 使用相同的：

- 起始 checkpoint；
- 训练数据和数据顺序；
- batch size；
- 完整训练步数；
- training seeds；
- evaluation questions 和 evaluation seeds；
- 生成参数和评价程序。

## 3. State 0 论文选择

### 3.1 EOPD

- 论文：[Entropy-Aware On-Policy Distillation of Language Models](https://arxiv.org/abs/2603.07079)
- 核心方法：根据 teacher entropy 调整 FKL 和 RKL 的组合。
- 选择原因：代表 divergence 目标的结构变化；所需信号明确，容易接入当前 OPD loss；与其他候选的机制重合较少。

### 3.2 AOPD

- 论文：[Asymmetric On-Policy Distillation: Bridging Exploitation and Imitation at the Token Level](https://arxiv.org/abs/2605.06387)
- 核心方法：在正 token advantage 区域保留 policy-gradient 目标，在非正 advantage 区域使用局部 divergence minimization。
- 选择原因：代表根据训练信号切换优化目标的结构变化，不只是调整 loss 权重。

### 3.3 TIP

- 论文：[TIP: Token Importance in On-Policy Distillation](https://arxiv.org/abs/2604.14084)
- 核心方法：根据 student entropy 和 teacher-student divergence 选择重要 token。
- 选择原因：代表 token-level selection；所需信号可以由现有 teacher/student 分布计算；能够研究不同 token 对 OPD 的贡献。

### 3.4 Prune-OPD

- 论文：[Prune-OPD: Efficient and Reliable On-Policy Distillation for Long-Horizon Reasoning](https://arxiv.org/abs/2605.07804)
- 核心方法：使用 teacher-student prefix compatibility 检测 prefix drift，并降低漂移后 token 的监督权重。
- 选择原因：代表 sequence-level reliability；适合长推理场景；可以与项目已有的 overlap 相关实现衔接。

### 3.5 ExOPD

- 论文：[Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation](https://arxiv.org/abs/2602.12125)
- 核心方法：将 teacher 相对 reference model 的 log-ratio 作为隐式 token reward，并通过外推系数调整 reward 强度。
- 选择原因：代表 reward transformation；能够探索标准 OPD 模仿目标之外的优化方向。

## 4. 选择原则

以上五个方法分别覆盖不同的主要设计方向：

| 方法 | 主要设计方向 |
|---|---|
| EOPD | divergence 组合 |
| AOPD | objective 切换 |
| TIP | token 选择 |
| Prune-OPD | prefix 可靠性 |
| ExOPD | reward 外推 |

选择标准为：

1. 方法对 loss 结构有明确改变，而不是只做普通超参数调整；
2. 五个方法之间具有足够的结构差异；
3. 所需输入信号原则上能在当前训练流程中获得；
4. 方法能够在统一接口下独立运行和评价；
5. 实验结果能够与 Fixed OPD 进行公平比较。

## 5. 实现要求

每个论文 loss 必须先完成论文忠实实现。第一版不得同时加入项目自定义修改，以免无法区分论文方法和额外修改的效果。

每个方法至少应提供：

```text
state_0/<method>/
├── loss.py
├── config.yaml
├── description.md
├── diagnostics.json
└── result.json
```

其中：

- `loss.py`：完整、可运行的 loss 实现；
- `config.yaml`：训练、模型、数据和方法参数；
- `description.md`：论文来源、公式、实现对应关系和所需输入；
- `diagnostics.json`：loss 数值、KL、entropy、有效 token 比例等运行诊断；
- `result.json`：完整实验的评价结果及相对 Fixed OPD 的差值。

## 6. 完整实验要求

每个方法都必须在本项目 setting 下重新运行一次完整实验，不能直接使用论文中的结果代替。

完整实验包括：

1. 从统一的完整 checkpoint 开始训练；
2. 使用与 Fixed OPD 相同的完整训练预算；
3. 使用 matched training seeds；
4. 在固定 validation set 上使用相同题目和真实 evaluation seeds；
5. 保存所有规定 checkpoint；
6. 保存逐题、逐 evaluation seed 的结果；
7. 报告最终性能、训练曲线和关键诊断指标；
8. 记录代码 commit、环境、启动命令和输出目录。

测试集不用于 loss 选择、实现调整或超参数调整。State 0 的建立只使用训练数据和 validation 数据。

## 7. 验收标准

一个论文 loss 只有同时满足以下条件，才可以进入 State 0：

- 论文公式与代码实现的对应关系清楚；
- 通过静态检查和基础数值检查；
- 能够在统一训练接口下稳定运行；
- 完成规定的完整训练实验；
- 完成与 Fixed OPD 配对的统一评测；
- 代码、配置、checkpoint、日志和结果齐全；
- 从记录的 commit 和命令可以重新运行。

如果某个方法由于缺少必要输入、实现错误或训练不稳定而无法完成实验，应记录失败阶段和具体原因，但不进入 State 0。

## 8. 最终交付

本阶段最终交付包括：

1. Fixed OPD 的统一对照结果；
2. EOPD、AOPD、TIP、Prune-OPD 和 ExOPD 的论文忠实实现；
3. 五个方法在本项目 setting 下的完整实验结果；
4. 一张统一对比表；
5. 可复现的代码、配置、命令、checkpoint 和评测输出。

完成上述交付后，State 0 才正式冻结。后续 evolve 的父节点选择、迭代方式和淘汰规则另行设计。
