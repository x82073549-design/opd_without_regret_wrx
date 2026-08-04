# OPD Loss Search 前期实验计划

日期：2026-08-04
验收时间：2026-08-05 下午

## 1. 目标

本阶段为 Codex 自动设计 OPD loss 准备两个必要条件：

1. 确定候选 loss 至少训练多少步后才可以进行初步比较；
2. 确定 Codex 应当自由生成 loss，还是从预先定义的组件中组合 loss。

两项任务并行推进：

- Liujiang 负责确定候选 loss 的训练步数；
- Ruxin 负责确定 Codex 修改 loss 的方式；
- 两人使用相同的 OPD baseline、验证集和结果格式。

本阶段不研究训练过程中何时切换 loss。动态切换需要后续在不同训练 checkpoint 上比较各个 loss 的效果。

## 2. 临时评价指标

本阶段先使用一个简单、通用的评价指标。

在固定训练步数 \(H\) 下，计算候选 loss 与 OPD baseline 在验证集上的任务表现差值：

\[
\operatorname{Score}_H(L)
=
J_{\mathrm{val}}(\theta_H^L)
-
J_{\mathrm{val}}(\theta_H^{\mathrm{OPD}}).
\]

其中：

- \(L\) 是候选 loss；
- \(\theta_H^L\) 是使用候选 loss 训练 \(H\) 步后的模型；
- \(\theta_H^{\mathrm{OPD}}\) 是从相同初始状态使用 OPD 训练 \(H\) 步后的模型；
- \(J_{\mathrm{val}}\) 是验证集上的平均任务 reward。

数学任务使用相同题目和相同生成种子下的 exact reward mean，并保留逐题配对结果。

### 2.1 固定条件

候选 loss 和 OPD baseline 必须使用相同的：

- student、teacher 和 tokenizer；
- 初始 model checkpoint；
- optimizer 和 scheduler state；
- learning rate；
- 训练数据和数据顺序；
- batch size 和有效训练 token 数；
- rollout 配置和随机种子；
- checkpoint 保存位置；
- validation questions 和 evaluation seeds；
- 代码版本。

每次实验必须使用独立输出目录。起点为基础模型时使用 `RESUME_MODE=disable`；起点为已有训练 checkpoint 时使用 `RESUME_MODE=resume_path` 和固定的 `RESUME_FROM_PATH`，禁止使用 `auto` 自动寻找 checkpoint。

### 2.2 数据划分

- Training set：用于模型参数更新；
- Validation set：用于当前实验和 Codex 的候选选择；
- Test set：只在最终方法确定后使用，不进入当前实验，也不提供给 Codex。

Training、Validation 和 Test 之间需要检查重复数据。

正式实验前需要提交：

- `validation_manifest.json`；
- `test_manifest.json`；
- `train_validation_dedup_report.json`。

manifest 至少记录数据文件路径、SHA256、样本数、题目 ID、用途和 evaluation seed list。Test manifest 必须注明在搜索阶段不可访问。

Validation manifest 的格式参考 `docs/verifier/validation_manifest.example.json`。正式运行时复制为冻结文件，并通过 `VALIDATION_MANIFEST` 传给评测脚本。

使用以下命令生成训练集与验证集的重复数据报告：

```bash
python3 scripts/val/check_data_overlap.py \
  --train datasets/dapo-math-17k.parquet \
  --validation-manifest docs/verifier/validation_manifest.json \
  --output docs/verifier/train_validation_dedup_report.json
```

### 2.3 无效实验

出现以下任一情况时，实验结果标记为无效：

- loss 或 gradient 出现 NaN/Inf；
- 训练失败或 checkpoint 不完整；
- 实际训练步数或有效 token 数不一致；
- 初始 model 或 optimizer state 不一致；
- 修改了 optimizer、learning rate、训练数据、rollout 配置或评价方法；
- loss 读取了 validation 或 test 信息；
- token weight 违反预先规定的范围或归一化规则。

该评价指标是当前的临时标准。后续需要用更长训练验证它是否能够预测最终结果。

### 2.4 运行和评测前置检查

以下检查通过后，训练步数实验才能用于方法选择。

#### 唯一运行标识

每次评测使用唯一的 `EVAL_RUN_ID`：

```text
{method}_trainseed{training_seed}_root{root_step}_delta{delta_steps}
```

`MODEL_NAME`、`MERGED_DIR` 和结果目录必须包含该标识。已合并模型必须保存原始 checkpoint 路径；路径不一致时禁止复用。

#### Evaluation seed

评测脚本必须把真实 seed 传给生成引擎。输出中的每条记录必须包含：

- task；
- question ID；
- evaluation seed；
- response；
- correctness。

修改后先执行一次复现检查：同一模型、同一题目、同一 seed list 和相同运行环境独立评测两次，比较逐题输出和分数。

#### 输出完整性

每个任务的预期输出数为：

\[
N_{\mathrm{expected}}
=
N_{\mathrm{questions}}
\times
N_{\mathrm{evaluation\ seeds}}.
\]

实际输出数或唯一 `(question_id, evaluation_seed)` 数不等于预期值时，评测失败。不能使用部分输出计算总体分数。

#### Checkpoint 步数

计划中的 \(H\) 表示相对起始 checkpoint 新增的训练步数：

\[
H
=
\text{final global step}
-
\text{root global step}.
\]

`trainer.total_training_steps` 是绝对终止步数。例如，从 `global_step_40` 开始再训练 20 步，应设置终止步数为 60。

## 3. 任务一：确定候选 Loss 的训练步数

负责人：Liujiang

### 3.1 目标

确定候选 loss 至少需要训练多少步，才能与 OPD baseline 进行初步比较。

候选训练步数为：

\[
H\in\{20,40,60,80,100\}.
\]

本任务确定的是候选 loss 的评价步数，不是训练过程中更换 loss 的时间。

### 3.2 实验方法

至少比较以下三个已有方法：

1. Fixed OPD；
2. Sampled-token gate OPD；
3. Overlap-based OPD。

三种方法固定为以下配置。未列出的可选 gate、route 和 mask 全部关闭。

可执行配置文件位于：

- `configs/presearch/fixed_opd.env`；
- `configs/presearch/sampled_token_gate_opd.env`；
- `configs/presearch/overlap_prune_opd.env`。

#### Fixed OPD

```bash
METHOD=fixed_opd
ADV_ESTIMATOR=token_reward_direct
LOG_PROB_TOP_K=16
TOP_K_STRATEGY=only_stu
REWARD_WEIGHT_MODE=student_p
USE_KL=False
SAMPLED_TOKEN_GATE_OPD_ENABLE=False
TOPK_TOKEN_GATE_OPD_ENABLE=False
OVERLAP_ROUTE_OPD_ENABLE=False
OUTCOME_OPD_MASK_ENABLE=False
```

#### Sampled-token gate OPD

在 Fixed OPD 基础上设置：

```bash
METHOD=sampled_token_gate_opd
SAMPLED_TOKEN_GATE_OPD_ENABLE=True
SAMPLED_TOKEN_GATE_OPD_BETA=1.0
SAMPLED_TOKEN_GATE_OPD_CENTER=0.0
SAMPLED_TOKEN_GATE_OPD_MIN_GATE=0.0
SAMPLED_TOKEN_GATE_OPD_OPD_COEF=1.0
```

#### Overlap-based OPD

在 Fixed OPD 基础上设置：

```bash
METHOD=overlap_prune_opd
OVERLAP_ROUTE_OPD_ENABLE=True
OVERLAP_ROUTE_MODE=prune_opd
OVERLAP_ROUTE_TAU=0.7
OVERLAP_ROUTE_TOP_K=16
OVERLAP_ROUTE_TRIGGER=first_low
OVERLAP_ROUTE_WDROP=0.01
OVERLAP_ROUTE_WBASE=0.5
```

本阶段的 `Overlap-based OPD` 只指 `prune_opd`，不包含 `prune_opd_event_fkl` 或其他模式。

如果计算资源允许，增加一个对照方法：

- `0.5 × OPD`；或
- `2.0 × OPD`；或
- shuffled token weight。

所有方法从相同完整 checkpoint 开始，并在相对起点新增以下训练步数后保存 checkpoint：

```text
20, 40, 60, 80, 100
```

使用 2-GPU 脚本时必须显式设置：

```bash
SAVE_FREQ=20
TRAIN_TOTAL_STEPS=100
```

如果从非零 root checkpoint 恢复，`TRAIN_TOTAL_STEPS` 应设置为 `root_step + 100`。

每个 checkpoint 使用第 2 节定义的临时评价指标计算结果。

所有方法使用相同的 training seed 集合。原则上每种方法运行两个随机种子。如果计算资源不足：

- OPD baseline 至少运行两个随机种子；
- 其他方法先运行一个随机种子；
- 对初步最优方法补充第二个随机种子；
- 结果必须注明为初步结论。

两个 training seeds 只能支持初步筛选。形成稳定结论时至少补充第三个 training seed。

### 3.3 需要记录的结果

- validation reward；
- 相对 OPD baseline 的 reward 差值；
- 不同训练步数下的方法排名；
- training loss；
- pre-clip gradient norm；
- gradient clipping rate；
- response length；
- truncation rate；
- NaN、Inf 和训练失败情况；
- wall-clock time 和 peak GPU memory。

### 3.4 训练步数选择标准

对方法 \(L\)、training seed \(s\) 和训练步数 \(H\)，计算 matched OPD 差值：

\[
\Delta_{H,s}(L)
=
J_{\mathrm{val}}(\theta_{H,s}^{L})
-
J_{\mathrm{val}}(\theta_{H,s}^{\mathrm{OPD}}).
\]

评价时需要：

- 按 `(question_id, evaluation_seed)` 计算 paired delta；
- 分别报告 training-seed variation 和 evaluation-seed variation；
- 将 evaluation seeds 预先分为两个不重叠的列表，分别计算结果；
- 使用 paired bootstrap，并以 question 为重采样单位；
- 同一 question 下所有方法和 evaluation seeds 必须一起重采样；
- 报告 95% confidence interval。

逐方法与 matched OPD 的配对报告可以使用：

```bash
python3 scripts/analysis/paired_eval_report.py \
  --candidate <candidate>/detailed_results.jsonl \
  --baseline <baseline>/detailed_results.jsonl \
  --output <candidate>/paired_report.json
```

选择满足以下条件的最小训练步数：

1. 该步数下的最优方法与 step 100 的最优方法一致；
2. 两个 training seeds 下，最优方法相对 matched OPD 的差值同号；
3. 两个 evaluation seed lists 的最优方法和关键差值方向一致；
4. paired bootstrap confidence interval 已报告；
5. 训练过程正常且输出完整。

如果 step 100 仍不能稳定区分方法，应明确报告以下一种或多种情况：

- 100 steps 不足；
- 当前方法差异不足；
- validation 噪声过大；
- 需要增加随机种子或评价样本。

当前 step 100 只是短程参考终点。该实验只能判断较早 checkpoint 是否能够预测 step-100 排名，不能证明其能够预测完整训练后的最终排名。后续需要把 OPD、短程最优方法、短程最差方法和一个 scale control 继续训练到至少 300 steps 进行确认。

### 3.5 Codex 分析要求

向 Codex 提供全部 checkpoint 结果，要求 Codex：

1. 推荐最小训练步数；
2. 说明选择依据；
3. 判断较早 checkpoint 是否能预测 step 100 的结果；
4. 指出是否需要继续训练到 200 或 300 steps；
5. 不根据单个最佳 checkpoint 作决定。

### 3.6 提交材料

1. 所有运行配置和随机种子；
2. step 20/40/60/80/100 的结果表；
3. validation reward 曲线；
4. 相对 OPD baseline 的结果曲线；
5. 逐题、逐 evaluation seed 的评分文件；
6. training-seed variation 和 evaluation-seed variation；
7. paired bootstrap confidence interval；
8. 三份数据 manifest 和去重报告；
9. OPD baseline 的随机种子波动；
10. Codex 的分析；
11. 推荐的最小训练步数；
12. 单次实验的时间和显存开销。

### 3.7 验收标准

必须得到以下两种结果之一：

- 一个有实验支持的最小训练步数；
- 当前无法确定，并明确说明需要补充的实验。

## 4. 任务二：确定 Codex 修改 Loss 的方式

负责人：Ruxin

### 4.1 目标

比较以下两种 Codex loss 设计方式：

1. Codex 自由生成 loss 公式和 Python 实现；
2. Codex 从预先定义的组件中组合 loss。

根据候选的合法性、可执行性和可比较性，推荐第一版采用的方式。

### 4.2 方式一：自由生成

Codex 可以提出：

- divergence；
- token 或 trajectory weight；
- gate；
- normalization；
- regularization；
- 数学公式和 Python 实现。

Codex 不允许修改：

- student 和 teacher；
- optimizer 和 learning rate；
- 训练数据；
- rollout 配置；
- validation；
- 训练框架；
- 评价代码。

### 4.3 方式二：组件组合

一个候选 loss 由以下部分组成。

#### Divergence

- RKL/OPD；
- FKL；
- FKL/RKL mixture；
- JSD。

#### Input

- student sampled-token log-probability；
- teacher sampled-token log-probability；
- sampled-token log-ratio；
- student entropy；
- teacher entropy；
- entropy gap；
- top-k overlap；
- token position；
- trajectory correctness；
- normalized training step。

#### Weight function

- constant；
- sigmoid；
- threshold；
- clipped linear function；
- power function；
- token mask；
- trajectory mask。

#### Normalization

- mean token weight equal to 1；
- weight clipping；
- loss scale matching；
- empty-batch fallback。

#### Optional regularization

- weight variance penalty；
- effective sample size constraint；
- gradient norm penalty。

第一版最多允许：

- 一个 divergence；
- 一个 weight function；
- 一个 normalization；
- 一个 optional regularization。

### 4.4 比较方法

向两种方式提供完全相同的：

- OPD baseline；
- 允许使用的输入；
- 当前训练统计；
- 禁止修改的内容；
- 现有 loss 实现。

分别要求 Codex 生成 5 个候选 loss。

对每个候选检查：

1. 数学定义是否完整；
2. 是否只使用允许的输入；
3. 是否修改禁止修改的内容；
4. 是否可以确定地转换为配置和代码；
5. forward 是否成功；
6. backward 是否成功；
7. loss 和 gradient 是否有限；
8. token weight 是否满足范围和归一化；
9. 是否与已有候选重复；
10. 需要多少人工修改才能运行。

### 4.5 必须支持的基础方法

推荐的 action 定义至少能够表示：

- Fixed OPD；
- `0.5 × OPD`；
- `1.0 × OPD`；
- `2.0 × OPD`；
- Fixed FKL；
- Uniform FKL/RKL mixture；
- Sampled-token gate OPD；
- Overlap-based OPD；
- shuffled token weight；
- sign-flipped token weight。

### 4.6 执行顺序

本任务不等待任务一确定最终训练步数，按以下顺序执行：

1. 对全部候选做静态检查；
2. 对合法候选做小 batch forward/backward；
3. 对通过检查的候选做 1–5 step smoke test；
4. 如果资源允许，从两种方式中各选择一个候选，按临时的 100-step 标准运行。

100-step 训练结果不是本任务的必要验收条件。本任务首先判断两种方式能否稳定产生合法、可执行和可比较的 loss。

### 4.7 比较指标

比较两种方式的：

- 合法候选比例；
- forward/backward 通过率；
- smoke test 通过率；
- 与 Fixed OPD 的一致性；
- 重复候选比例；
- 人工修改量；
- 是否能够自动检查；
- 是否能够控制 loss scale；
- 是否能清楚说明候选之间的差异。

### 4.8 Codex 分析要求

根据比较结果，要求 Codex：

1. 推荐自由生成或组件组合；
2. 说明推荐依据；
3. 指出当前组件是否完整；
4. 推荐保留、删除或增加的组件；
5. 给出第一批可以进入训练实验的候选 loss。

### 4.9 提交材料

1. 两种方式使用的 prompt；
2. 每种方式生成的 5 个候选；
3. 候选的数学公式和代码；
4. 静态检查结果；
5. forward/backward 和 smoke test 结果；
6. Fixed OPD 一致性测试；
7. 两种方式的比较表；
8. 推荐的 action 定义；
9. 第一批可执行候选列表。

### 4.10 验收标准

必须明确回答：

- 第一版使用自由生成还是组件组合；
- 推荐的 action 结构；
- 允许使用的输入；
- 允许使用的组件；
- 参数范围和归一化规则；
- 哪些候选可以进入下一轮训练实验。

## 5. 协作要求

### 5.1 共享内容

两项任务共享：

- Fixed OPD baseline；
- 初始 checkpoint；
- Validation set；
- evaluation seeds；
- 结果字段和文件格式；
- 代码版本和配置 hash。

### 5.2 并行推进

- Liujiang 直接使用现有 loss 变体确定训练步数；
- Ruxin 不等待训练步数结论，先完成两种 loss 设计方式的比较和执行检查；
- Ruxin 产生的新候选，在任务一完成后使用推荐的训练步数进行正式筛选；
- 相同配置的 OPD baseline 结果只运行一次并共享，避免重复使用计算资源。

## 6. 明天下午验收内容

### Liujiang

- 推荐的最小训练步数，或证据不足的明确结论；
- 完整结果表和曲线；
- OPD baseline 的随机种子波动；
- Codex 的分析；
- 计算成本。

### Ruxin

- 自由生成与组件组合的比较结果；
- 推荐的 loss action 定义；
- 合法性和执行测试结果；
- 第一批可执行候选 loss。

### 共同结论

验收会议需要判断：

1. 当前是否已经得到可用的候选训练步数；
2. 当前是否已经得到可用的 loss action 定义；
3. 是否具备启动第一轮 Codex OPD loss search 的条件；
4. 临时评价指标需要补充哪些长期验证。
