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

Ruxin 不等待训练步数结论，先完成 action 定义、静态检查、forward/backward 和 smoke test。Liujiang 确定训练步数后，Ruxin 的首批合法候选再使用该步数进入正式筛选。

本阶段不研究训练过程中何时切换 loss、checkpoint controller 或最终 test 表现。候选参数不能根据 Validation 结果反复调整。

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
- 起始 model checkpoint；
- 起始方式，以及对应的 optimizer 和 scheduler state/初始化；
- learning rate；
- 训练数据和数据顺序；
- batch size 和有效训练 token 数；
- rollout 配置和随机种子；
- checkpoint 保存频率；
- validation questions 和 evaluation seeds；
- 代码版本。

每次实验必须使用独立输出目录。起始方式必须二选一：

1. 从完整 trainer checkpoint 开始：使用 `RESUME_MODE=resume_path` 和固定的 `RESUME_FROM_PATH`，恢复 model、optimizer、scheduler、global step 以及框架能够恢复的 dataloader/RNG state；
2. 从 model checkpoint 开始：使用 `RESUME_MODE=disable`，所有方法重新初始化相同的 optimizer 和 scheduler，并在 manifest 中记录 `fresh_optimizer=true`。

两种起始方式不能混用。禁止使用 `auto` 自动寻找 checkpoint，也不能把第二种方式描述为恢复了相同 optimizer state。

### 2.2 数据划分

- Training set：用于模型参数更新；
- Validation set：用于当前实验和 Codex 的候选选择；
- Test set：只在最终方法确定后使用，不进入当前实验，也不提供给 Codex。

Training、Validation 和 Test 之间需要检查重复数据。

正式实验前需要提交：

- `validation_manifest.json`；
- `test_manifest.json`；
- `train_validation_dedup_report.json`。

Validation manifest 至少记录数据文件路径、SHA256、样本数、题目 ID、顺序、用途、prompt template hash 和 evaluation seed list。Test manifest 在搜索阶段只能向 Codex 提供 dataset-level hash、样本数和访问策略，不能提供题面、答案或可访问路径。

不能因为仓库目录名为 `test_data` 就把数据继续视为最终 Test：任何用于本轮选择的数据都必须冻结为 Validation，之后不能再作为最终 Test 报告。当前 manifest 示例暂列 AIME24 和 Minerva，但它们也必须通过真实 parquet 去重检查才能用于正式实验；AIME25、AMC23 或其他数据不能未经检查直接恢复为默认集合。

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

#### Training run

一条 training run 由起始 checkpoint、method、training seed、代码 commit 和 resolved config 唯一确定。每条 run 连续训练到相对步数 100，并在 20/40/60/80/100 保存 checkpoint；不得为五个步数分别启动五条训练。

正式运行前记录 Git commit、dirty patch hash、Python/PyTorch/verl/vLLM/CUDA 版本、GPU 型号、完整 resolved config 和启动命令。正式 run 期间不得切换代码版本。

#### Training seed

training seed 与 evaluation seed 分开。training seed 至少控制 Python、NumPy、PyTorch CPU/CUDA、rollout engine 和数据顺序中框架允许控制的随机性。正式运行前必须验证：

1. 同方法、同 seed 的短运行能够复现；
2. 同方法、不同 seed 的 rollout 或训练轨迹确实不同；
3. resolved config 和 run manifest 中能看到实际生效的 seed；
4. 同一 seed 在所有方法中于相同执行阶段设置。

仅修改运行名称中的 `seed` 不算完成 seed 控制。

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

预注册两个互斥的 seed lists：

```text
E0 = [0, 1, 2, 3]
E1 = [4, 5, 6, 7]
E  = E0 + E1
```

主结果使用 `mean@8`，`mean@4(E0)` 和 `mean@4(E1)` 用于检查 evaluation noise。每个 request seed 由 stable question ID 和 evaluation seed 确定；映射规则和 seed list 写入 evaluation manifest。

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

本计划中的 Fixed OPD 专指以下 fixed top-k OPD，不指 sampled-token OPD。

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

如果计算资源允许，优先增加 shuffled token weight 作为负对照，并固定 shuffle seed。`0.5 × OPD` 或 `2.0 × OPD` 只作为 loss-scale 诊断，因为 Adam 和 gradient clipping 可能减弱单纯缩放的影响。

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

标准矩阵为 `3 methods × 2 training seeds = 6 training runs`，共 30 次 checkpoint evaluations。资源不足时按以下顺序完成最低矩阵：

1. 三种方法的 training seed 0；
2. Fixed OPD 的 training seed 1；
3. seed 0 下最优的非 baseline 方法补 training seed 1。

正式矩阵前必须通过：三种方法的静态配置检查、小 batch forward/backward、1-step smoke test、training/evaluation seed 审计，以及一条 checkpoint save → merge → generate → grade → paired report 的完整链路。

### 3.3 需要记录的结果

- validation reward；
- 相对 OPD baseline 的 reward 差值；
- 不同训练步数下的方法排名；
- training loss；
- pre-clip gradient norm；
- gradient clipping rate；
- response length；
- truncation rate；
- effective response/training tokens；
- expected/actual generation count；
- NaN、Inf 和训练失败情况；
- wall-clock time 和 peak GPU memory。

逐题文件至少保留 method、training seed、relative horizon、task、question ID、evaluation seed、request seed、exact reward、response length、truncation 状态、checkpoint hash 和 evaluation config hash。

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

明天下午的 pilot 选择满足以下条件的最小训练步数：

1. 该步数下的最优方法与 step 100 的最优方法一致；
2. 两个 training seeds 下，top method 相对 runner-up 的差值同号；top method 不是 Fixed OPD 时，同时检查其相对 matched Fixed OPD 的差值；
3. E0 和 E1 的最优方法及关键差值方向一致；
4. paired bootstrap confidence interval 已报告，并检查结论是否由单个题目驱动；
5. 训练过程正常且输出完整。

pilot 结论只能标为“推荐的初步筛选步数”。形成稳定结论还需要：完整方法排名在该步数与 step 100 一致、top-vs-runner-up 的 paired bootstrap 95% CI 不跨 0、leave-one-question-out 后 winner 不变，并至少补充第三个 training seed。若这些条件未满足，不应把 pilot 结论表述为稳定最小步数。

如果 top score 并列，或不同 tie-breaking 会改变 winner，则该步数记为“当前不可区分”。如果 top method 是 Fixed OPD，差值应计算 Fixed OPD 与 runner-up，而不是 Fixed OPD 与自身。

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

两种方式必须使用相同的 Codex model/version、temperature、token budget 和 generation seeds `[0,1,2,3,4]`。每种方式生成 5 个候选：seed 0 固定生成 Fixed OPD identity candidate，seed 1–4 生成新候选。identity candidate 只用于接口一致性测试，不计入新颖性或重复率。

生成前先冻结 `tensor_contract.json`。每个可用输入至少记录 name、shape、dtype、device、valid mask、support、producer、是否需要额外 forward 和是否 stop-gradient。`available_in_current_code=false` 的输入不能进入第一批可执行候选。

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

自由生成只允许实现受限的 loss 函数，不能自由修改训练循环。候选代码禁止文件、网络、子进程、环境变量访问，禁止动态 import、反射、`eval`、`exec`，禁止修改全局 RNG、model parameters、optimizer 或输入 tensor。静态安全检查失败的代码不能进入训练进程。

### 4.3 方式二：组件组合

一个候选 loss 由以下部分组成。

#### Divergence

- RKL/OPD；
- FKL；
- FKL/RKL mixture；
- JSD。

每个 divergence 必须声明计算 support。只有取得 full-vocabulary 分布时才能称为 exact FKL/JSD；sampled-token 或 top-k 实现必须在名称和公式中明确近似方式，并定义 renormalization、support mismatch 和 missing probability mass 的处理。

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

entropy、overlap、trajectory correctness 和 normalized training step 用作 weight/gate 时默认 stop-gradient。本列表表示研究候选，不表示这些输入当前都已实现；第一批 schema 只能开放 `tensor_contract.json` 中确认可用的输入。

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
- effective sample size constraint。

gradient norm penalty 不进入 v1：真实 gradient norm 通常在 backward 后才可观测，可能需要二阶梯度或修改训练循环。

第一版最多允许：

- 一个 divergence；
- 一个 weight function；
- 一个 normalization；
- 一个 optional regularization。

所有参数必须有有限范围和默认值。v1 中 mixture coefficient 位于 `[0,1]`；active token weight 经过确定性 projection 后同时满足 `[0,2]` 和 active-token mean 1；active set 为空或 raw weights 全为 0 时回退到 Fixed OPD unit weights 并记录 fallback；没有 valid token 时返回有限 zero loss 并记录 empty-batch event。sign-flipped weight 只能作为 diagnostic，不进入安全候选集合。

### 4.4 比较方法

向两种方式提供完全相同的：

- OPD baseline；
- 允许使用的输入；
- 当前训练统计；
- 禁止修改的内容；
- 现有 loss 实现；
- `tensor_contract.json`；
- 候选输出 schema 和数值约束。

每个候选必须输出 candidate ID、generation mode/seed、tensor contract version、数学定义、使用的输入、divergence support、参数值与范围、normalization/fallback、Python 实现、code hash、与 Fixed OPD 的预期关系和已知数值风险。缺少必要字段的候选记为 schema failure，不由人工补全。

按以下顺序检查候选：

1. schema、数学定义和 support 完整性；
2. AST/import/I/O/副作用安全检查；
3. tensor contract、允许输入和 stop-gradient 检查；
4. 重复候选和确定性 code/config hash；
5. synthetic boundary cases；
6. recorded real batch forward/backward；
7. finite loss/gradient、mask、weight、projection 和 fallback；
8. 1–5 step smoke test；
9. Fixed OPD identity test；
10. 人工修改量。

synthetic tests 至少覆盖 padding、空 active set、全零/极端 raw weights、bf16/fp32 和 normalization fallback。上一步失败的候选不进入下一步，所有失败均保留原始响应和原因。

identity candidate 与仓库 Fixed OPD reference 使用相同 batch，要求 fp32 loss `rtol≤1e-5, atol≤1e-6`，bf16 loss `rtol≤1e-3, atol≤1e-4`，fp32 gradient cosine similarity `≥0.999`、relative L2 error `≤1e-3`，并保持 mask、empty-batch 和 fallback 行为一致。

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

“能够表示”和“当前能够执行”必须分开记录。缺少 full-vocabulary 分布时，Fixed FKL/JSD 只能标为 top-k 或 sampled approximation。`0.5×/2.0× OPD` 作为 projection 之外的整体 loss-scale diagnostic；shuffled/sign-flipped weight 必须标为 diagnostic-only，与可进入训练搜索的安全 action 隔离。

### 4.6 执行顺序

本任务不等待任务一确定最终训练步数，按以下顺序执行：

1. 对全部候选做静态检查；
2. 对合法候选做小 batch forward/backward；
3. 对通过检查的候选做 1–5 step smoke test；
4. 如果资源允许，从两种方式中各选择一个候选，按临时的 100-step 标准运行。

100-step 训练结果不是本任务的必要验收条件。本任务首先判断两种方式能否稳定产生合法、可执行和可比较的 loss。

### 4.7 比较指标

比较两种方式的：

- schema 完整率；
- 合法候选比例；
- 确定性 code/config 转换率；
- forward/backward 通过率；
- smoke test 通过率；
- 与 Fixed OPD 的一致性；
- 重复候选比例；
- 人工修改量；
- 是否能够自动检查；
- 是否能够控制 loss scale；
- 是否能清楚说明候选之间的差异。

人工修改量分为：零修改、格式修复、局部逻辑修复和实质性重写。需要实质性重写的候选不计为自动生成成功。

一种方式通过最低可用标准需要同时满足：

1. Fixed OPD identity candidate 通过全部阈值；
2. validator 能阻止发现的安全违规候选进入执行；
3. 4 个新候选中至少 3 个 schema 完整、合法并可完成 forward/backward，且不需要实质性重写；
4. 至少 2 个新候选通过 smoke test；
5. 所有接受候选能够从原始响应确定性地产生相同 code/config hash。

每种方式只有 4 个新候选，因此这些比例只用于本次 pipeline pilot，不能外推为 Codex 的长期生成成功率。

如果两种方式均通过且前四项没有实质差异，第一版优先组件组合，因为更易审计和复现。自由生成只有在同样通过安全和执行标准，并产生组件 grammar 无法表达的合法候选时才优先。两种方式均未通过时，结论为“当前 action 接口不足”，不能强行二选一。

### 4.8 Codex 分析要求

根据比较结果，要求 Codex：

1. 推荐自由生成或组件组合；
2. 说明推荐依据；
3. 指出当前组件是否完整；
4. 推荐保留、删除或增加的组件；
5. 给出第一批可以进入训练实验的候选 loss。

### 4.9 提交材料

1. 两种方式使用的 prompt；
2. Codex model/version、generation 参数和 paired generation seeds；
3. `tensor_contract.json` 和 validator 规则；
4. 每种方式生成的 5 个原始候选；
5. 候选的数学公式、resolved config、代码和 hash；
6. 安全、静态和数值边界检查结果；
7. forward/backward 和 smoke test 结果；
8. Fixed OPD identity test 的逐项误差；
9. 两种方式的比较表；
10. 推荐的 action 定义；
11. 第一批可执行候选及失败候选列表。

### 4.10 验收标准

必须明确回答：

- 第一版使用自由生成还是组件组合；
- 推荐的 action 结构；
- 允许使用的输入；
- 允许使用的组件；
- 参数范围和归一化规则；
- diagnostic-only action 如何隔离；
- 两种方式是否分别通过最低可用标准；
- 哪些候选可以进入下一轮训练实验；
- 如果均未通过，需要修改 tensor contract、schema 还是 validator。

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

资源不足时优先保证实验有效性，顺序为：seed/目录/manifest/逐题结果正确，Fixed OPD 两个 training seeds，三种方法的 seed 0，初步最优候选补 seed 1，其他候选补 seed，负对照，最后才是 200/300-step 长期确认。不能为了按时给出数字而降低无效实验标准或使用 Locked Test。

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
