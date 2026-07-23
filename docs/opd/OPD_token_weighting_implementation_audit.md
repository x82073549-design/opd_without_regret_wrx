# OPD Token Weighting 实现审计与改动清单

> 审计对象：当前分支 `my-opd-upload-20260629`，基线提交 `c1616fb`。本文件只描述当前调用链、缺口和实施顺序；实验结论以《OPD 跨粒度统一信号：研究契约 v0》和《本周实验计划》为准。

## 1. 当前实际调用链

第一轮 pilot 应使用根目录 `on_policy_distillation.sh`，而不是 `verl_example/opd.sh`。

当前可运行路径为：

1. `on_policy_distillation.sh`
   - `algorithm.adv_estimator=token_reward_direct`
   - `reward_model.enable=True`
   - `actor_rollout_ref.rollout.log_prob_top_k>0`
   - 通过 `+algorithm.token_feature_weighting.*` 打开 weighting。
2. `verl/verl/workers/actor/dp_actor.py`
   - student 前向产生 `old_log_probs`、`entropys`、`student_top_k_ids`、`student_top_k_log_probs`；
   - `compute_distillation_reward` 产生 top-k OPD 的 `rm_scores`。
3. `verl/verl/workers/fsdp_workers.py`
   - teacher 在 student rollout 上计算 `teacher_top_k_*`、`teacher_entropy`、`overlap_mask` 等张量。
4. `verl/verl/workers/reward_manager/naive.py`
   - 规则验证器计算 student correctness，并把 teacher 的 `rm_scores` 作为训练 reward；
   - correctness 以 `true_reward_score` 返回。
5. `verl/verl/trainer/ppo/ray_trainer.py`
   - 将 `rm_scores` 放入 `token_level_rewards`；
   - `apply_token_feature_weighting` 在 advantage 计算之前修改 token reward；
   - `token_reward_direct` 将加权 reward 直接作为 advantage。

这个插入位置能够实现“只重新分配 token OPD 信号，不改变 rollout 数和 optimizer step”。

`verl_example/opd.sh` 使用 `distillation.*` 配置，但当前 vendored `verl/verl` 源码中没有对应的新 distillation trainer 实现。除非运行环境安装的是另一版本 verl，否则该脚本与当前 weighting 代码不属于同一调用链，不能用于本轮实验。

## 2. 已有能力

- 支持 `teacher_confidence` 单一特征；当前定义为 teacher top-1 概率：

  \[
  c_t^T=\exp(\log p_T^{\mathrm{top1}})
  \]

- 支持 `normal`、`reverse`、`uniform` 三种方向。
- 支持 2D sampled-token reward 和 3D top-k reward 的广播加权。
- 序列内 z-score，padding token 不参与统计。
- clipping 后重新归一化，数值上维持每序列平均权重约为 1。
- 已有 5 个 CPU 单元测试，覆盖均匀、无 clipping 时的正反对称、mask、3D reward 广播和缺失输入报错。
- 启动脚本能够记录主要模型、数据、top-k 与 weighting 配置。

## 3. 阻塞正式 pilot 的 P0 缺口

### P0.1 当前非负处理会破坏严格正反对称

当前实现先构造中心化权重，再分别 clipping 和 renormalize。normal 与 reverse 一旦触发 clipping，通常不再满足：

\[
w_t^+ + w_t^- = 2
\]

现有对称性测试通过 `min_weight=-10` 主动绕过了 clipping，因此没有覆盖实际启动配置 `min_weight=0`。

修复：按实验计划对整条序列的中心化扰动做统一安全缩放，不做逐 token clipping：

\[
d_t^{safe}=\lambda(d_t-\bar d),\qquad
\lambda=\min\left(1,\frac{1-\epsilon_w}{\max(\max_t|d_t-\bar d|,\epsilon)}\right)
\]

随后统一定义 \(w^+=1+d^{safe}\)、\(w^-=1-d^{safe}\)。需要记录安全缩放触发率和平均 \(\lambda\)。

### P0.2 缺少随机打乱负对照

当前 `direction` 只接受 `normal/reverse/uniform`，无法区分“权重分布形状”与“特征对应关系”的效果。

修复：增加 `shuffled` 实验臂。它应先构造 normal 权重，再仅在每条 response 的有效 token 内置换；必须：

- 保持每条序列权重多重集合不变；
- padding 权重保持 1；
- 使用显式 `shuffle_seed`，在相同 batch 顺序和 step 下可复现；
- 记录跨序列汇总的 feature–weight Spearman 相关；
- 不复用训练/rollout RNG，避免随机对照改变 student rollout。

### P0.3 只支持 teacher confidence，四特征路径未闭合

`_get_token_feature_values` 目前对其他特征直接报错。首轮需要固定以下定义：

| 配置名 | 张量来源 | 首版操作化定义 |
| --- | --- | --- |
| `teacher_confidence` | `teacher_top_k_log_probs` | `exp(top1 teacher log-prob)` |
| `student_entropy` | `entropys` | full-vocab student entropy |
| `overlap_ratio` | `overlap_mask` | student top-k 中也属于 teacher top-k 的比例，即 `mean(K)` |
| `fire_confidence_product` | student/teacher top-k log-prob | `exp(student top1) × exp(teacher top1)` |

注意：`entropys` 当前在 `apply_token_feature_weighting` 之前被清理。必须延后清理，或在清理前把所选标量特征缓存为 2D `token_feature_values`。推荐后者，以免 teacher/student 大张量在 trainer 中存活更久。

### P0.4 测试环境与模块耦合

当前测试导入整个 `ray_trainer.py`，会连带导入 Ray、transformers 和全部 actor config。本机默认 Python 3.13 环境在测试收集阶段因 `AutoModelForVision2Seq` 缺失失败，尚未真正执行 5 个测试。

修复：把纯函数和特征提取移动到独立模块，例如：

`verl/verl/trainer/ppo/token_feature_weighting.py`

CPU 测试直接导入该模块；`ray_trainer.py` 只负责在正确时间调用。这样权重数学测试不依赖 GPU、Ray 或完整 transformers 版本。

### P0.5 缺少可比较的 held-out reverse-KL 评测器

当前训练日志包含 top-k/overlap 分析和任务 accuracy 验证，但没有一个锁定 probe states 的 held-out reverse-KL evaluator。只比较不同 checkpoint 各自 on-policy rollout 上的 KL，会同时改变状态分布和局部分歧，解释不唯一。

推荐新增离线筛选流程：

1. 用固定初始 student checkpoint 在 validation prompts 上生成并冻结 probe trajectories；
2. 预计算 teacher 在这些 states 上的 top-k/logits；
3. 对每个训练条件的 checkpoint，在同一 states 上重算 student 分布；
4. 使用相同 top-k 近似与 mask 计算 reverse-KL；
5. 单独运行正常 generation accuracy，检验 KL 与 accuracy 是否同向。

固定-state reverse-KL 是筛选代理，不替代最终 on-policy accuracy。

## 4. P1：正式方向分析需要的日志

当前日志只有全 batch 的 feature mean/std、weight min/max/mean 和 clipped fraction，不足以完成研究计划中的分层与 harness 验收。

至少新增：

- 每序列权重均值最大误差；
- normal/reverse 对称误差；
- `degenerate_feature` 序列比例；
- 安全缩放触发比例、平均/最小 \(\lambda\)；
- feature–weight Pearson/Spearman；
- response length、截断率、重复率；
- student correct/error 两层的 feature mean、weight distribution 和 OPD reward；
- α、direction、shuffle seed、feature definition version；
- 训练 token、有效 response token、optimizer step 和实际运行配置。

`true_reward_score.sum(-1)>0` 可作为当前数学任务的 student-correct 指示。teacher correctness 需要 teacher 自己生成答案并经 verifier 判分，当前 teacher-on-student-trace 前向无法提供该量；本轮应标记为 unavailable，而不是把 teacher confidence 当作 teacher correctness。

对于逐 token 原始数据，建议只在 exploration 的指定 step/batch dump，内容至少包括 mask、position、feature、weight、OPD reward、student correctness 和匿名 sample id。不要默认每步全量写盘。

## 5. P1：实验配置与矩阵生成

当前只有 `configs/teacher_confidence_normal_alpha1.env`，但文件名写 alpha1、内容实际为 alpha2，容易误标实验。

建议配置维度固定为：

- `TOKEN_FEATURE_NAME`
- `TOKEN_FEATURE_ALPHA`
- `TOKEN_FEATURE_MODE=normal|reverse|shuffled|uniform`
- `TOKEN_FEATURE_SHUFFLE_SEED`
- `TOKEN_FEATURE_DEFINITION_VERSION=v1`

增加 dry-run 模式，仅打印完整 Hydra command/config，不启动 Ray 或模型。增加实验矩阵生成器，并做以下去重：

- uniform 每个训练 seed 只跑一次，不随 feature 和 α 重复；
- smoke test 先覆盖四个 feature × 四个 mode，但只跑极小 batch；
- 完整 R=3 pilot 先跑 normal/reverse；shuffled 优先用于 validation 上有方向迹象的条件；
- accuracy 确认只保留 1–2 个冻结条件。

所有实验必须显式设置训练 seed、rollout seed、shuffle seed 和输出目录，禁止多个条件自动 resume 到同一 checkpoint。

## 6. P1：正确性信息保留

`NaiveRewardManager` 在存在 `rm_scores` 时会重新构造 `reward_extra_info`，当前计算得到的 `acc`、`format_score` 等逐样本字段不会完整保留，只额外放入 `true_reward_score`。

短期分析可以直接使用 `true_reward_score`。长期应修复 reward manager：保留规则验证器原始字典，同时额外加入 `true_reward_score`，避免训练和 rollout dump 中丢失 `acc/pred/format_score`。

## 7. 建议代码结构

| 文件 | 责任 |
| --- | --- |
| `verl/verl/trainer/ppo/token_feature_weighting.py` | 纯权重数学、四特征提取、随机打乱、诊断量 |
| `verl/verl/trainer/ppo/ray_trainer.py` | 在 reward→advantage 边界调用；保留/清理必要张量；记录分层指标 |
| `verl/verl/workers/reward_manager/naive.py` | 不丢失 verifier 的 correctness extra info |
| `on_policy_distillation.sh` | 暴露 mode/seed/version/dump/dry-run，打印可复现配置 |
| `configs/token_feature/` | 小型 overlay 或由矩阵生成器生成的 env 配置 |
| `scripts/opd/run_token_feature_matrix.py` | 去重并生成/提交实验矩阵 |
| `scripts/opd/eval_fixed_probe_kl.py` | 固定 probe states 的 held-out reverse-KL |
| `verl/tests/trainer/ppo/test_token_feature_weighting_on_cpu.py` | 数学、特征、mask、shuffle 和 2D/3D reward 测试 |

## 8. 测试验收矩阵

### 8.1 CPU 单元测试

1. uniform 对所有有效 token 恒为 1；
2. α=0 的 normal/reverse 与 uniform 完全一致；
3. 极偏特征、α=4 时 normal/reverse 仍非负、均值为 1、逐 token 和为 2；
4. padding 和 prompt 不参与统计；
5. 零方差、单 token、全 mask 序列稳定返回；
6. shuffled 与 normal 权重多重集合逐序列相等且可复现；
7. shuffled 跨足够多序列后与 feature 排序近似无关；
8. 四个 feature extractor 的数值与 shape 正确；
9. 2D 与 3D reward 广播正确；
10. 缺失依赖张量时给出包含 feature 名和所需配置的错误。

### 8.2 小型集成测试

1. α=0 与 weighting disabled 的 `token_level_rewards`、advantages、actor loss 相同；
2. 四个 mode 的有效 token 数、训练 token 和 optimizer step 相同；
3. normal/reverse 使用同一 rollout batch 时，只有 reward 权重不同；
4. correctness 分层样本数与 verifier 输出一致；
5. resume 后 shuffle 和实验 metadata 可复现。

### 8.3 GPU smoke test

- 1 个 feature、四个 mode、每个 50–100 update；
- 无 NaN/Inf，安全缩放和 degenerate 指标合理；
- α=0 loss 曲线与 baseline 在随机噪声允许范围内一致；
- 训练结束能由固定 probe evaluator 计算 KL，并由现有 validation pipeline 计算 accuracy。

## 9. 实施顺序

1. 抽离纯模块，修复安全缩放，增加 shuffled 与完整 CPU 测试；
2. 接通四个原始特征，解决 student entropy 生命周期；
3. 增加 correctness 分层、harness 指标和小规模 token dump；
4. 修复启动配置、seed、dry-run 与矩阵去重；
5. 实现固定-state reverse-KL evaluator；
6. 先跑 α=0/disabled 对齐测试，再跑单 feature 四臂 smoke test；
7. smoke 全部通过后才启动 R=3 方向 pilot。

前四步属于训练正确性 P0/P1；第五步属于研究结论的指标闭环。二者都完成前，不应开始大规模 α × feature × seed 扫描。
