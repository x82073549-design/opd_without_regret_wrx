# Codex Evolve OPD Loss 会议更新

日期：2026-08-05
范围：只记录 OPD loss evolving 相关决策、现状和待办；multi-agent failure mode 为另一研究任务，不在本文展开。

## 1. 已确定的方向

### 1.1 Memory 与 action 分离

- 论文中的 loss、仓库已有 loss、团队方案和历史实验结果作为长期 loss memory；
- memory 用来提供设计参考、结果证据和失败案例，不定义封闭 action space；
- action 采用受限 loss 接口内的自由 program generation；
- 旧组件可以作为 memory tag 或实现参考，但不再限制 Codex 只能搭积木；
- 保留副 action，用于指定当前迭代方向；主 action 负责生成新的 loss program。

每条 memory 至少记录：来源、公式、代码位置、tensor/support、配置 hash、parent IDs、训练步数、matched baseline 差值、seeds/uncertainty、运行状态、失败原因和解释。搜索只使用 Training/Validation 结果，Locked Test 不进入 memory。

同一批候选读取相同的 memory snapshot；该批实验结束后再把结果追加到下一版 memory。这样允许模型从实验中学习，同时避免在一批比较中临时改变上下文。

### 1.2 优先搜索宏观 loss 结构

M0 优先探索：

- divergence 或 support；
- token/trajectory weighting；
- gating/filtering；
- normalization 与 fallback；
- 多个 loss 结构的组合。

只修改少量常数或超参数不作为主要 Codex Evolve 结果。超参数范围由人或 Codex 预先定义，具体优化优先使用 grid、random、Bayesian 等常规方法。

### 1.3 先跑通朴素闭环

第一阶段先跑通：

```text
memory snapshot
→ parent selection
→ sub-action / iteration direction
→ loss proposal
→ code write
→ validation and smoke test
→ short training
→ result written back to memory
```

暂不把训练阶段动态切换 loss 作为 M0 必需功能。等固定 loss 搜索积累足够数据后，再研究 Codex 从“探索新 loss”转为“按训练阶段优化已有 loss”。

## 2. 当前发现的风险

### 2.1 连续两轮 loss 文件基本一致

当前对比发现两轮输出的 Python loss 文件基本一致。这不能直接视为有效 evolve，可能原因包括：

- memory 没有被实际读取；
- parent node 没有变化；
- 副 action 没有影响 prompt 或生成；
- 新代码没有写入预期文件；
- 训练进程仍加载旧 module/cache；
- proposal 只改变了无关参数或注释。

后续每轮必须保存 parent IDs、副 action、prompt/memory hash、Python diff、候选文件 hash，并在训练日志中记录实际加载的 loss module hash。除 identity test 外，连续两轮代码无实质差异时应停止搜索并排查链路。

### 2.2 Loss 接口边界尚未充分验证

需要确认上层训练代码实际读取哪些 loss 输入和输出，避免 Codex 返回了额外信息但被静默忽略。执行前冻结 `tensor_contract.json` 和输出 schema，并用 Fixed OPD identity test、forward/backward、数值边界和 1–5 step smoke test 验证接口。

### 2.3 训练成本和运行稳定性

- 当前估计确定训练步数后，单个候选实验约 3 小时；
- Liujiang 的夜间实验因网络/代理故障中断，当前尚无可验收结论；
- 实验代码改动需要及时提交，不能只保存在运行机器工作区。

## 3. Early stopping 方向

考虑两类策略：

1. 人工预注册规则：数值失败立即停止；连续多个 checkpoint 明显低于 matched baseline 且超过噪声范围时停止；
2. Codex 判断：输入中间 Validation、训练诊断和 uncertainty，输出 `continue/stop`、confidence 和理由。

可研究的停止信号包括 confidence collapse、梯度异常、token weight/ESS 塌缩和长期无改进。但 early stopping 必须先在完整 100-step 历史轨迹上回放，报告：

- 预计节省的 GPU time；
- 误停率；
- 是否会停止最终 winner；
- 不同 seeds 下是否稳定。

回放审计通过前只记录停止建议，不实际终止正式 run。Locked Test 不参与 early stopping。

## 4. 代码与分支协作

- 主分支保留稳定、可运行的基础代码；
- Ruxin 在独立 branch 整合 loss memory、论文/现有 loss 和 evolving 流程；
- Liujiang 在独立 branch 保存训练步数和 early-stopping 实验；
- 多机并行时通过 Git 同步代码、config、memory 和结果摘要；
- 每次实验记录 code commit、config hash、data manifest、seed、parent IDs 和 run ID。

当前 pilot 暂时沿用 batch size 64，后续若修改必须建立新的配置版本，不能和已有结果混合。

## 5. 负责人和近期交付

### Ruxin

1. 将现有 loss、论文 loss 和历史实验结果整理进独立 branch；
2. 建立版本化 loss memory 和来源记录；
3. 将 action 改为自由 loss program，保留副 action；
4. 连续运行至少两轮 proposal，提交 Python diff/hash 和运行时加载核验；
5. 跑通 identity、forward/backward 和 smoke test；
6. 说明当前 loss 输入输出边界和上层实际消费字段。

### Liujiang

1. 继续完成训练步数实验；
2. 保留 20/40/60/80/100 的完整轨迹，用于训练步数和 early-stopping 回放；
3. 统计单 run 时间、显存和失败原因；
4. 提出实验结果进入长期 memory 的文件格式；
5. 排查网络依赖，确保中断可恢复且代码及时提交。

## 6. 尚未决定的问题

- parent node 的选择规则；
- 副 action 的枚举、选择和更新策略；
- Codex 每轮生成多少候选、保留多少父节点；
- early stopping 的最终阈值与 Codex confidence 校准方法；
- 固定 loss 搜索何时升级为训练阶段动态 loss；
- 长期 memory 规模变大后的检索、压缩或模型微调方案。
