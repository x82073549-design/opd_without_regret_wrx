# Evolutionary Survey of On-Policy Distillation in Large Language Model Post-Training

## Theoretical Mechanics of the Paradigm Shift

The alignment and capability scaling of Large Language Models (LLMs) have progressed rapidly, driven by post-training methodologies that translate raw text prediction into structured problem-solving behavior. While classical supervised fine-tuning (SFT) establishes the foundational parameters of downstream instruction compliance, transferring these multi-step logical execution paths to smaller, computationally efficient student models remains a primary engineering bottleneck. Traditional offline distillation paradigms rely on static, teacher-generated corpora to train the student via next-token cross-entropy. Under this offline framework, the student conditions exclusively on flawless expert prefixes during training, yet must generate text autoregressively at deployment, conditioning on its own historically generated tokens.

This structural divergence introduces exposure bias. When the student model commits a minor prediction error early in a sequence, it enters an out-of-distribution state (a prefix) that it never encountered during its offline training. Lacking the capability to recover from self-generated errors, these discrepancies compound quadratically over a sequence horizon $T$, yielding an expected compounding error bound of $O(\epsilon T^2)$, where $\epsilon$ represents the per-step error. For high-depth deductive tasks where long sequence paths are standard, this quadratic error propagation severely degrades small-model execution precision.

On-Policy Distillation (OPD) reorganizes this training loop by shifting the sampling distribution from a static, pre-recorded teacher dataset to the student’s own evolving policy. By forcing the student to generate its own rollout trajectories, the model populates the state space it will visit at deployment. The teacher then acts as an online supervisor, scoring and correcting the tokens generated along these on-policy trajectories. Translating autoregressive generation into a sequential decision-making process reduces the quadratic error bound to a linear scaling of $O(\epsilon T)$. The sequence-level Kullback-Leibler (KL) divergence under student-generated trajectories decomposes into a sum of per-step conditional KL divergences evaluated along the student's own trajectory:

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim \pi_\theta} \left( \dots \right)$$

By evaluating the teacher's vocabulary-level probabilities over prefixes the student actually visits, OPD provides a dense gradient signal that bypasses the high-variance policy updates characteristic of reinforcement learning with sparse rewards. The orthogonal dimensions of post-training paradigms are systematically decoupled by analyzing the interaction between the prefix source (where the sequence context originates) and the token-level KL divergence direction.

| Prefix Source | KL Divergence Direction | Paradigm / Equivalent Classical Regime | Key Properties and Tradeoffs |
|---|---|---|---|
| Teacher-Generated (Off-Policy) | Forward KL ($P_T \parallel P_S$) | Off-Policy Supervised Fine-Tuning (SFT) | Computations are highly efficient but suffer from severe exposure bias, yielding compounding errors scaling at $O(\epsilon T^2)$. |
| Student-Generated (On-Policy) | Forward KL ($P_T \parallel P_S$) | DAgger-Style On-Policy SFT | Matches student prefixes but forces a mode-covering behavior, which often dilutes the precision of small models. |
| Teacher-Generated (Off-Policy) | Reverse KL ($P_S \parallel P_T$) | Offline Reinforcement Learning Distillation | Leverages high-quality static states but operates completely off-policy, restricting active feedback to pre-recorded paths. |
| Student-Generated (On-Policy) | Reverse KL ($P_S \parallel P_T$) | Classic On-Policy Distillation (OPD) | Optimizes student performance on visited states to reduce error bounds to $O(\epsilon T)$, though it causes severe diversity collapse. |

This decoupling demonstrates that the classical division between supervised imitation and reinforcement learning is not a binary choice, but a multi-dimensional design space where prefix distributions and statistical distances dictate training stability.

## Objective Diversification and Adaptive Divergence Control

A critical design consideration in on-policy distillation is the mathematical behavior of the selected divergence metric. When formulating the f-divergence over student-sampled trajectories, the choice of the convex function f defines the student's qualitative behavior across multi-modal output distributions. Minimizing the Forward KL divergence encourages mode-covering, forcing the student to allocate probability mass across all outcomes the teacher considers likely. If the student's parametric capacity is insufficient to bridge these distinct teacher modes, it is forced to average the distributions, resulting in high-entropy, diffuse outputs and hallucinations in the inter-mode space.

Conversely, the Reverse KL divergence operates as a mode-seeking objective. By penalizing the student for generating any token that the teacher disfavors, Reverse KL forces the student to collapse its probability mass onto the teacher's single highest peak, ensuring highly precise and formatting-adherent generations.

However, this mode-seeking behavior introduces diversity collapse. When the teacher distribution exhibits high entropy—signaling a critical branch point with multiple valid continuation paths—Reverse KL yields highly unstable learning signals as the student's parameters aggressively oscillate between these viable alternatives. Standard on-policy distillation causes severe token-level entropy degradation, retaining only a fraction of high-entropy tokens compared to the teacher model. This loss of diversity directly harms the model’s performance on quantitative assessment benchmarks when executing multiple sampled passes.

To address this, Entropy-Aware On-Policy Distillation (EOPD) implements an adaptive divergence blend. The EOPD framework dynamically balances mode-seeking precision with mode-covering robustness by augmenting the standard Reverse KL objective with Forward KL on tokens where the teacher distribution has high entropy. This captures the full range of plausible outputs at uncertain steps while retaining precise imitation elsewhere, preserving the teacher's generative uncertainty without the computational overhead of naive global Forward KL.

Concurrently, researchers have analyzed the boundaries of reward-extrapolation, where the OPD objective is scaled by an amplification coefficient $\lambda > 1$ to sharpen the target distribution and push student performance past the teacher's native capabilities. On structured-output tasks, such as generating calibrated JSON structures, exceeding a critical safety threshold $\lambda^*$ triggers sudden formatting collapse. Under a single-position Bernoulli reduction, this safety threshold is derived in closed form:

$$\lambda^*(p,c) = \frac{\log\left(\frac{p}{1-p}\right)}{\log\left(c - 1 + \frac{p}{1-p}\right)}$$

where $p$ represents the teacher's modal token probability, and $c$ is the importance-sampling clip strength. When the sharpened target's off-modal mass falls below the clipped tail mass $\frac{c}{1-p}$ enforced by importance-sampling clipping, the optimization exits the safe region. The training transitions from format-preserving to format-collapsing, illustrating the delicate boundary between aggressive target sharpening and structural execution stability.

These conceptual mechanics have motivated several distinct architectural innovations in the OPD landscape since the second half of 2025, each targeting specific failure modes of vanilla trajectory alignment.

| Method | Focus | Divergence / Objective | Signal Source | Granularity | Key Innovation |
|---|---|---|---|---|---|
| EOPD | Adaptive Objective | Continuous Forward-Reverse KL blend | White-box Logits | Token | Blends Forward and Reverse KL dynamically based on teacher entropy to mitigate diversity collapse. |
| TIP | Token Dynamics | Adaptive Token Weighting (Soft-OR) | White-box Logits | Token | Identifies overconfident, low-entropy but high-divergence states to prune up to 90% of token compute. |
| SCOPE | Signal Quality | Correctness-routed Dual-Path (MLE/OPD) | White-box Logits | Sequence/Token | Routes correct rollouts to student-perplexity MLE and incorrect ones to teacher-perplexity KL. |
| SOD | Step-wise Optimization | Step-level divergence-reweighted OPD | White-box Logits | Step / Token | Adaptively decays distillation strength based on step-level divergence to mitigate corrupted state signals. |
| OPSD | Self-Evolution | Privileged Information KL | Self-Generated (PI) | Token | Replaces external teachers by using same model conditioned on privileged information as self-teacher. |
| PW-OPSD | Self-Evolution | Sigmoid Position-Weighted Forward KL | Self-Generated (PI) | Token | Mitigates information leakage in OPSD via sequence position weighting representing trajectory-level reliability. |
| TCOD | Multi-Turn Agents | Forward/Backward temporal curriculums | Trajectory Rollout | Turn / Token | Restricts active trajectory steps to mitigate Trajectory-Level KL Instability from compounding errors. |
| VA-OPD | Multimodal (VLM) | Visual Advantage Grouped KL | Image Counterfactual | Rollout / Token | Evaluates teacher reliance on visual details to prevent language dilution via rollout & token grouping. |
| DiffusionOPD | Continuous Models | Per-Step KL Mean-matching | SDE / ODE Denoising | Transition step | Lifts OPD to continuous-state Markov chains, bypassing policy-gradient variance via analytic KL. |

These advancements demonstrate a structural shift toward reliability-aware, highly targeted post-training paradigms that explicitly reject uniform token imitation.

## Fine-Grained Token Selection and Information Dynamics

A core developmental focus in modern OPD is the optimization of gradient information density. Applying a uniform distillation loss to every generated token is structurally inefficient and introduces substantial gradient noise, as a significant portion of student-sampled sequences consists of trivial, already mastered tokens.

The Token Importance in On-Policy Distillation (TIP) framework addresses this by profiling token-level information dynamics along two axes: student entropy $h_t$ (representing prediction uncertainty) and teacher-student divergence $\delta_t$ (representing policy misalignment). This formulation organizes the token space into a clear, four-quadrant taxonomy:

* **Quadrant 1 (High Entropy, High Divergence):** This region represents critical decision points where the student is uncertain and misaligned with the teacher, requiring strong corrective gradients to establish proper logical paths.
* **Quadrant 2 (High Entropy, Low Divergence):** This region contains underconfident predictions where the student's distribution is diffuse but centered on correct teacher alternatives, requiring stabilization gradients.
* **Quadrant 3 (Low Entropy, High Divergence):** This region isolates overconfident student errors, where the student confidently generates incorrect tokens. These errors are structurally invisible to entropy-only selection rules.
* **Quadrant 4 (Low Entropy, Low Divergence):** This region contains already solved tokens that provide negligible learning signal, making up the vast majority of standard sequence lengths.

To capture the highly informative but entropy-blind Quadrant 3 overconfident states, the TIP framework introduces a parameter-free Soft-OR score. First, the token-level Forward KL divergence is computed:

$$\delta_t^{\text{fwd}} = D_{\text{KL}}\left(P_T(\cdot \mid c_t) \parallel P_S(\cdot \mid c_t)\right)$$

The raw student entropy $h_t$ is clipped at the 98th batch percentile and min-max normalized to $\hat{h}_t \in (0, 1)$. This yields a normalized confidence score:

$$\text{conf}_t = 1 - \hat{h}_t$$

The Soft-OR score combines these parameters to isolate critical tokens:

$$SO_t = \text{conf}_t \cdot \delta_t^{\text{fwd}}$$

By training exclusively on the top 10% to 20% of tokens sorted by this Soft-OR score, student models match or exceed the performance of 100% all-token distillation while reducing peak training memory footprint by up to 58%.

Further expanding the selective supervision paradigm, Signal-Calibrated On-Policy Distillation Enhancement (SCOPE) addresses signal quality heterogeneity by routing student rollouts by correctness. Within each prompt's trajectory group, rollouts are split into correct paths ($\Omega_c$) and incorrect paths ($\Omega_w$). Correct trajectories represent valid logical pathways discovered by the student’s own policy. Rather than imposing strict teacher KL matching—which risks suppressing valid alternative strategies—SCOPE applies a student-perplexity-weighted Maximum Likelihood Estimation (MLE) to reinforce these successful paths:

$$w_i^{\text{stu}} = \frac{PPL_S(y_i \mid x)^{1/\tau}}{\sum_{j \in \Omega_c} PPL_S(y_j \mid x)^{1/\tau}}$$

$$\mathcal{L}_{\text{MLE}} = - \sum_{t=1}^{|y_i|} w_i^{\text{stu}} \cdot \log \pi_\theta(y_{i,t} \mid y_{i,<t}, x)$$

This concentrates reinforcement on high-difficulty, low-confidence completions at the student's capability boundary. Conversely, incorrect trajectories require active teacher correction. Because teachers conditioned on flawed student prefixes can generate noisy distributions, SCOPE weights the on-policy distillation by the inverse of the teacher's perplexity to prioritize instances where the teacher demonstrates stable, corrective mastery:

$$w_i^{\text{tea}} = \frac{PPL_T(y_i \mid x)^{-1/\tau}}{\sum_{j \in \Omega_w} PPL_T(y_j \mid x)^{-1/\tau}}$$

$$\mathcal{L}_{\text{OPD}} = \sum_{t=1}^{|y_i|} w_i^{\text{tea}} \cdot D_{\text{KL}}\left(\pi_T(\cdot \mid y_{i,<t}, x) \parallel \pi_\theta(\cdot \mid y_{i,<t}, x)\right)$$

By separating student exploration reinforcement from targeted expert correction, SCOPE resolves the classic trade-off between imitation precision and diversity collapse.

Similarly, the Step-wise On-policy Distillation (SOD) framework adaptively modulates distillation strength at each step based on step-level divergence. Instead of applying uniform weights, the SOD objective scales token losses by a step-specific reliability weight $w_k$:

$$\mathcal{L}_{\text{OPD}}^{\text{step}} = \mathbb{E}_{y \sim \pi_\theta} \left( \sum_{k=1}^{K+1} w_k \sum_{t \in I_k} \left(\log \pi_\theta(y_t \mid y_{<t}) - \log \pi_{\text{teacher}}(y_t \mid y_{<t})\right) \right)$$

When the student's intermediate trajectories diverge sharply from the teacher (e.g., during corrupted tool-call states), SOD decays $w_k$, preventing corrupt gradients from destabilizing the student's base parameters.

## Information-Asymmetric Self-Evolution Systems

The resource costs associated with querying large external frontier models have accelerated the development of teacher-free, self-evolved post-training systems. On-Policy Self-Distillation (OPSD) enables a single model to act simultaneously as both teacher and student. Under this framework, the teacher role is instantiated by conditioning the model on privileged information (r), such as verified reference traces, solution hints, or programmatic environment feedback. The student role is unconditioned, receiving only the prompt (x). The model is trained to minimize the token-level distributional divergence between these two roles over student-sampled rollouts:

$$\mathcal{L}_{\text{OPSD}} = \mathbb{E}_{y \sim p_S(\cdot \mid x)} \left( \dots \right)$$

Although OPSD improves training token efficiency over standard reinforcement learning methods like GRPO, formal analyses reveal a fundamental mathematical vulnerability in asymmetric distillation objectives. Under information asymmetry, the target distribution contains an irreducible mutual information gap $I(Y_t; R \mid X, Y_{<t}) > 0$ that the unconditioned student policy cannot resolve. Minimizing this asymmetric objective forces the student to mimic distributions that depend on hidden variables.

This causes privileged information leakage, where the student's parameters learn to reference the invisible privileged context (e.g., explicitly outputting phrases like "according to the reference solution") during test-time inference. This leakage intensifies over training, leading to performance degradation and stagnating KL divergence.

To resolve this leakage, Position-Weighted OPSD (PW-OPSD) applies an increasing sigmoid positional weight to the sequence-level loss. Recognizing that early tokens are highly sensitive to privileged context, PW-OPSD attenuates the loss on prefix tokens and intensifies it toward the later steps where the dependency on the privileged prefix is resolved.

Concurrently, Direction-Adaptive Self-Distillation (DASD) reframes this setup from uniform imitation to entropy-routed directional supervision. High-entropy tokens are pushed away from the privileged teacher to maintain exploration, while low-entropy tokens are pulled toward the teacher to secure formatting stability.

To stabilize these interactions, the UniSD (Unified Self-Distillation) framework organizes evolutionary learning around three axes:

* **Supervision Reliability:** Implements multi-teacher agreement to down-weight targets with high disagreement, and applies token-level contrastive learning to separate valid supervision from plausible but incorrect alternatives.
* **Representation Alignment:** Integrates feature matching to align internal representations alongside output vocabulary logit distributions.
* **Optimization Stabilizers:** Integrates Exponential Moving Average (EMA) teacher models and vocabulary-level divergence clipping to smooth the evolving target and mitigate gradient spikes.

These safety constraints allow self-evolving systems to stabilize and continue improving without relying on larger external models.

## Multi-Turn Temporal Optimization and Curriculums

While OPD effectively stabilizes single-turn tasks, extending it to multi-turn agent environments (e.g., interactive planning, embodied control, and sequential web navigation) exposes a critical vulnerability known as Trajectory-Level KL Instability.

During multi-turn interactions, student policies generate sequential actions that modify the active environment, yielding new observations that condition subsequent generations. In this interactive loop, early student action errors compound over time. As the student agent drifts further from the teacher's expected state distribution, the teacher model is forced to evaluate highly out-of-distribution, corrupted prefixes. Under these conditions, the teacher's next-token distributions become uncalibrated, assigning progressively lower probabilities to the student's generated tokens. This causes the per-turn KL divergence to escalate continuously during training, resulting in a sudden drop in success rates and severe training instability.

To stabilize long-horizon agent learning, the Temporal Curriculum On-Policy Distillation (TCOD) framework introduces a dynamic pacing strategy. Instead of exposing the student to the entire trajectory from the start, TCOD limits the active optimization sequence depth to a curriculum horizon $k$:

$$k = \min(k_{\text{start}} + \lfloor \eta n \rfloor, T_{\text{max}})$$

where $n$ represents the training iteration, $\eta$ is a pacing parameter controlling the curriculum growth rate, and $T_{\text{max}}$ is the maximum interaction depth. TCOD implements this curriculum via two routing schedules:

* **Forward-to-Backward (TCOD-F2B):** Restricts the student to early, stable steps of the trajectory and progressively extends the distillation horizon deeper into the environment. This ensures the student masters prefix states before navigating deep, error-prone horizons.
* **Backward-to-Forward (TCOD-B2F):** Commences distillation from the final steps of the trajectory—where task outcomes are resolved—and progressively extends supervision backward toward the initial turn.

By isolating active optimization steps, TCOD prevents early compounding errors from corrupting the online supervision signal, allowing student agents to maintain stable KL trajectories and successfully generalize to tasks where the teacher model itself fails.

## Multi-Domain Adaptations: VLMs and Continuous Process Diffusion

The development of on-policy distillation has expanded beyond text models, establishing new frameworks for Vision-Language Models (VLMs) and continuous Diffusion models.

### Visual-Advantage On-Policy Distillation (VA-OPD)
Applying standard OPD to VLMs exposes a visual dilution failure mode. Because sequence generation is dominated by linguistic formatting tokens, uniform KL minimization over the entire rollout averages out the visual alignment signals. Consequently, the student model learns language templates but fails to attend to fine-grained visual details, generating identical output tokens even when the input image is degraded.

VA-OPD addresses this dilution by introducing the Visual Advantage (VA) metric. For each generated token $y_t$, VA quantifies the teacher’s dependence on fine-grained visual detail by calculating the counterfactual log-probability difference between a pristine image $v$ and a degraded image $\tilde{v}$:

$$a_t = \max\left(\log p_T(y_t \mid v, q, y_{<t}) - \log p_T(y_t \mid \tilde{v}, q, y_{<t}), 0\right)$$

VA-OPD leverages this token-level signal at two levels of granularity:
* **Rollout-level Reweighting:** Sibling rollouts are weighted based on their trajectory-averaged VA, $\bar{a}^{(k)}$, relative to their siblings, focusing gradients on paths with high visual dependence:

$$w^{(k)} = \frac{\exp(\hat{z}^{(k)}/\tau)}{\sum_j \exp(\hat{z}^{(j)}/\tau)}$$

* **Token-level Grouped KL:** Tokens are sorted by their VA values and partitioned into high-VA ($V^{(k)}$) and low-VA ($L^{(k)}$) groups. The KL divergence is averaged within each group separately, preventing visual alignment gradients from being diluted by standard language patterns:

$$\mathcal{L}_{\text{group}}^{(k)} = \lambda \cdot \frac{1}{|V^{(k)}|} \sum_{t \in V^{(k)}} D_{\text{KL},t} + (1-\lambda) \cdot \frac{1}{|L^{(k)}|} \sum_{t \in L^{(k)}} D_{\text{KL},t}$$

### Continuous State-Space On-Policy Distillation (DiffusionOPD)
While discrete vocabulary targets define LLM distillation, DiffusionOPD lifts the on-policy paradigm to continuous-state Markov processes. It models the reverse denoising process as a discrete-time Markov chain induced by a reverse-time SDE discretized via the Euler-Maruyama scheme on a schedule $1 = t_1 > t_2 > \dots > t_N = 0$.

Under this framework, both the student and teacher transition kernels are Gaussian distributions that share the same scheduler-dependent covariance matrix $\bar{\sigma}^2 I$. This covariance identity allows the reverse KL divergence over the continuous transitions to be derived analytically in closed form:

$$D_{\text{KL}}\left(p_{\text{teacher}}(\cdot \mid x) \parallel p_{\text{student}}(\cdot \mid x)\right) = \frac{1}{2\bar{\sigma}^2} \|\mu_{\text{teacher}}(x) - \mu_{\text{student}}(x)\|^2$$

Rather than relying on high-variance policy gradient updates (like PPO-style actor models) that inject Gaussian noise gradients, DiffusionOPD directly optimizes this closed-form mean-matching objective. This minimizes gradient variance, stabilizing multi-task learning in continuous diffusion state-spaces.

## Industrial Deployment and Co-Evolutionary Synthesis

The transition of on-policy distillation from theoretical research to industrial post-training pipelines is driven by its computational and sample efficiency. Empirical results across multiple quantitative evaluation suites demonstrate the capability of modern OPD and its stabilized variants to compress high-tier logical competence into deployable, compact models.

| Base Student Model | Target Paradigm | Evaluation Benchmark | Vanilla OPD Baseline | Proposed Method | Performance Delta |
|---|---|---|---|---|---|
| Qwen3-0.6B-Base | EOPD | Six-Benchmark Math Avg | Standard Baseline | +1.16 Avg / +1.37 Pass@8 | +1.37 Pass@8 |
| Qwen3-1.7B-Base | EOPD | Six-Benchmark Math Avg | Standard Baseline | +2.39 Pass@8 | +2.39 Pass@8 |
| Qwen3-4B-Base | EOPD | Six-Benchmark Math Avg | Standard Baseline | +5.05 Pass@8 | +5.05 Pass@8 |
| Distill-Qwen-1.5B | SCOPE | AIME 2024 (Pass@32) | 71.68% (approx) | 77.90% | +6.22% |
| Distill-Qwen-1.5B | SCOPE | AIME 2025 (Pass@32) | 45.71% (approx) | 50.90% | +5.19% |
| Qwen3-1.7B | TCOD | ALFWorld (Success Rate) | Near-Zero % | Decisive Recovery | +15.71 points |
| Qwen2.5-7B | TCOD | ALFWorld (Success Rate) | Standard Baseline | +15.71 points | +15.71 points |
| 0.6B-Scale Student | SOD | AIME 2025 Accuracy | Standard Baseline | 26.13% | Up to +20.86% rel. |

Industrial pipelines—including Qwen3, DeepSeek-V4, GLM-5, and MiMo-V2-Flash—have integrated adaptive on-policy distillation as an essential optimization layer. Rather than treating distillation and reinforcement learning as isolated, alternative paradigms, modern post-training pipelines combine them into a sequenced, co-evolving post-training pipeline.

Under this sequenced framework, post-training proceeds in three distinct phases:
1.  **Off-Policy Initialization:** The student model undergoes supervised learning on high-quality teacher demonstrations to establish initial domain priors and formatting rules.
2.  **Fine-Grained Adaptive OPD:** The student generates on-policy trajectories, and its parameters are optimized using reliability-weighted, selective token objectives (e.g., TIP or SCOPE). This resolves exposure bias, stabilizes intermediate planning, and transfers the teacher’s generative distribution with high token efficiency.
3.  **Reinforcement Learning with Verifiable Rewards (RLVR):** The distilled student is optimized using reinforcement learning (e.g., GRPO) under programmatically verified rewards. Because the student's policy entropy and logical paths were aligned during the OPD phase, the model avoids early entropy collapse, allowing it to explore alternative solutions and transcend the performance ceiling of its original teacher model.

This co-evolutionary post-training paradigm leverages on-policy distillation to bridge the gap between passive imitation and active self-exploration, enabling small, deployable models to achieve high execution accuracy in complex interactive environments.