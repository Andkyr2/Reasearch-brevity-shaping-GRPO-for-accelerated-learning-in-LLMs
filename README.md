# Reasearch-brevity-shaping-GRPO-for-accelerated-learning-in-LLMs
Preliminary research and draft of my upcoming paper on brevity shaping for accelerating GRPO training for reasoning in LLMs

# [cite_start]Accelerating Reinforcement Learning for Mathematical Reasoning with Brevity Shaping [cite: 1]

[cite_start]**Kyriakos Ftellehas** [cite: 2]
[cite_start]*University College London* [cite: 2]
[cite_start]*September 23, 2025* [cite: 3]

## [cite_start]Abstract [cite: 5]

[cite_start]Enhancing the multi-step reasoning capabilities of Large Language Models (LLMs) is a key focus in AI research[cite: 6]. [cite_start]Reinforcement learning (RL) has proven effective for this, by rewarding correct reasoning chains, but it is computationally bottlenecked by sampling and evaluating long sequences[cite: 7]. [cite_start]This paper proposes a curriculum-based approach to accelerate RL fine-tuning for mathematical reasoning[cite: 8]. [cite_start]Integrated into a Group Relative Policy Optimization (GRPO) framework, our method first rewards concise, correct answers via a length-penalizing reward function, reducing per-step training time[cite: 9]. [cite_start]It then inverts the reward to promote detailed, step-by-step explanations[cite: 10]. [cite_start]We show that this aggressive shaping requires Kullback-Leibler (KL) divergence regularization against a frozen reference model to prevent policy collapse[cite: 11]. [cite_start]Experiments on GSM8K using Qwen2-0.5B-Instruct yield a 17.4% reduction in wall-clock training time, with a final accuracy of 44% nearly matching the 45% of a standard RL baseline[cite: 12].

## [cite_start]Introduction [cite: 13]

[cite_start]As Large Language Models (LLMs) scale to larger sizes, unlocking their full potential for complex multi-step reasoning such as solving intricate mathematical problems remains a pivotal challenge in AI[cite: 14]. [cite_start]While Chain-of-Thought (CoT) prompting has revolutionized how we elicit step-by-step reasoning from these models, achieving robust and reliable performance demands more than zero-shot techniques[cite: 15]. [cite_start]Reinforcement learning (RL) fine-tuning emerges as a powerful solution, directly rewarding models for correct answers or valid reasoning steps, leading to substantial gains in reasoning accuracy[cite: 16].

[cite_start]Yet, the scalability of RL methods poses a formidable challenge[cite: 17]. [cite_start]Algorithms like Proximal Policy Optimization (PPO) rely on on-policy sampling, necessitating the generation of thousands of lengthy reasoning chains per training step[cite: 18]. [cite_start]This computational overhead, which scales linearly with sequence length and model size, can render RL prohibitively expensive especially for larger models or extended training regimes limiting its accessibility[cite: 19].

[cite_start]To address this critical bottleneck, we present a novel curriculum learning strategy that dramatically accelerates RL fine-tuning without sacrificing performance[cite: 20]. [cite_start]Our innovative approach decouples the discovery of correct answers from their detailed explanation, training the model in two synergistic phases[cite: 21]:

1.  [cite_start]**Brevity Phase:** Aggressively reward concise, correct solutions to minimize sequence lengths, enabling fast generation and updates during early training[cite: 22].
2.  [cite_start]**Elaboration Phase:** Transition by inverting the reward to encourage verbose, step-by-step reasoning, building on the foundational accuracy gained[cite: 23].

[cite_start]This brevity shaping, while potent, risks policy instability[cite: 24]. [cite_start]Our key insight is that incorporating Kullback-Leibler (KL) divergence regularization against a frozen reference model ensures the model can transition from the brevity phase to the elaboration phase, without losing its prior linguistic knowledge[cite: 25, 27].

[cite_start]Integrated into Group Relative Policy Optimization (GRPO) our method delivers compelling results: on the GSM8K benchmark, it achieves 44% accuracy, virtually identical to a standard baseline's 45%, while slashing wall-clock training time by 17.4%[cite: 28]. [cite_start]This can contribute to significant cost cutting in training reasoning models[cite: 29].

## [cite_start]Related Work [cite: 30]

### [cite_start]Reinforcement Learning for Reasoning [cite: 31]

[cite_start]RL improves LLM reasoning via outcome-based rewards (final answer correctness) or process-based rewards (step validity)[cite: 32]. [cite_start]Outcome rewards are scalable but risk rewarding flawed paths[cite: 33]. [cite_start]Our curriculum accelerates outcome-based RL by prioritizing brevity before detail[cite: 33].

### [cite_start]Policy Optimization and Stability [cite: 34]

[cite_start]PPO constrains updates via KL-divergence to avoid collapse, crucial for dense rewards prone to hacking[cite: 35]. [cite_start]Our work emphasizes KL's role in stabilizing aggressive curricula[cite: 36]. [cite_start]When the KL-divergence was high, the model couldn't recover its elaboration capability[cite: 37].

## [cite_start]Methodology [cite: 38]

### [cite_start]Group-Relative Policy Optimization for Reasoning [cite: 39]

[cite_start]GRPO adapts PPO for outcome rewards[cite: 40]. [cite_start]Let $\pi_{\theta}$ be the policy and $\pi_{ref}$ its frozen initial copy[cite: 40]. [cite_start]For each problem in batch D[cite: 41]:

1.  [cite_start]**Sampling:** Generate k chains $\{y_{1},...,y_{k}\}\sim\pi_{\theta}(\cdot|x)$[cite: 42].
2.  [cite_start]**Reward:** Compute curriculum-based $R(x,y_{i})$[cite: 43].
3.  [cite_start]**Advantage:** $\hat{A}(x,y_{i})=R(x,y_{i})-\frac{1}{k}\sum_{j=1}^{k}R(x,y_{j})$[cite: 44].
4.  **Loss:** For each,
    [cite_start]$$L_{x}(\theta)=\frac{1}{k}\sum_{i=1}^{k}[-log~\pi_{\theta}(y_{i}|x)\cdot\hat{A}(x,y_{i})+\beta\cdot KL(\pi_{\theta}(\cdot|x)||\pi_{ref}(\cdot|x))]$$ [cite: 46]
    with sequence-level KL approximation:
    [cite_start]$$KL_{sequence}\approx\frac{1}{|y_{i}|}(log~\pi_{\theta}(y_{i}|x)-log~\pi_{ref}(y_{i}|x))$$ [cite: 48, 49]

### [cite_start]Curriculum-Based Brevity Shaping [cite: 52]

[cite_start]The reward combines correctness $R_{task}$ with length shaping[cite: 53]:
[cite_start]$$R(x,y_{i})=max(0,1-\lambda\cdot max(0,|y_{i}|-35)) \text{ if correct, else 0,}$$ [cite: 54]
[cite_start]where 35 is an empirical short-answer threshold, and $\lambda$ controls shaping[cite: 55].

**Curriculum:**
* [cite_start]**Phase 1 (Brevity, batches 1-200):** $\lambda=0.01$ (penalize length)[cite: 56].
* [cite_start]**Phase 2 (Elaboration, batches 201-225):** $\lambda=-0.01$ (reward length)[cite: 57].
* [cite_start]**Phase 3 (Standard, remainder):** $\lambda=0$ (correctness only)[cite: 58].

## [cite_start]Experiments [cite: 60]

### [cite_start]Setup [cite: 61]

* [cite_start]**Model:** Qwen2-0.5B-Instruct (0.5B parameters)[cite: 63].
* [cite_start]**Dataset:** GSM8K train (7.5K problems), test for evaluation[cite: 64].
* [cite_start]**Hyperparameters:** AdamW, LR $10^{-5}$, batch 32, $k=8$, $\beta=0.1$[cite: 65].
* [cite_start]**Hardware:** NVIDIA A100 GPU[cite: 66].

### [cite_start]Conditions [cite: 67]

1.  [cite_start]**GRPO Baseline:** $\lambda=0$, with KL $(\beta=0.1)$[cite: 68].
2.  [cite_start]**GRPO + Curriculum (No KL):** Curriculum, $\beta=0$[cite: 69].
3.  [cite_start]**GRPO + Curriculum (With KL):** Full method[cite: 70].

## [cite_start]Results and Analysis [cite: 72]

[cite_start]**Table 1: GSM8K test results after one epoch** [cite: 74]

| Method                      | Pass@8 Accuracy | Time (s) | Speedup               |
| --------------------------- | --------------- | -------- | --------------------- |
| GRPO (Baseline)             | 45%             | 16087    | 1.00x                 |
| GRPO + Curriculum (No KL)   | Collapsed       | N/A      | N/A                   |
| GRPO + Curriculum (With KL) | 44%             | 13288    | 1.21x (17.4% faster)  |

---

[cite_start]The curriculum reduces lengths to < 80 tokens early, accelerating training, then recovers[cite: 116]. [cite_start]Without KL, the policy collapses[cite: 116]. [cite_start]With it, KL stays low[cite: 117]. [cite_start]The high entropy in the brevity phase supports learning efficiency[cite: 117]. [cite_start]Overall, we see 17.4% faster training with negligible accuracy loss[cite: 118].

*Note: The following images are representations of the data from the paper.*

**Figure 1: Completion Lengths**
![Figure 1](placeholder_figure1.png)
*Caption: Completion lengths during training. [cite_start]Left: Curriculum shows completion-length drop then recovery at batch 200. Right: Baseline stabilizes at 140 tokens. [cite: 80, 81]*

**Figure 2: KL-Divergence**
![Figure 2](placeholder_figure2.png)
*Caption: KL-divergence during curriculum training (low and bounded). [cite_start]Baseline without explicit KL penalty reaches 0.2. [cite: 96]*

**Figure 3: Token Entropy Distributions**
![Figure 3](placeholder_figure3.png)
*Caption: Token entropy distributions. Left: Full answers. [cite_start]Right: Compressed phase shows 36% higher average entropy, aiding effective learning. [cite: 115]*

## [cite_start]Discussion [cite: 120]

[cite_start]Our curriculum accelerates RL by prioritizing concise solutions before elaboration, but requires KL to avoid forgetting[cite: 121]. [cite_start]Entropy retention ensures the brevity phase drives meaningful updates[cite: 123].

[cite_start]**Limitations:** Potential shortcut learning in brevity; fixed schedules may not generalize; tested on a small model/dataset[cite: 122].

### [cite_start]Future Work [cite: 124]

* [cite_start]Apply to code/logic domains or larger models (e.g., scaling laws for speedup)[cite: 126].
* [cite_start]Hybrid rewards: Outcome in brevity, process in elaboration[cite: 127].
* [cite_start]Adaptive shaping (e.g., entropy-based phase transitions)[cite: 128].
* [cite_start]Ablations on $\lambda$, $\beta$, and verifier integration[cite: 129].

## [cite_start]Conclusion [cite: 130]

[cite_start]We present a brevity-shaping curriculum that cuts RL training time by 17.4% on GSM8K with comparable accuracy, stabilized by KL regularization[cite: 131]. [cite_start]This advances efficient RL for reasoning LLMs[cite: 132].

## [cite_start]References [cite: 133]

[1] Shenzhi Wang, Le Yu, Chang Gao, et al. Beyond the $80/20$ Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. [cite_start]2025[cite: 134, 135].

[2] Idan Shenfeld, Jyothish Pari, Pulkit Agrawal. RL's Razor: Why Online Reinforcement Learning Forgets Less. [cite_start]2025[cite: 136].

[3] J. Wei, X. Wang, D. Schuurmans, et al. Chain-of-thought prompting elicits reasoning in large language models. [cite_start]In Advances in Neural Information Processing Systems, 2022[cite: 137, 138].

[4] H. Lightman, V. Kosaraju, Y. Burda, et al. [cite_start]Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023[cite: 139, 140].

[5] J. Uesato, N. Kushman, R. Kumar, et al. [cite_start]Solving math word problems with process- and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022[cite: 141, 142].

[6] J. Schulman, F. Wolski, P. Dhariwal, et al. [cite_start]Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017[cite: 143].

[7] Y. Bengio, J. Louradour, R. Collobert, J. Weston. Curriculum learning. [cite_start]In Proceedings of the 26th Annual International Conference on Machine Learning, 2009[cite: 144, 145].

[8] X. Wang, J. Wei, D. Schuurmans, et al. [cite_start]Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022[cite: 146, 147].

[9] L. Ouyang, J. Wu, X. Jiang, et al. Training language models to follow instructions with human feedback. [cite_start]In Advances in Neural Information Processing Systems, 2022[cite: 148, 149].

[10] L. Gao, J. Schulman, J. Hilton. Scaling laws for reward model overoptimization. [cite_start]In International Conference on Machine Learning, 2023[cite: 151].
