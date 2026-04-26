# LLM Probability Calibration — Literature Review

> Research date: 2026-04-26

## Key Papers

### Foundational
- **Guo et al. 2017 (ICML)** — *On Calibration of Modern Neural Networks* (arXiv:1706.04599). Introduces ECE, reliability diagrams, temperature scaling. Modern deep NNs are overconfident.
- **Kadavath et al. 2022 (Anthropic)** — *Language Models (Mostly) Know What They Know* (arXiv:2207.05221). P(True)/P(IK): pre-RLHF models well-calibrated on closed-form tasks.
- **Geng et al. 2024 (NAACL)** — *Survey of Confidence Estimation and Calibration in LLMs* (arXiv:2311.08298). Best survey.

### RLHF and Logit Calibration
- **Mitchell et al. 2023 (EMNLP)** — *Strategies for Eliciting Calibrated Confidence Scores from RLHF-LMs* (arXiv:2305.14975). RLHF breaks logit calibration. Verbalized confidence outperforms raw logprobs for RLHF models.
- **Ma et al. 2025** — *Estimating LLM Uncertainty with Logits (LogTokU)* (arXiv:2502.00290). Logits lose evidence strength; standard probability-based methods fail.

### Verbalized Confidence
- **Xiong et al. 2023 (ICLR)** — *Can LLMs Express Their Uncertainty?* (arXiv:2306.13063). LLMs overstate verbalized confidence. Scale helps.
- **Yang et al. 2024** — *On Verbalized Confidence Scores for LLMs* (arXiv:2412.14737). Confidence-Probability Alignment framework.
- **Clinical study 2026 (Nature npj Digital Medicine)** — 48 LLMs, universal poor calibration.

### Entropy and Uncertainty
- **Kuhn et al. 2023 (ICLR Spotlight)** — *Semantic Entropy* (arXiv:2302.09664). More predictive of accuracy than token-level entropy.
- **Li et al. 2024** — *Graph-based Confidence Calibration for LLMs* (arXiv:2411.02454).

### Benchmarks
- **ForecastBench (Karger et al. 2024, NeurIPS)** — forecastbench.org. LLMs beaten by expert forecasters.
- **BIG-Bench (Srivastava et al. 2022, TMLR)** — Includes probability reasoning tasks.

### Tokenization
- **Kreitner et al. 2025** — *BitTokens* (arXiv:2510.06824). Multi-token number problem confirmed.

## Gaps This Experiment Fills
1. Full logit distribution multimodality for numeric probability outputs
2. Calibration where the correct answer is itself a probability
3. Logit entropy vs verbalized confidence alignment for probability-valued questions
4. Elicitation order effects on logit distributions (P-first vs C-first)

## Citation Chain
Guo 2017 → Kadavath 2022 → Mitchell 2023 → Xiong 2023 → Yang 2024 → **this work**
