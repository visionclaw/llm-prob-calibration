# High-Diffusion Pattern Analysis

> Analysis of samples showing the most spread/diffuse distributions and highest multimodality.
> Based on selected examples from Qwen3-8B (n=9) and Gemma-3-12B (n=9).

---

## What "high diffusion" means here

- **High entropy** (P distribution, Condition A): ≥ 4.0 bits
- **High multimodality**: ≥ 6 distinct peaks

High-diffusion samples = questions where the model's logit distribution over 0–100 is genuinely spread across many competing values rather than concentrated near any single number.

---

## Which questions trigger it?

**Qwen3-8B high-diffusion (5/9):**
- Coin flip — 14 flips, P(at least 9 heads) → GT=21.2% | entropy=5.79, modes=6
- Urn without replacement — 7R/14B, draw 3, P(0 red) → GT=27.37% | entropy=5.65, modes=12
- Card — 3 draws from 52, P(same suit) → GT=5.18% | entropy=5.11, modes=6
- Epistemic misc — P(Nobel winner from US) → entropy=5.15, modes=9
- Epistemic general — P(restaurant closes in year 1) → entropy=4.56, modes=18

**Gemma-3-12B high-diffusion (4/9):**
- Bayesian PPV — 0.5% prevalence, 95% sens, 90% spec → GT=4.56% | entropy=4.86, modes=12
- Bayesian PPV — 10% prevalence, 80% sens, 99% spec → GT=89.89% | entropy=4.78, modes=18
- Coin flip — 20 flips, P(at least 10 heads) → GT=58.81% | entropy=3.36, modes=7
- Urn without replacement — 13R/3B, draw 6, P(exactly 4 red) → GT=26.79% | entropy=4.79, modes=8

---

## Structural patterns

Structural features across all high-diffusion vs. low-diffusion examples (combined across both models):

| Feature | High-diffusion (n=9) | Low-diffusion (n=9) |
|---------|----------------------|----------------------|
| Multi-step computation required | **78%** | 89% |
| Large numerical operands (N≥10) | **78%** | 67% |
| Ground truth in mid-range (10–90%) | **78%** | **22%** |
| Epistemic (Tier 3, no formula) | 22% | 22% |

The most discriminating feature by far is **ground truth in the mid-range (10–90%)**. Low-diffusion examples almost all have GT near the extremes (0–10%) — the model can lock onto "it's very small" without needing to compute precisely. Mid-range GTs force a genuine numeric choice the model can't escape with a low-probability default.

---

## Summary: What they have in common

### 1. Ground truth is in the middle range (10–90%)
The clearest separator. When the answer is near 0 or near 100, the model's logit distribution is typically peaked at a small number (its default low-probability bias). When the true answer is somewhere in the middle — 20%, 27%, 58% — the model has **no safe default** and its probability mass fragments across many plausible values.

### 2. Multiple numerical operands with non-trivial interaction
High-diffusion questions consistently involve 3–6 numerical quantities that must be combined: urn compositions, flip counts, sensitivity+specificity+prevalence triples. Low-diffusion examples like *"Two dice, sum = 2?"* (GT=2.78%) have a simple, singular arithmetic path. The more quantities need to combine, the more the model hedges across multiple partial-computation trajectories.

### 3. "Without replacement" / combinatorial branching
A recurring structural motif: urn sampling without replacement, card draws without replacement, multi-flip coin sequences. These require **combinatorial enumeration** (hypergeometric, binomial), not just a single formula application. Each sub-path corresponds to a different partial answer, and the model spreads mass across those paths rather than integrating them.

### 4. Gemma: high diffusion correlates with large order effects
For Gemma specifically, every high-diffusion example also shows strong order effects (mean KL=2.71 vs 1.68 for low-diffusion). The model is uncertain *and* easily anchored. This combination — wide distribution + high sensitivity to elicitation order — suggests Gemma's representations for these questions are genuinely underspecified: no single answer is dominant, so whichever framing comes first steers the output.

For Qwen3-8B, the opposite: high-diffusion samples have *lower* mean order effects (0.40 vs 1.79). Qwen3 is diffuse but stable — it spreads mass across many values regardless of elicitation order, suggesting the uncertainty is structural (the model genuinely has no good answer) rather than frame-sensitive.

### 5. NOT a tier effect
Notably, high diffusion is **not** confined to Tier 3 (epistemic questions). In both models, ~75–100% of high-diffusion examples come from Tier 1 or Tier 2 — questions with exact computable ground truth. The model's uncertainty here is **computational failure**, not genuine epistemic uncertainty. It knows these are probability questions with definite answers; it just can't compute them.

### 6. Low diffusion = extreme answers or trivial computations
The clearest low-diffusion pattern: questions where the GT is near 0 or near 100, or where the computation is a single lookup (P(heart from deck) = 25%, dice sum = 2/36). When the answer is structurally obvious or the true value is near the model's default low-probability output, the distribution collapses to a spike. This isn't calibration — it's coincidence: the model defaults to small numbers, and some questions happen to have small answers.

---

## Implications for CoT

The multi-step computation pattern is a strong predictor of diffusion. These are exactly the questions where chain-of-thought reasoning is most likely to help — by explicitly decomposing the computation (enumerate cases, apply Bayes, multiply binomials) into a sequence of token-level steps, the model might commit to a single computational path rather than superimposing many. CoT experiment design: [cot_experiment_design.md](cot_experiment_design.md).

---

## Caveats

This analysis is based on the diversity-selected example sets (9 per model), not the full 400–500 question dataset. The patterns are suggestive but would need to be validated at scale using the full results. In particular, the GT mid-range finding should be tested as a regression: `entropy ~ I(10 < GT < 90) + category + tier + model` across all samples.
