# Chain-of-Thought Probability Estimation Experiment

> Design document for a follow-up to the logit-level calibration study.
> Status: **Pending implementation**

## Motivation

The baseline experiment extracted logit-level distributions at the **direct answer position** — forcing the model to emit a numeric token immediately after the `=` sign. This deliberately bypasses any internal reasoning the model might do before settling on a number.

The key open question is: **does chain-of-thought reasoning improve calibration, change the shape of the distribution, or merely shift the argmax?**

Specifically:
- Does CoT reduce entropy (sharper distributions)?
- Does CoT reduce multimodality (fewer competing hypotheses)?
- Does CoT correct the low-probability bias observed in the baseline?
- Does CoT reduce order effects (less anchoring)?
- Does CoT align verbalized confidence more tightly with logit entropy?

---

## Experimental Design

### Conditions

Three conditions, each eliciting probability (P) and confidence (C):

**Condition A — Direct (baseline, already run)**
```
Q: <question>? Answer with ONLY "P=[0-100] C=[0-100]" where P=your probability estimate and C=your confidence in that estimate (both integers 0-100).
A: P=
```

**Condition C — CoT then answer (P-first)**
```
Q: <question>? Think step by step, then answer with "P=[0-100] C=[0-100]" where P=your probability estimate and C=your confidence in that estimate (both integers 0-100).
A: <model generates reasoning>
Final answer: P=
```

**Condition D — CoT then answer (C-first)**
```
Q: <question>? Think step by step, then answer with "C=[0-100] P=[0-100]" where C=your confidence in your estimate and P=your probability estimate (both integers 0-100).
A: <model generates reasoning>
Final answer: C=
```

### Logit Extraction with CoT

CoT introduces a complication: the model generates a variable-length reasoning trace before the answer. Logit extraction must use **teacher-forcing** at the correct answer token position, which requires:

1. Run the model **autoregressively** to generate the full CoT trace
2. Locate the answer format (`P=` or `C=`) in the generated sequence
3. Extract logits **at the first digit position** of the answer, conditioned on the full prefix (question + CoT trace)
4. Apply the same multi-token reconstruction as baseline: `P(N) = P(d1) × P(d2 | d1)`, normalized

This means the CoT condition requires **two forward passes per question per condition**:
- Pass 1: generate CoT trace (autoregressive)
- Pass 2: teacher-forced logit extraction at the answer position

**Alternative (cheaper)**: Extract logits only at Pass 1 answer position (i.e., treat the CoT as the prefix and take the logits immediately after the format token). This is equivalent to reading the model's "mind" right after it finishes reasoning — likely the more informative measure.

### Sentinel Format

To reliably locate the answer position after free-form CoT, use a structured sentinel:

```
Q: <question>? Think step by step. After reasoning, output your final answer on a new line as:
FINAL: P=[0-100] C=[0-100]

A: <CoT reasoning>
FINAL: P=
```

The `FINAL:` sentinel makes regex-based position finding robust.

### Prompting Variants to Explore

| Variant | Description |
|---------|-------------|
| `cot_free` | "Think step by step" — unconstrained reasoning |
| `cot_bayes` | "Apply Bayes' theorem step by step" — guided for Tier 1 |
| `cot_explicit` | Force explicit sub-computations: "State the prior, likelihood, and posterior separately" |
| `cot_budget` | "Think for at most 3 sentences, then answer" — controlled reasoning length |

Start with `cot_free` to maximize comparability with existing literature.

---

## Dataset

Use the **same 500-question dataset** as the baseline (Tier 1/2/3) for direct comparison. No new questions needed.

---

## Implementation Plan

### New source file: `src/run_experiment_cot.py`

Based on `run_experiment.py` with these modifications:

```python
# Key change: generate CoT first, then extract logits at answer position
def run_cot_condition(model, tokenizer, question, condition="C"):
    # 1. Build CoT prompt
    prompt = build_cot_prompt(question, condition)
    
    # 2. Autoregressively generate until FINAL: marker
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,  # greedy CoT for reproducibility
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=[SentinelStoppingCriteria("FINAL:", tokenizer)],
        )
    
    # 3. Decode CoT trace
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    cot_trace = generated.strip()
    
    # 4. Build full prefix up to answer position
    # e.g., original_prompt + cot_trace + "\nFINAL: P="
    answer_prefix = prompt + cot_trace + f"\nFINAL: {condition}="
    
    # 5. Teacher-forced logit extraction at first digit
    dist = extract_distribution(model, tokenizer, answer_prefix)
    
    return dist, cot_trace
```

Key implementation details:
- **SentinelStoppingCriteria**: stop generation when `FINAL:` is produced
- **CoT trace logging**: save the full reasoning trace alongside logits (for qualitative analysis)
- **Same multi-token reconstruction** as baseline for P(N)
- **Temperature=0 / greedy** for CoT generation to keep results deterministic

### New analysis: CoT vs. Direct comparison

`src/analyze_cot.py` — extend `analyze.py` to:
- Overlay entropy distributions: Direct vs. CoT (paired violin plots)
- Calibration curve comparison: Direct vs. CoT per tier
- Scatter: Δentropy (CoT-Direct) vs. ground truth value — does CoT help more for hard questions?
- Qualitative: show top-5 cases where CoT *helps* vs. *hurts* calibration
- CoT trace length correlation: longer reasoning → better calibration?

### SLURM job

```bash
# slurm/run_cot_experiment.sbatch
#SBATCH --job-name=llm-cot-calib
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00  # CoT 2-4x slower per sample

MODEL=${MODEL:-Qwen/Qwen3-8B}
OUTPUT=${OUTPUT:-results/${MODEL//\//-}_cot/}

python src/run_experiment_cot.py \
    --model $MODEL \
    --output $OUTPUT \
    --cot-variant cot_free \
    --max-cot-tokens 512 \
    --no-show
```

---

## Expected Results and Hypotheses

| Hypothesis | Direction | Rationale |
|------------|-----------|-----------|
| H1: CoT reduces entropy for Tier 1 (Bayesian) | ↓ entropy | Bayes rule is computable; CoT might correctly compute it |
| H2: CoT has less effect on Tier 3 (epistemic) | null | No ground truth to reason toward |
| H3: CoT reduces order effects | ↓ KL(A‖B) | Reasoning anchors the estimate before verbalization |
| H4: CoT doesn't fix low-probability bias | null | Bias may be in the weight space, not reasoning |
| H5: CoT increases argmax accuracy for Tier 1/2 | ↑ accuracy | Computation access improves precision |

---

## Additional Variants (Future)

- **Scratchpad suppressed**: prompt model to reason *internally* and output only the answer (tests whether the internal computation matters, not just showing CoT)
- **Incorrect CoT injection**: provide a deliberately wrong reasoning trace — does the model correct it or follow it?
- **Calibration feedback loop**: after showing model its own calibration error, does subsequent performance improve?

---

## References

- Kojima et al. (2022) — "Large Language Models are Zero-Shot Reasoners"
- Wang et al. (2022) — Self-Consistency improves CoT reasoning
- Xiong et al. (2023) — Can LLMs Express Their Uncertainty?
- Kadavath et al. (2022) — Language Models (Mostly) Know What They Know

---

*This document should be implemented after the baseline analysis is complete and published.*
