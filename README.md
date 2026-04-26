# LLM Probability Calibration: Logit-Level Analysis

How do LLMs behave when asked to estimate probabilities (0–100%)? This experiment goes beyond top-1 greedy decoding to examine the **full softmax distribution over numeric tokens** at the answer position.

## Research Questions

1. **Distribution shape**: Is the logit distribution over probability values peaked (low entropy), bimodal, uniform? Are there attractor values (multiples of 10, 50%)?
2. **Verbalized confidence vs. logit entropy**: When a model states "90% confident," does its logit distribution actually concentrate around 90, or is it spread across many values?
3. **Calibration**: For questions with computable ground truth, how well-calibrated are the model's probability estimates?
4. **Elicitation order effects**: Does asking for probability first (P-first) vs. confidence first (C-first) shift the logit distributions? Does the first value anchor the second?

## Experimental Design

### Conditions

**Condition A — P-first:**
```
Q: <question>? Answer with ONLY "P=[0-100] C=[0-100]" where P=your probability estimate and C=your confidence in that estimate (both integers 0-100).
A: P=
```

**Condition B — C-first:**
```
Q: <question>? Answer with ONLY "C=[0-100] P=[0-100]" where C=your confidence in your estimate and P=your probability estimate (both integers 0-100).
A: C=
```

For each condition, we extract the full probability distribution over integers 0–100 at each answer position using teacher-forcing and multi-token aggregation.

### Dataset

~500 probability questions across 3 tiers:

| Tier | Count | Description | Ground Truth |
|------|-------|-------------|--------------|
| 1 — Bayesian Inference | ~200 | Disease/test PPV via Bayes rule | Exact (computed) |
| 2 — Classical Probability | ~150 | Urns, coins, dice, cards | Exact (computed) |
| 3 — Epistemic / Real-world | ~150 | Weather, medical, Kahneman-style | Empirical estimate |

### Logit Extraction

Multi-token number handling:
1. At the first digit position, collect softmax over digit tokens (0–9)
2. For each first digit, condition and collect second-digit distribution
3. Reconstruct P(number=N) = P(first_digit) × P(second_digit | first_digit)
4. Normalize over all integers 0–100

### Analysis

- Entropy histograms (P and C distributions)
- Calibration curves (reliability diagrams) for Tier 1/2
- Confidence vs. entropy scatter plots
- Modality analysis (number of distribution modes)
- Attractor analysis (clustering at round numbers)
- Order effect quantification (KL divergence between conditions)

## Quick Start

```bash
# Setup
bash setup.sh

# Generate dataset
source .venv/bin/activate
python data/generate_dataset.py

# Run experiment (requires GPU)
python src/run_experiment.py --model Qwen/Qwen2.5-7B-Instruct --output results/

# Analyze results
python src/analyze.py --results results/ --output reports/figures/
```

## SLURM

```bash
sbatch slurm/run_experiment.sbatch
```

## Project Structure

```
├── README.md
├── requirements.txt
├── setup.sh
├── data/
│   ├── generate_dataset.py    # Dataset generation
│   └── questions.json         # Pre-generated dataset
├── src/
│   ├── logit_extractor.py     # Core logit extraction
│   ├── run_experiment.py      # Main experiment loop
│   └── analyze.py             # Analysis + plotting
├── results/                   # .gitignored
├── reports/
│   └── literature_review.md
└── slurm/
    └── run_experiment.sbatch
```

## Citation Chain

Guo 2017 → Kadavath 2022 → Mitchell 2023 → Xiong 2023 → Yang 2024 → **this work**

## License

MIT
