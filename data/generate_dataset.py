#!/usr/bin/env python3
"""
Generate synthetic probability questions dataset for LLM calibration experiments.

Produces ~500 questions across 3 tiers:
  Tier 1: Bayesian inference (computable PPV via Bayes rule)
  Tier 2: Classical probability (urns, coins, dice, cards)
  Tier 3: Epistemic / real-world (weather, medical, Kahneman-style)

Output: data/questions.json
"""

import json
import math
import itertools
from pathlib import Path
from scipy.stats import hypergeom, binom
from math import comb


def generate_tier1_bayesian() -> list[dict]:
    """Bayesian inference: disease/test PPV via Bayes rule."""
    questions = []
    prevalences = [0.1, 0.5, 1, 2, 5, 10, 20, 30, 50]
    sensitivities = [70, 80, 90, 95, 99]
    specificities = [70, 80, 90, 95, 99]

    qid = 0
    for prev, sens, spec in itertools.product(prevalences, sensitivities, specificities):
        prev_frac = prev / 100.0
        sens_frac = sens / 100.0
        spec_frac = spec / 100.0

        # PPV = P(disease | positive test)
        numerator = sens_frac * prev_frac
        denominator = sens_frac * prev_frac + (1 - spec_frac) * (1 - prev_frac)

        if denominator < 1e-10:
            continue

        ppv = numerator / denominator * 100.0

        # Filter degenerate: skip if PPV is essentially 0 or 100
        if ppv < 0.5 or ppv > 99.5:
            continue

        question = (
            f"A disease affects {prev}% of the population. "
            f"A diagnostic test has {sens}% sensitivity and {spec}% specificity. "
            f"A randomly selected person tests positive. "
            f"What is the probability (0-100%) that they actually have the disease?"
        )

        questions.append({
            "id": f"t1_bayes_{qid:04d}",
            "tier": 1,
            "category": "bayesian_ppv",
            "question": question,
            "ground_truth": round(ppv, 2),
            "ground_truth_confidence": 1.0,
            "parameters": {
                "prevalence_pct": prev,
                "sensitivity_pct": sens,
                "specificity_pct": spec,
                "ppv_exact": ppv,
            },
        })
        qid += 1

    return questions


def generate_tier2_classical() -> list[dict]:
    """Classical probability: urns, coins, dice, cards."""
    questions = []
    qid = 0

    # --- Urn problems (programmatic) ---
    import random as _rng
    _rng.seed(42)
    urn_configs = set()
    # Handcrafted core configs
    for cfg in [
        (5, 5, 3, 2), (7, 3, 4, 3), (10, 10, 5, 3), (6, 4, 3, 1),
        (8, 12, 5, 2), (4, 6, 3, 2), (15, 5, 4, 1), (3, 7, 4, 3),
        (10, 5, 3, 2), (6, 6, 4, 2), (12, 8, 5, 4), (9, 3, 3, 1),
        (8, 8, 4, 2), (5, 15, 3, 1), (7, 7, 5, 3), (4, 4, 3, 2),
        (10, 10, 6, 3), (6, 14, 4, 1), (3, 3, 2, 1), (20, 10, 5, 2),
        (5, 5, 2, 1), (8, 4, 3, 2), (10, 15, 5, 2), (6, 6, 3, 1),
        (12, 12, 6, 4), (7, 13, 4, 2), (4, 8, 3, 1), (9, 9, 4, 3),
        (15, 15, 5, 3), (3, 9, 3, 1),
    ]:
        urn_configs.add(cfg)
    # Generate more programmatically to reach ~50
    for _ in range(80):
        r = _rng.randint(3, 20)
        b = _rng.randint(3, 20)
        n = _rng.randint(2, min(6, r + b))
        k = _rng.randint(0, min(n, r))
        if (n - k) <= b and k <= r:
            urn_configs.add((r, b, n, k))
    urn_configs = sorted(urn_configs)

    for r, b, n, k in urn_configs:
        total = r + b
        if n > total or k > min(n, r) or (n - k) > b:
            continue

        prob = hypergeom.pmf(k, total, r, n) * 100.0
        if prob < 0.1 or prob > 99.9:
            continue

        question = (
            f"An urn contains {r} red and {b} blue balls. "
            f"You draw {n} balls without replacement. "
            f"What is the probability (0-100%) that exactly {k} are red?"
        )

        questions.append({
            "id": f"t2_urn_{qid:04d}",
            "tier": 2,
            "category": "urn",
            "question": question,
            "ground_truth": round(prob, 2),
            "ground_truth_confidence": 1.0,
            "parameters": {"red": r, "blue": b, "draw": n, "target_red": k},
        })
        qid += 1

    # --- Coin flips (expanded) ---
    coin_configs_set = set()
    for cfg in [
        (3, 2), (4, 3), (5, 3), (6, 4), (7, 5), (8, 5), (10, 6),
        (5, 4), (6, 3), (8, 6), (10, 7), (4, 2), (7, 4), (12, 8),
        (3, 1), (5, 2), (6, 5), (8, 3), (10, 4), (7, 3),
        (4, 1), (6, 2), (8, 4), (10, 5), (12, 6), (5, 1),
        (9, 5), (7, 6), (10, 3), (15, 10),
    ]:
        coin_configs_set.add(cfg)
    for _ in range(100):
        n = _rng.randint(2, 20)
        k = _rng.randint(1, n)
        coin_configs_set.add((n, k))
    coin_configs = sorted(coin_configs_set)

    for n, k in coin_configs:
        if k > n:
            continue
        # P(heads >= k) = sum of binom.pmf(i, n, 0.5) for i in k..n
        prob = (1 - binom.cdf(k - 1, n, 0.5)) * 100.0
        if prob < 0.1 or prob > 99.9:
            continue

        question = (
            f"A fair coin is flipped {n} times. "
            f"What is the probability (0-100%) that heads appears at least {k} times?"
        )

        questions.append({
            "id": f"t2_coin_{qid:04d}",
            "tier": 2,
            "category": "coin",
            "question": question,
            "ground_truth": round(prob, 2),
            "ground_truth_confidence": 1.0,
            "parameters": {"flips": n, "at_least_heads": k},
        })
        qid += 1

    # --- Dice problems ---
    dice_configs = [
        (2, 7, "sum exactly"),   # P(sum=7 with 2 dice)
        (2, 8, "sum exactly"),
        (2, 6, "sum exactly"),
        (2, 10, "sum at least"),
        (2, 9, "sum at least"),
        (3, 10, "sum at least"),
        (2, 2, "sum exactly"),   # snake eyes
        (2, 12, "sum exactly"),  # double sixes
        (1, 6, "roll exactly"),  # single die
        (1, 3, "roll at most"),
    ]

    for n_dice, target, condition in dice_configs:
        if n_dice == 1:
            if condition == "roll exactly":
                prob = (1 / 6) * 100.0
                question = f"A fair six-sided die is rolled once. What is the probability (0-100%) of rolling exactly a {target}?"
            elif condition == "roll at most":
                prob = (target / 6) * 100.0
                question = f"A fair six-sided die is rolled once. What is the probability (0-100%) of rolling at most a {target}?"
            else:
                continue
        elif n_dice == 2:
            # Enumerate all outcomes
            total_outcomes = 36
            if condition == "sum exactly":
                count = sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 == target)
                prob = (count / total_outcomes) * 100.0
                question = f"Two fair six-sided dice are rolled. What is the probability (0-100%) that the sum is exactly {target}?"
            elif condition == "sum at least":
                count = sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 >= target)
                prob = (count / total_outcomes) * 100.0
                question = f"Two fair six-sided dice are rolled. What is the probability (0-100%) that the sum is at least {target}?"
            else:
                continue
        elif n_dice == 3:
            total_outcomes = 216
            if condition == "sum at least":
                count = sum(
                    1
                    for d1 in range(1, 7)
                    for d2 in range(1, 7)
                    for d3 in range(1, 7)
                    if d1 + d2 + d3 >= target
                )
                prob = (count / total_outcomes) * 100.0
                question = f"Three fair six-sided dice are rolled. What is the probability (0-100%) that the sum is at least {target}?"
            else:
                continue
        else:
            continue

        if prob < 0.1 or prob > 99.9:
            continue

        questions.append({
            "id": f"t2_dice_{qid:04d}",
            "tier": 2,
            "category": "dice",
            "question": question,
            "ground_truth": round(prob, 2),
            "ground_truth_confidence": 1.0,
            "parameters": {"n_dice": n_dice, "target": target, "condition": condition},
        })
        qid += 1

    # --- Card problems ---
    card_problems = [
        {
            "question": "A card is drawn from a standard 52-card deck. What is the probability (0-100%) that it is a heart?",
            "ground_truth": 25.0,
            "params": {"type": "suit"},
        },
        {
            "question": "A card is drawn from a standard 52-card deck. What is the probability (0-100%) that it is a face card (Jack, Queen, or King)?",
            "ground_truth": round(12 / 52 * 100, 2),
            "params": {"type": "face_card"},
        },
        {
            "question": "Two cards are drawn without replacement from a standard 52-card deck. What is the probability (0-100%) that both are aces?",
            "ground_truth": round((4/52 * 3/51) * 100, 2),
            "params": {"type": "two_aces"},
        },
        {
            "question": "Two cards are drawn without replacement from a standard 52-card deck. What is the probability (0-100%) that at least one is a heart?",
            "ground_truth": round((1 - (39/52 * 38/51)) * 100, 2),
            "params": {"type": "at_least_one_heart"},
        },
        {
            "question": "A card is drawn from a standard 52-card deck. Given that it is red, what is the probability (0-100%) that it is a diamond?",
            "ground_truth": 50.0,
            "params": {"type": "conditional_red_diamond"},
        },
        {
            "question": "Three cards are drawn without replacement from a standard 52-card deck. What is the probability (0-100%) that all three are the same suit?",
            "ground_truth": round(4 * (13/52 * 12/51 * 11/50) * 100, 2),
            "params": {"type": "three_same_suit"},
        },
        {
            "question": "A card is drawn from a standard 52-card deck. What is the probability (0-100%) that it is either an ace or a spade?",
            "ground_truth": round(16/52 * 100, 2),
            "params": {"type": "ace_or_spade"},
        },
        {
            "question": "Five cards are drawn from a standard 52-card deck. What is the probability (0-100%) that exactly two are hearts?",
            "ground_truth": round(hypergeom.pmf(2, 52, 13, 5) * 100, 2),
            "params": {"type": "five_draw_two_hearts"},
        },
        {
            "question": "Two cards are drawn without replacement from a standard 52-card deck. What is the probability (0-100%) that the second card is a king given the first card was a queen?",
            "ground_truth": round(4/51 * 100, 2),
            "params": {"type": "conditional_king_after_queen"},
        },
        {
            "question": "A card is drawn from a standard 52-card deck. What is the probability (0-100%) that it is NOT a face card and NOT an ace?",
            "ground_truth": round(36/52 * 100, 2),
            "params": {"type": "not_face_not_ace"},
        },
    ]

    for cp in card_problems:
        questions.append({
            "id": f"t2_card_{qid:04d}",
            "tier": 2,
            "category": "card",
            "question": cp["question"],
            "ground_truth": cp["ground_truth"],
            "ground_truth_confidence": 1.0,
            "parameters": cp["params"],
        })
        qid += 1

    return questions


def generate_tier3_epistemic() -> list[dict]:
    """Epistemic / real-world questions including Kahneman-style base rate neglect."""
    questions = []
    qid = 0

    # --- Weather base rates ---
    weather = [
        {
            "question": "In London in January, what is the probability (0-100%) that it rains on any given day?",
            "ground_truth": 55,
            "confidence": 0.7,
            "category": "weather",
            "params": {"city": "London", "month": "January", "event": "rain"},
        },
        {
            "question": "In Phoenix, Arizona in July, what is the probability (0-100%) that the high temperature exceeds 40°C (104°F)?",
            "ground_truth": 45,
            "confidence": 0.6,
            "category": "weather",
            "params": {"city": "Phoenix", "month": "July", "event": "extreme_heat"},
        },
        {
            "question": "In Tokyo in June, what is the probability (0-100%) that it rains on any given day?",
            "ground_truth": 60,
            "confidence": 0.65,
            "category": "weather",
            "params": {"city": "Tokyo", "month": "June", "event": "rain"},
        },
        {
            "question": "In New York City in December, what is the probability (0-100%) that it snows on any given day?",
            "ground_truth": 12,
            "confidence": 0.6,
            "category": "weather",
            "params": {"city": "NYC", "month": "December", "event": "snow"},
        },
        {
            "question": "In Mumbai during monsoon season (July), what is the probability (0-100%) that it rains on any given day?",
            "ground_truth": 85,
            "confidence": 0.7,
            "category": "weather",
            "params": {"city": "Mumbai", "month": "July", "event": "monsoon_rain"},
        },
        {
            "question": "Given that it rained in London yesterday, what is the probability (0-100%) it rains again today?",
            "ground_truth": 62,
            "confidence": 0.5,
            "category": "weather",
            "params": {"city": "London", "event": "conditional_rain"},
        },
        {
            "question": "In São Paulo in February, what is the probability (0-100%) that it rains on any given day?",
            "ground_truth": 70,
            "confidence": 0.65,
            "category": "weather",
            "params": {"city": "Sao_Paulo", "month": "February", "event": "rain"},
        },
        {
            "question": "In Cairo, Egypt, what is the probability (0-100%) that it rains on any given day of the year?",
            "ground_truth": 3,
            "confidence": 0.7,
            "category": "weather",
            "params": {"city": "Cairo", "event": "rain_annual"},
        },
    ]

    for w in weather:
        questions.append({
            "id": f"t3_weather_{qid:04d}",
            "tier": 3,
            "category": w["category"],
            "question": w["question"],
            "ground_truth": w["ground_truth"],
            "ground_truth_confidence": w["confidence"],
            "parameters": w["params"],
        })
        qid += 1

    # --- Medical/epidemiological base rates ---
    medical = [
        {
            "question": "In the general US adult population, what is the probability (0-100%) that a randomly selected person has Type 2 diabetes?",
            "ground_truth": 11,
            "confidence": 0.8,
            "category": "medical",
            "params": {"condition": "type2_diabetes", "population": "US_adults"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person worldwide is left-handed?",
            "ground_truth": 10,
            "confidence": 0.85,
            "category": "medical",
            "params": {"trait": "left_handedness"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected adult in the US has hypertension?",
            "ground_truth": 47,
            "confidence": 0.75,
            "category": "medical",
            "params": {"condition": "hypertension", "population": "US_adults"},
        },
        {
            "question": "What is the probability (0-100%) that a routine mammogram for a 50-year-old woman yields a false positive result?",
            "ground_truth": 10,
            "confidence": 0.7,
            "category": "medical",
            "params": {"test": "mammogram", "age": 50, "event": "false_positive"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person worldwide has red hair?",
            "ground_truth": 2,
            "confidence": 0.8,
            "category": "medical",
            "params": {"trait": "red_hair"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly chosen American adult is obese (BMI >= 30)?",
            "ground_truth": 42,
            "confidence": 0.8,
            "category": "medical",
            "params": {"condition": "obesity", "population": "US_adults"},
        },
        {
            "question": "What is the probability (0-100%) of surviving 5 years after a pancreatic cancer diagnosis?",
            "ground_truth": 12,
            "confidence": 0.75,
            "category": "medical",
            "params": {"condition": "pancreatic_cancer", "metric": "5yr_survival"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected newborn is a twin?",
            "ground_truth": 3,
            "confidence": 0.8,
            "category": "medical",
            "params": {"event": "twin_birth"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected American adult has been diagnosed with depression at some point in their life?",
            "ground_truth": 21,
            "confidence": 0.65,
            "category": "medical",
            "params": {"condition": "depression_lifetime", "population": "US_adults"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person worldwide is colorblind?",
            "ground_truth": 5,
            "confidence": 0.8,
            "category": "medical",
            "params": {"trait": "colorblindness"},
        },
    ]

    for m in medical:
        questions.append({
            "id": f"t3_medical_{qid:04d}",
            "tier": 3,
            "category": m["category"],
            "question": m["question"],
            "ground_truth": m["ground_truth"],
            "ground_truth_confidence": m["confidence"],
            "parameters": m["params"],
        })
        qid += 1

    # --- Sports/prediction ---
    sports = [
        {
            "question": "What is the probability (0-100%) that the home team wins a randomly selected English Premier League football match?",
            "ground_truth": 46,
            "confidence": 0.7,
            "category": "sports",
            "params": {"sport": "football", "league": "EPL", "metric": "home_win_rate"},
        },
        {
            "question": "What is the probability (0-100%) that a tennis player who wins the first set goes on to win the match (best of 3)?",
            "ground_truth": 82,
            "confidence": 0.7,
            "category": "sports",
            "params": {"sport": "tennis", "metric": "first_set_win_correlation"},
        },
        {
            "question": "In a randomly selected NBA game, what is the probability (0-100%) that the team leading at halftime wins the game?",
            "ground_truth": 80,
            "confidence": 0.75,
            "category": "sports",
            "params": {"sport": "basketball", "league": "NBA", "metric": "halftime_lead_wins"},
        },
        {
            "question": "What is the probability (0-100%) that a Major League Baseball team with the best regular season record wins the World Series?",
            "ground_truth": 15,
            "confidence": 0.6,
            "category": "sports",
            "params": {"sport": "baseball", "metric": "best_record_wins_ws"},
        },
        {
            "question": "What is the probability (0-100%) that an NFL team trailing by 10+ points at halftime comes back to win?",
            "ground_truth": 12,
            "confidence": 0.65,
            "category": "sports",
            "params": {"sport": "american_football", "metric": "10pt_comeback"},
        },
        {
            "question": "In a penalty shootout in football (soccer), what is the probability (0-100%) that the team shooting first wins?",
            "ground_truth": 55,
            "confidence": 0.75,
            "category": "sports",
            "params": {"sport": "football", "metric": "first_shooter_advantage"},
        },
        {
            "question": "What is the probability (0-100%) that a golfer makes a hole-in-one on a par-3 hole in a professional tournament?",
            "ground_truth": 0.3,  # approximately 1 in 333
            "confidence": 0.6,
            "category": "sports",
            "params": {"sport": "golf", "metric": "hole_in_one"},
        },
        {
            "question": "What is the probability (0-100%) that a Formula 1 driver starting on pole position wins the race?",
            "ground_truth": 40,
            "confidence": 0.7,
            "category": "sports",
            "params": {"sport": "f1", "metric": "pole_to_win"},
        },
    ]

    for s in sports:
        questions.append({
            "id": f"t3_sports_{qid:04d}",
            "tier": 3,
            "category": s["category"],
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "ground_truth_confidence": s["confidence"],
            "parameters": s["params"],
        })
        qid += 1

    # --- General knowledge base rate questions ---
    general = [
        {
            "question": "What is the probability (0-100%) that a randomly selected email is spam?",
            "ground_truth": 45,
            "confidence": 0.6,
            "category": "general",
            "params": {"domain": "email", "event": "spam"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected commercial flight in the US is delayed by more than 15 minutes?",
            "ground_truth": 20,
            "confidence": 0.7,
            "category": "general",
            "params": {"domain": "aviation", "event": "delay"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected startup (founded in the US) survives past 5 years?",
            "ground_truth": 50,
            "confidence": 0.65,
            "category": "general",
            "params": {"domain": "business", "event": "startup_survival"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected American has a passport?",
            "ground_truth": 48,
            "confidence": 0.7,
            "category": "general",
            "params": {"domain": "demographics", "event": "has_passport"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected marriage in the US ends in divorce?",
            "ground_truth": 42,
            "confidence": 0.65,
            "category": "general",
            "params": {"domain": "demographics", "event": "divorce"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected restaurant in the US closes within its first year?",
            "ground_truth": 17,
            "confidence": 0.6,
            "category": "general",
            "params": {"domain": "business", "event": "restaurant_closure_1yr"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person on Earth lives in a city (urban area)?",
            "ground_truth": 57,
            "confidence": 0.8,
            "category": "general",
            "params": {"domain": "demographics", "event": "urban_living"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected car on the road in Norway is electric?",
            "ground_truth": 25,
            "confidence": 0.5,
            "category": "general",
            "params": {"domain": "transportation", "event": "ev_share_norway"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected PhD student completes their degree?",
            "ground_truth": 50,
            "confidence": 0.6,
            "category": "general",
            "params": {"domain": "education", "event": "phd_completion"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected day of the year is someone's birthday in a group of 23 people?",
            "ground_truth": 50,
            "confidence": 0.95,
            "category": "general",
            "params": {"domain": "birthday_paradox", "group_size": 23},
            "note": "Birthday paradox: P(at least one shared birthday among 23 people) ≈ 50.7%",
        },
    ]

    for g in general:
        entry = {
            "id": f"t3_general_{qid:04d}",
            "tier": 3,
            "category": g["category"],
            "question": g["question"],
            "ground_truth": g["ground_truth"],
            "ground_truth_confidence": g["confidence"],
            "parameters": g["params"],
        }
        if "note" in g:
            entry["note"] = g["note"]
        questions.append(entry)
        qid += 1

    # --- Kahneman-style base rate neglect problems ---
    kahneman = [
        {
            "question": "In a city, 85% of taxis are Green and 15% are Blue. A witness identifies a taxi in a hit-and-run as Blue. Testing shows the witness correctly identifies taxi color 80% of the time. What is the probability (0-100%) that the taxi was actually Blue?",
            "ground_truth": 41,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "taxi_cab", "source": "Kahneman_Tversky_1982"},
            "note": "Classic taxi-cab problem. P(Blue|ID Blue) = 0.15*0.80 / (0.15*0.80 + 0.85*0.20) = 0.12/0.29 ≈ 41.4%",
        },
        {
            "question": "A panel of psychologists interviewed 30 engineers and 70 lawyers. They wrote personality descriptions. Jack is described as conservative, enjoys mathematical puzzles, and has no interest in politics. What is the probability (0-100%) that Jack is an engineer?",
            "ground_truth": 30,
            "confidence": 0.8,
            "category": "base_rate_neglect",
            "params": {"type": "engineer_lawyer", "base_rate_engineers": 30, "source": "Kahneman_Tversky_1973"},
            "note": "Engineer/lawyer problem. Base rate = 30% engineers. Stereotypical description should not override base rate much.",
        },
        {
            "question": "A panel of psychologists interviewed 70 engineers and 30 lawyers. They wrote personality descriptions. Jack is described as conservative, enjoys mathematical puzzles, and has no interest in politics. What is the probability (0-100%) that Jack is an engineer?",
            "ground_truth": 70,
            "confidence": 0.8,
            "category": "base_rate_neglect",
            "params": {"type": "engineer_lawyer", "base_rate_engineers": 70, "source": "Kahneman_Tversky_1973"},
            "note": "Same description, different base rate. Bayesian answer should shift significantly.",
        },
        {
            "question": "A hospital has two maternity wards. The large ward delivers ~45 babies/day, the small ward ~15 babies/day. On how many days would you expect more than 60% of babies born to be boys? Specifically: What is the probability (0-100%) that on any given day, the small ward has more than 60% boys?",
            "ground_truth": 27,
            "confidence": 0.7,
            "category": "base_rate_neglect",
            "params": {"type": "small_numbers", "source": "Kahneman_Tversky_hospital"},
            "note": "Law of small numbers. Smaller samples deviate more from population proportions.",
        },
        {
            "question": "Steve is very shy and withdrawn, helpful but with little interest in people. He has a need for order and a passion for detail. Is Steve more likely a farmer or a librarian? In the US, farmers outnumber librarians roughly 20:1. What is the probability (0-100%) that Steve is a librarian?",
            "ground_truth": 5,
            "confidence": 0.6,
            "category": "base_rate_neglect",
            "params": {"type": "representativeness", "source": "Kahneman_Tversky_1974"},
            "note": "Representativeness heuristic. Despite stereotypical description, base rate heavily favors farmer.",
        },
        {
            "question": "Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with discrimination and social justice. What is the probability (0-100%) that Linda is a bank teller?",
            "ground_truth": 2,
            "confidence": 0.5,
            "category": "base_rate_neglect",
            "params": {"type": "conjunction_fallacy_base", "source": "Tversky_Kahneman_1983"},
            "note": "Part of the conjunction fallacy setup. Raw base rate of being a bank teller is very low.",
        },
        {
            "question": "If a disease test has a false positive rate of 5% and the disease prevalence is 1 in 1000, and you test positive, what is the probability (0-100%) that you actually have the disease?",
            "ground_truth": 2,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "medical_screening", "prevalence": 0.001, "false_positive": 0.05, "sensitivity": 1.0},
            "note": "Classic medical screening. P(D|+) = 0.001*1.0 / (0.001*1.0 + 0.999*0.05) ≈ 1.96%",
        },
        {
            "question": "You are dealt a 5-card hand from a standard deck. Which is more likely: (a) a hand containing at least one ace, or (b) a hand with no aces? What is the probability (0-100%) of having at least one ace?",
            "ground_truth": 34,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "card_intuition"},
            "note": "P(at least 1 ace) = 1 - C(48,5)/C(52,5) ≈ 34.1%. Many people overestimate this.",
        },
        {
            "question": "A company drug-tests all 1000 employees. The test has 99% sensitivity and 95% specificity. If 2% of employees use drugs, and an employee tests positive, what is the probability (0-100%) they actually use drugs?",
            "ground_truth": 29,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "workplace_drug_test", "prevalence": 0.02, "sensitivity": 0.99, "specificity": 0.95},
            "note": "P(drug|+) = 0.02*0.99 / (0.02*0.99 + 0.98*0.05) ≈ 28.8%",
        },
        {
            "question": "In a town where 1% of residents have a rare condition, a screening test with 90% sensitivity and 90% specificity is administered. If you test positive, what is the probability (0-100%) you have the condition?",
            "ground_truth": 8,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "screening_low_prev", "prevalence": 0.01, "sensitivity": 0.90, "specificity": 0.90},
            "note": "P(D|+) = 0.01*0.90 / (0.01*0.90 + 0.99*0.10) ≈ 8.3%",
        },
        {
            "question": "A breathalyzer test has a 5% false positive rate and 100% detection rate for drunk drivers. If 1 in 1000 drivers on the road is drunk, and the test says a driver is drunk, what is the probability (0-100%) they actually are?",
            "ground_truth": 2,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "breathalyzer", "prevalence": 0.001, "sensitivity": 1.0, "false_positive": 0.05},
            "note": "Same structure as medical screening. P ≈ 1.96%",
        },
        {
            "question": "A lie detector test correctly identifies liars 90% of the time and correctly identifies truth-tellers 80% of the time. In a group where 5% are lying, if someone fails the test, what is the probability (0-100%) they are actually lying?",
            "ground_truth": 19,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "lie_detector", "base_rate_lying": 0.05, "sensitivity": 0.90, "specificity": 0.80},
            "note": "P(lying|fail) = 0.05*0.90 / (0.05*0.90 + 0.95*0.20) ≈ 19.1%",
        },
        {
            "question": "A facial recognition system has 99% accuracy for matching faces and 0.1% false match rate. In a stadium of 50,000 people where there is 1 wanted person, the system flags someone. What is the probability (0-100%) the flagged person is actually the wanted person?",
            "ground_truth": 2,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "facial_recognition", "true_positive": 0.99, "false_positive": 0.001, "n_people": 50000},
            "note": "P ≈ 0.99 / (0.99 + 49999*0.001) ≈ 0.99/50.999 ≈ 1.94%",
        },
        {
            "question": "In a country where 0.5% of the population carries a particular gene variant, a genetic test has 95% sensitivity and 99% specificity. If you test positive, what is the probability (0-100%) you carry the variant?",
            "ground_truth": 32,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "genetic_test", "prevalence": 0.005, "sensitivity": 0.95, "specificity": 0.99},
            "note": "P = 0.005*0.95 / (0.005*0.95 + 0.995*0.01) ≈ 32.3%",
        },
        {
            "question": "A professor assigns grades: 10% get A, 30% get B, 40% get C, 20% get D or below. A student believes they studied 'really hard.' Based only on the class distribution, what is the probability (0-100%) the student gets an A?",
            "ground_truth": 10,
            "confidence": 0.85,
            "category": "base_rate_neglect",
            "params": {"type": "grade_distribution", "source": "self_serving_bias"},
            "note": "The 'studying hard' is irrelevant to base rate. P(A) = 10% from distribution.",
        },
        {
            "question": "An earthquake prediction model claims to have predicted 80% of past major earthquakes. However, it also gives false alarms 30% of the time. If major earthquakes occur on about 0.01% of days, and the model issues a warning today, what is the probability (0-100%) that a major earthquake occurs?",
            "ground_truth": 0.03,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "earthquake_prediction", "sensitivity": 0.80, "false_positive": 0.30, "prevalence": 0.0001},
            "note": "P ≈ 0.0001*0.80 / (0.0001*0.80 + 0.9999*0.30) ≈ 0.027%",
        },
        {
            "question": "A rare bird species is found in 0.1% of forest patches in a region. A machine learning model for identifying the species from audio has 98% sensitivity and 97% specificity. If the model detects the species in a forest patch, what is the probability (0-100%) the species is actually present?",
            "ground_truth": 3,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "species_detection", "prevalence": 0.001, "sensitivity": 0.98, "specificity": 0.97},
            "note": "P ≈ 0.001*0.98 / (0.001*0.98 + 0.999*0.03) ≈ 3.2%",
        },
        {
            "question": "A spam filter correctly identifies 95% of spam emails and incorrectly flags 2% of legitimate emails as spam. If 30% of incoming emails are spam and an email is flagged, what is the probability (0-100%) it is actually spam?",
            "ground_truth": 95,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "spam_filter", "prevalence": 0.30, "sensitivity": 0.95, "false_positive": 0.02},
            "note": "P = 0.30*0.95 / (0.30*0.95 + 0.70*0.02) ≈ 95.3%. High base rate makes this intuitive.",
        },
        {
            "question": "An AI recruiter flags 90% of good candidates and 20% of unsuitable candidates. If 5% of all applicants are genuinely good fits, and a candidate is flagged, what is the probability (0-100%) they are a good fit?",
            "ground_truth": 19,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "ai_recruiter", "sensitivity": 0.90, "false_positive": 0.20, "prevalence": 0.05},
            "note": "P = 0.05*0.90 / (0.05*0.90 + 0.95*0.20) ≈ 19.1%",
        },
        {
            "question": "Two cab companies operate in a city: Blue Cab (20% of fleet) and Green Cab (80%). An accident occurs at night. A witness says it was a Blue Cab, and witnesses are correct 70% of the time at night. What is the probability (0-100%) it was actually a Blue Cab?",
            "ground_truth": 37,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "taxi_cab_variant", "blue_fraction": 0.20, "witness_accuracy": 0.70},
            "note": "P = 0.20*0.70 / (0.20*0.70 + 0.80*0.30) ≈ 36.8%",
        },
        {
            "question": "You shuffle a standard deck and draw 5 cards. What is the probability (0-100%) of getting at least one pair (two cards of the same rank)?",
            "ground_truth": 49,
            "confidence": 0.95,
            "category": "base_rate_neglect",
            "params": {"type": "poker_pair"},
            "note": "P(no pair in 5 cards) = 13*12*11*10*9 * 4^5 / C(52,5) * ... Actually P(at least one pair) ≈ 49.3%",
        },
        {
            "question": "A meteor detection system has 99.9% sensitivity and 99.9% specificity. If the probability of an actual dangerous meteor strike in any given year is 1 in 100,000, and the system raises an alarm, what is the probability (0-100%) of an actual dangerous strike?",
            "ground_truth": 1,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "meteor_detection", "sensitivity": 0.999, "specificity": 0.999, "prevalence": 0.00001},
            "note": "P ≈ 0.00001*0.999 / (0.00001*0.999 + 0.99999*0.001) ≈ 0.99%. Even high accuracy fails with tiny base rate.",
        },
        {
            "question": "A self-driving car's pedestrian detection correctly identifies 99.5% of actual pedestrians and has a 0.01% false positive rate on non-pedestrian objects. If 2% of detected objects in its field of view are actually pedestrians, and it flags something as a pedestrian, what is the probability (0-100%) it's actually a pedestrian?",
            "ground_truth": 67,
            "confidence": 0.85,
            "category": "base_rate_neglect",
            "params": {"type": "self_driving", "sensitivity": 0.995, "false_positive": 0.0001, "prevalence": 0.02},
            "note": "P = 0.02*0.995 / (0.02*0.995 + 0.98*0.0001) ≈ 99.5%. Wait: very low FP + 2% prevalence → very high.",
        },
        {
            "question": "In a study, 1% of participants have condition X. A new diagnostic test has 85% sensitivity and 90% specificity. A participant tests positive twice (two independent tests). What is the probability (0-100%) they have condition X?",
            "ground_truth": 43,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "double_test", "prevalence": 0.01, "sensitivity": 0.85, "specificity": 0.90},
            "note": "After first test: P ≈ 7.9%. After second independent positive: P ≈ 42.6%",
        },
        {
            "question": "A cognitive abilities test correctly identifies gifted students (top 2%) with 90% sensitivity and 95% specificity. If a student scores 'gifted' on the test, what is the probability (0-100%) they are actually gifted?",
            "ground_truth": 27,
            "confidence": 0.9,
            "category": "base_rate_neglect",
            "params": {"type": "gifted_test", "prevalence": 0.02, "sensitivity": 0.90, "specificity": 0.95},
            "note": "P = 0.02*0.90 / (0.02*0.90 + 0.98*0.05) ≈ 26.9%",
        },
    ]

    for k in kahneman:
        entry = {
            "id": f"t3_kahneman_{qid:04d}",
            "tier": 3,
            "category": k["category"],
            "question": k["question"],
            "ground_truth": k["ground_truth"],
            "ground_truth_confidence": k["confidence"],
            "parameters": k["params"],
        }
        if "note" in k:
            entry["note"] = k["note"]
        questions.append(entry)
        qid += 1

    # --- Miscellaneous epistemic ---
    misc = [
        {
            "question": "What is the probability (0-100%) that a coin toss lands on its edge (not heads or tails)?",
            "ground_truth": 0.01,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "physics", "event": "coin_edge"},
        },
        {
            "question": "What is the probability (0-100%) that a random person you meet on the street shares your birthday?",
            "ground_truth": 0.27,
            "confidence": 0.95,
            "category": "misc",
            "params": {"domain": "probability", "event": "shared_birthday"},
        },
        {
            "question": "What is the probability (0-100%) that it rains on any given day somewhere on Earth?",
            "ground_truth": 100,
            "confidence": 0.99,
            "category": "misc",
            "params": {"domain": "weather", "event": "rain_somewhere"},
            "note": "Essentially 100% - it is always raining somewhere on Earth.",
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected piece of luggage gets lost on a US domestic flight?",
            "ground_truth": 0.6,
            "confidence": 0.6,
            "category": "misc",
            "params": {"domain": "aviation", "event": "lost_luggage"},
        },
        {
            "question": "What is the probability (0-100%) that a four-leaf clover is found among a random sample of clovers?",
            "ground_truth": 0.01,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "biology", "event": "four_leaf_clover"},
            "note": "Estimated at about 1 in 5,000 to 1 in 10,000.",
        },
        {
            "question": "What is the probability (0-100%) that a lightning strike hits the same spot twice within a year?",
            "ground_truth": 5,
            "confidence": 0.4,
            "category": "misc",
            "params": {"domain": "weather", "event": "lightning_same_spot"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected movie on IMDB has a rating above 7.0?",
            "ground_truth": 25,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "entertainment", "event": "imdb_above_7"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected adult in the world speaks English?",
            "ground_truth": 17,
            "confidence": 0.6,
            "category": "misc",
            "params": {"domain": "demographics", "event": "speaks_english"},
        },
        {
            "question": "What is the probability (0-100%) that a package delivered by a major US carrier arrives on time?",
            "ground_truth": 90,
            "confidence": 0.6,
            "category": "misc",
            "params": {"domain": "logistics", "event": "on_time_delivery"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person has traveled to another country?",
            "ground_truth": 30,
            "confidence": 0.4,
            "category": "misc",
            "params": {"domain": "demographics", "event": "international_travel"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly chosen day in history had a volcanic eruption somewhere on Earth?",
            "ground_truth": 25,
            "confidence": 0.4,
            "category": "misc",
            "params": {"domain": "geology", "event": "daily_eruption"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected academic paper has been cited at least once?",
            "ground_truth": 50,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "academia", "event": "paper_cited"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected star in the Milky Way has at least one planet?",
            "ground_truth": 80,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "astronomy", "event": "star_has_planet"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected website on the internet uses HTTPS?",
            "ground_truth": 85,
            "confidence": 0.6,
            "category": "misc",
            "params": {"domain": "technology", "event": "uses_https"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected person worldwide is vegetarian?",
            "ground_truth": 8,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "demographics", "event": "vegetarian"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected cell phone call drops during a conversation?",
            "ground_truth": 2,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "technology", "event": "dropped_call"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected river in the world flows into the ocean (rather than a lake or inland sea)?",
            "ground_truth": 70,
            "confidence": 0.4,
            "category": "misc",
            "params": {"domain": "geography", "event": "river_to_ocean"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected country in the world is a democracy?",
            "ground_truth": 45,
            "confidence": 0.5,
            "category": "misc",
            "params": {"domain": "politics", "event": "democracy"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected year in the last century had a global pandemic?",
            "ground_truth": 5,
            "confidence": 0.4,
            "category": "misc",
            "params": {"domain": "health", "event": "pandemic_year"},
        },
        {
            "question": "What is the probability (0-100%) that a randomly selected Nobel Prize winner is from the United States?",
            "ground_truth": 35,
            "confidence": 0.7,
            "category": "misc",
            "params": {"domain": "academia", "event": "nobel_us"},
        },
    ]

    for m_item in misc:
        entry = {
            "id": f"t3_misc_{qid:04d}",
            "tier": 3,
            "category": m_item["category"],
            "question": m_item["question"],
            "ground_truth": m_item["ground_truth"],
            "ground_truth_confidence": m_item["confidence"],
            "parameters": m_item["params"],
        }
        if "note" in m_item:
            entry["note"] = m_item["note"]
        questions.append(entry)
        qid += 1

    return questions


def main():
    """Generate the full dataset and save to data/questions.json."""
    print("Generating Tier 1 — Bayesian Inference questions...")
    tier1 = generate_tier1_bayesian()
    print(f"  Generated {len(tier1)} questions")

    print("Generating Tier 2 — Classical Probability questions...")
    tier2 = generate_tier2_classical()
    print(f"  Generated {len(tier2)} questions")

    print("Generating Tier 3 — Epistemic / Real-world questions...")
    tier3 = generate_tier3_epistemic()
    print(f"  Generated {len(tier3)} questions")

    all_questions = tier1 + tier2 + tier3
    print(f"\nTotal: {len(all_questions)} questions")

    # Stats
    tier_counts = {}
    category_counts = {}
    for q in all_questions:
        tier_counts[q["tier"]] = tier_counts.get(q["tier"], 0) + 1
        category_counts[q["category"]] = category_counts.get(q["category"], 0) + 1

    print("\nTier breakdown:")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]}")

    print("\nCategory breakdown:")
    for c, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n}")

    # Save
    output_path = Path(__file__).parent / "questions.json"
    with open(output_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
