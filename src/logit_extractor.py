#!/usr/bin/env python3
"""
Core logit extraction for LLM probability calibration.

Extracts the full probability distribution over integers 0-100 from a causal LM's
logits using teacher-forcing with multi-token number aggregation.

Key insight: numbers like "45" may be tokenized as ["4","5"] or as a single token "45".
We handle this by:
1. Getting first-position logits over all digit tokens
2. For each first digit, conditionally getting second-digit logits
3. Aggregating into P(number=N) for N in 0..100
4. Normalizing
"""

import numpy as np
import torch
from typing import Optional


class LogitExtractor:
    """Extracts probability distributions over integers from LLM logits."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Pre-compute token IDs for digits 0-9
        self._build_digit_token_map()

    def _build_digit_token_map(self):
        """Map digit characters to their token IDs.

        Handles the fact that tokenizers may encode digits differently:
        - Some have single-char tokens: "0", "1", ..., "9"
        - Some may have multi-char number tokens: "10", "42", etc.

        We build mappings for both single-digit tokens and any direct
        integer tokens (0-100) that exist in the vocabulary.
        """
        self.digit_token_ids = {}  # char -> token_id for "0" through "9"
        self.direct_number_tokens = {}  # int -> token_id for numbers with single tokens

        vocab = self.tokenizer.get_vocab()

        # Find single-digit token IDs
        for d in range(10):
            d_str = str(d)
            # Try encoding just the digit
            tokens = self.tokenizer.encode(d_str, add_special_tokens=False)
            if len(tokens) == 1:
                self.digit_token_ids[d] = tokens[0]
            else:
                # Try finding it in vocab directly
                for key, tid in vocab.items():
                    # Strip any special chars (like Ġ in GPT-2 style)
                    clean = key.replace("Ġ", "").replace("▁", "").strip()
                    if clean == d_str:
                        self.digit_token_ids[d] = tid
                        break

        # Find direct number tokens (numbers that tokenize as a single token)
        for n in range(101):
            n_str = str(n)
            tokens = self.tokenizer.encode(n_str, add_special_tokens=False)
            if len(tokens) == 1:
                self.direct_number_tokens[n] = tokens[0]

        # Validate we have all digits
        missing = [d for d in range(10) if d not in self.digit_token_ids]
        if missing:
            raise ValueError(
                f"Could not find token IDs for digits: {missing}. "
                f"This tokenizer may need special handling."
            )

        print(f"[LogitExtractor] Digit tokens: {self.digit_token_ids}")
        print(f"[LogitExtractor] Direct number tokens found: {len(self.direct_number_tokens)}/101")

    @torch.no_grad()
    def _get_logits_at_position(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits at the last position of input_ids.

        Args:
            input_ids: Shape (1, seq_len)

        Returns:
            Logits tensor of shape (vocab_size,)
        """
        outputs = self.model(input_ids)
        # Logits at the last position predict the next token
        return outputs.logits[0, -1, :]

    def extract_number_distribution(
        self,
        prefix: str,
        max_value: int = 100,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Extract probability distribution over integers 0..max_value.

        Given a text prefix ending just before where a number should appear,
        computes P(number=N) for N in 0..max_value by:
        1. Single-token path: if N has a direct single token, use its probability
        2. Multi-token path: P(N) = P(first_digit) * P(second_digit | prefix + first_digit)
        3. For 3-digit (100): P("1") * P("0"|"1") * P("0"|"10")

        Args:
            prefix: Text ending just before the number position
            max_value: Maximum integer (inclusive), default 100
            temperature: Softmax temperature for logit scaling

        Returns:
            np.ndarray of shape (max_value+1,) summing to 1.0
        """
        probs = np.zeros(max_value + 1)

        # Tokenize prefix
        prefix_ids = self.tokenizer.encode(prefix, return_tensors="pt").to(self.device)

        # Get logits at first position (predicting first digit/token)
        first_logits = self._get_logits_at_position(prefix_ids)
        if temperature != 1.0:
            first_logits = first_logits / temperature
        first_probs = torch.softmax(first_logits, dim=-1)

        # === Strategy 1: Direct single-token numbers ===
        # For any number that tokenizes as a single token, grab its probability directly
        for n, tid in self.direct_number_tokens.items():
            if n <= max_value:
                probs[n] += first_probs[tid].item()

        # === Strategy 2: Multi-token digit-by-digit ===
        # For numbers that require multiple tokens, compute conditionally

        # Single-digit numbers (0-9): just the digit token probability
        # But skip if already covered by direct token
        for d in range(min(10, max_value + 1)):
            if d not in self.direct_number_tokens:
                tid = self.digit_token_ids[d]
                probs[d] += first_probs[tid].item()

        # Two-digit numbers (10-99)
        for first_d in range(1, 10):  # First digit: 1-9
            first_tid = self.digit_token_ids[first_d]
            p_first = first_probs[first_tid].item()

            if p_first < 1e-10:
                continue  # Skip negligible first digits

            # Get conditional distribution for second digit
            extended_ids = torch.cat([
                prefix_ids,
                torch.tensor([[first_tid]], device=self.device)
            ], dim=1)
            second_logits = self._get_logits_at_position(extended_ids)
            if temperature != 1.0:
                second_logits = second_logits / temperature
            second_probs = torch.softmax(second_logits, dim=-1)

            for second_d in range(10):
                n = first_d * 10 + second_d
                if n > max_value or n < 10:
                    continue
                if n in self.direct_number_tokens:
                    continue  # Already counted via direct token

                second_tid = self.digit_token_ids[second_d]
                probs[n] += p_first * second_probs[second_tid].item()

        # Three-digit: only 100
        if max_value >= 100 and 100 not in self.direct_number_tokens:
            first_tid = self.digit_token_ids[1]
            p_first = first_probs[first_tid].item()

            if p_first > 1e-10:
                extended_1 = torch.cat([
                    prefix_ids,
                    torch.tensor([[first_tid]], device=self.device)
                ], dim=1)
                second_logits = self._get_logits_at_position(extended_1)
                if temperature != 1.0:
                    second_logits = second_logits / temperature
                second_probs = torch.softmax(second_logits, dim=-1)

                zero_tid = self.digit_token_ids[0]
                p_second = second_probs[zero_tid].item()

                if p_second > 1e-10:
                    extended_10 = torch.cat([
                        extended_1,
                        torch.tensor([[zero_tid]], device=self.device)
                    ], dim=1)
                    third_logits = self._get_logits_at_position(extended_10)
                    if temperature != 1.0:
                        third_logits = third_logits / temperature
                    third_probs = torch.softmax(third_logits, dim=-1)

                    probs[100] += p_first * p_second * third_probs[zero_tid].item()

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            # Uniform fallback (should not happen)
            probs = np.ones(max_value + 1) / (max_value + 1)

        return probs

    def extract_full_response(
        self,
        question: str,
        condition: str = "P_first",
        max_value: int = 100,
        temperature: float = 1.0,
    ) -> dict:
        """Extract distributions for both P and C values in a single response.

        Args:
            question: The probability question
            condition: "P_first" (Condition A) or "C_first" (Condition B)
            max_value: Maximum integer for distributions
            temperature: Softmax temperature

        Returns:
            Dict with keys: p_dist, c_dist, p_argmax, c_argmax,
                           first_value_prefix, second_value_prefix
        """
        if condition == "P_first":
            prompt = (
                f'Q: {question} Answer with ONLY "P=[0-100] C=[0-100]" '
                f'where P=your probability estimate and C=your confidence '
                f'in that estimate (both integers 0-100).\nA: P='
            )
            first_label = "p"
            second_label = "c"
        else:  # C_first
            prompt = (
                f'Q: {question} Answer with ONLY "C=[0-100] P=[0-100]" '
                f'where C=your confidence in your estimate and P=your '
                f'probability estimate (both integers 0-100).\nA: C='
            )
            first_label = "c"
            second_label = "p"

        # Extract first value distribution
        first_dist = self.extract_number_distribution(
            prompt, max_value=max_value, temperature=temperature
        )
        first_argmax = int(np.argmax(first_dist))

        # Build prefix for second value
        # After first number, expect " C=" or " P="
        if condition == "P_first":
            second_prefix = f"{prompt}{first_argmax} C="
        else:
            second_prefix = f"{prompt}{first_argmax} P="

        # Extract second value distribution
        second_dist = self.extract_number_distribution(
            second_prefix, max_value=max_value, temperature=temperature
        )
        second_argmax = int(np.argmax(second_dist))

        result = {
            f"{first_label}_dist": first_dist,
            f"{second_label}_dist": second_dist,
            f"{first_label}_argmax": first_argmax,
            f"{second_label}_argmax": second_argmax,
            "condition": condition,
            "first_value_prefix": prompt,
            "second_value_prefix": second_prefix,
        }

        return result


def compute_entropy(dist: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy of a probability distribution.

    Args:
        dist: Probability distribution (sums to 1)
        base: Log base (2 for bits, e for nats)

    Returns:
        Entropy value
    """
    # Filter zero probabilities to avoid log(0)
    nonzero = dist[dist > 0]
    if len(nonzero) == 0:
        return 0.0
    return -np.sum(nonzero * np.log(nonzero) / np.log(base))


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """KL divergence KL(P || Q).

    Args:
        p: Distribution P
        q: Distribution Q
        epsilon: Small value to avoid log(0)

    Returns:
        KL(P || Q) in nats
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))


def count_modes(dist: np.ndarray, threshold_fraction: float = 0.1) -> int:
    """Count local maxima in a distribution above a threshold.

    Args:
        dist: Probability distribution
        threshold_fraction: Minimum fraction of max probability to count as a mode

    Returns:
        Number of modes
    """
    if len(dist) < 3:
        return len(dist)

    max_prob = dist.max()
    threshold = max_prob * threshold_fraction

    modes = 0
    for i in range(len(dist)):
        if dist[i] < threshold:
            continue

        is_mode = True
        # Check if local maximum
        if i > 0 and dist[i] < dist[i - 1]:
            is_mode = False
        if i < len(dist) - 1 and dist[i] < dist[i + 1]:
            is_mode = False

        if is_mode:
            modes += 1

    return max(modes, 1)  # At least 1 mode
