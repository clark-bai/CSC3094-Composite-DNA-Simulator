"""Noisy sequencing channel simulation.

Simulates the multinomial sampling process that converts idealised
composite DNA strands into observed nucleotide frequency matrices.
For each position, the true (A, C, G, T) distribution is sampled
``coverage`` times, producing a noisy frequency vector that converges
to the true distribution as coverage increases.

This is the probabilistic model from Anavy et al. (2019).
"""

from __future__ import annotations

import numpy as np

from .alphabet import CompositeAlphabet
from .config import SimulatorConfig
from .library import Library


def simulate_reads(
    strand: np.ndarray,
    alphabet: CompositeAlphabet,
    coverage: int,
    rng: np.random.Generator,
    sub_rate: float = 0.0,
) -> np.ndarray:
    """Simulate sequencing for a single strand.

    For each position, draws ``coverage`` reads from a multinomial
    distribution defined by the composite letter's (A, C, G, T)
    probabilities, then normalises the counts to frequencies.

    Args:
        strand: Integer array of letter indices, shape
            ``(strand_length,)``.
        alphabet: The composite alphabet (provides true distributions).
        coverage: Number of reads per position.
        rng: NumPy random generator for reproducibility.
        sub_rate: Per-read substitution error probability (v2).
            Defaults to 0.0 (no substitution errors).

    Returns:
        Observed frequency matrix, shape ``(strand_length, 4)``.
        Each row sums to 1.
    """
    counts = simulate_reads_raw(strand, alphabet, coverage, rng)

    if sub_rate > 0.0:
        # v2: apply substitution errors to each individual read
        corrupted = np.zeros_like(counts)
        for pos in range(len(strand)):
            for base, n in enumerate(counts[pos]):
                for _ in range(n):
                    corrupted[pos, apply_substitutions(base, sub_rate, rng)] += 1
        counts = corrupted

    return counts / counts.sum(axis=1, keepdims=True)


def simulate_reads_raw(
    strand: np.ndarray,
    alphabet: CompositeAlphabet,
    coverage: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate sequencing and return raw read counts.

    Same as :func:`simulate_reads` but returns integer counts
    instead of normalised frequencies. Useful when the model should
    have access to coverage information.

    Args:
        strand: Integer array of letter indices, shape
            ``(strand_length,)``.
        alphabet: The composite alphabet (provides true distributions).
        coverage: Number of reads per position.
        rng: NumPy random generator for reproducibility.

    Returns:
        Raw count matrix, shape ``(strand_length, 4)``, dtype ``int``.
        Each row sums to ``coverage``.
    """
    n_positions = len(strand)
    counts = np.empty((n_positions, 4), dtype=int)
    for i in range(n_positions):
        probs = alphabet.get_distribution(strand[i])
        counts[i] = rng.multinomial(coverage, probs)
    return counts


def apply_substitutions(
    base: int,
    sub_rate: float,
    rng: np.random.Generator,
) -> int:
    """Randomly substitute a single base read (v2).

    With probability ``sub_rate``, replaces the base with one of the
    other three bases chosen uniformly at random.

    Args:
        base: Original base index (0=A, 1=C, 2=G, 3=T).
        sub_rate: Probability of substitution per read.
        rng: NumPy random generator for reproducibility.

    Returns:
        The (possibly substituted) base index.
    """
    if rng.random() < sub_rate:
        others = [b for b in range(4) if b != base]
        return int(rng.choice(others))
    return base


def simulate_library(
    library: Library,
    config: SimulatorConfig,
) -> list[np.ndarray]:
    """Simulate sequencing for every strand in a library.

    Convenience wrapper that applies :func:`simulate_reads` to each
    strand using parameters from ``config``.

    Args:
        library: The composite DNA library to sequence.
        config: Simulator configuration (provides ``coverage``,
            ``sub_rate``, and ``seed``).

    Returns:
        List of observed frequency matrices, one per strand.
        Each has shape ``(strand_length, 4)``.
    """
    rng = np.random.default_rng(config.seed)
    return [
        simulate_reads(
            strand.sequence, library.alphabet, config.coverage, rng,
            sub_rate=config.sub_rate,
        )
        for strand in library.strands
    ]
