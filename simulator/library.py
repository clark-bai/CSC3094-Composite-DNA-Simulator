"""Composite DNA library generation.

Generates pools of composite DNA strands, each tagged with a file ID.
Supports two modes:

- **Random** (primary): each strand is an independently random composite
  sequence; file labels are arbitrary groupings. The model must learn
  strand identity purely from sequence content through noise.
- **Biased** (ablation): each file receives a Dirichlet-perturbed
  distribution over alphabet letters, giving same-file strands a shared
  statistical fingerprint. Controlled by ``bias_strength``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .alphabet import CompositeAlphabet
from .config import SimulatorConfig


@dataclasses.dataclass
class Strand:
    """A single composite DNA strand with its file label.

    Attributes:
        sequence: Integer array of letter indices into the alphabet.
            Shape ``(strand_length,)``, dtype ``int``.
        file_id: Which file (0-indexed) this strand belongs to.
    """

    sequence: np.ndarray
    file_id: int


@dataclasses.dataclass
class Library:
    """A collection of composite DNA strands with metadata.

    Attributes:
        strands: All strands in the library.
        alphabet: The composite alphabet used to generate the strands.
        config: The simulator configuration that produced this library.
    """

    strands: list[Strand]
    alphabet: CompositeAlphabet
    config: SimulatorConfig


def generate_strand(
    length: int,
    alphabet: CompositeAlphabet,
    rng: np.random.Generator,
    letter_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Generate a single random composite strand.

    Each position is sampled independently from the alphabet, either
    uniformly or according to ``letter_weights``.

    Args:
        length: Number of composite positions in the strand.
        alphabet: The composite alphabet to sample letters from.
        rng: NumPy random generator for reproducibility.
        letter_weights: Optional probability weights over alphabet letters.
            Shape ``(alphabet.size,)``, must sum to 1. If ``None``, letters
            are chosen uniformly.

    Returns:
        Integer array of letter indices, shape ``(length,)``.
    """
    return rng.choice(alphabet.size, size=length, p=letter_weights)


def generate_library(
    config: SimulatorConfig,
    alphabet: CompositeAlphabet,
) -> Library:
    """Generate a full composite DNA library.

    Dispatches to random or biased mode based on
    ``config.library_mode``.

    In **random mode**, every strand is independently random and file
    labels are arbitrary sequential groupings.

    In **biased mode**, each file gets a Dirichlet-perturbed letter
    distribution controlled by ``config.bias_strength`` (0.0 = uniform,
    1.0 = maximally different). Strands within a file are sampled from
    that file's distribution.

    Args:
        config: Simulator configuration (provides ``n_files``,
            ``n_strands_per_file``, ``strand_length``, ``library_mode``,
            ``bias_strength``, and ``seed``).
        alphabet: The composite alphabet to use.

    Returns:
        A :class:`Library` containing all generated strands.

    Raises:
        ValueError: If ``config.library_mode`` is not ``"random"``
            or ``"biased"``.
    """
    if config.library_mode not in ("random", "biased"):
        raise ValueError(
            f"Unknown library_mode '{config.library_mode}'. "
            "Choose 'random' or 'biased'."
        )

    rng = np.random.default_rng(config.seed)
    strands: list[Strand] = []

    if config.library_mode == "random":
        for file_id in range(config.n_files):
            for _ in range(config.n_strands_per_file):
                seq = generate_strand(config.strand_length, alphabet, rng)
                strands.append(Strand(sequence=seq, file_id=file_id))
    else:
        # Dirichlet concentration inversely related to bias_strength:
        #   bias_strength=0.0 → alpha=10.1 → samples ≈ uniform (same as random)
        #   bias_strength=1.0 → alpha=0.1  → sparse, peaked per-file distributions
        concentration = (1.0 - config.bias_strength) * 10.0 + 0.1
        alpha = np.full(alphabet.size, concentration)

        for file_id in range(config.n_files):
            file_weights = rng.dirichlet(alpha)
            for _ in range(config.n_strands_per_file):
                seq = generate_strand(
                    config.strand_length, alphabet, rng,
                    letter_weights=file_weights,
                )
                strands.append(Strand(sequence=seq, file_id=file_id))

    return Library(strands=strands, alphabet=alphabet, config=config)
