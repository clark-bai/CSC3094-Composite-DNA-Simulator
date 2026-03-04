"""Composite DNA alphabet definition and KL-based decoding utilities."""

import numpy as np


class CompositeAlphabet:
    """
    A composite DNA alphabet where each letter is an (A, C, G, T) probability vector.

    Use ``from_preset`` or ``from_distributions`` to construct an instance.
    """

    def __init__(
        self,
        distributions: list[tuple[float, float, float, float]],
        labels: list[str],
    ) -> None:
        if len(distributions) != len(labels):
            raise ValueError("distributions and labels must be the same length")
        arr = np.array(distributions, dtype=float)
        
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("each distribution must be a 4-element (A, C, G, T) vector")
        if np.any(arr < 0):
            raise ValueError("distribution values must be non-negative")
        sums = arr.sum(axis=1)
        if not np.allclose(sums, 1.0):
            raise ValueError(f"each distribution must sum to 1.0; got sums: {sums}")
        
        self._labels = labels
        self._distributions = arr  # shape (size, 4)

    @classmethod
    def from_preset(cls, name: str) -> "CompositeAlphabet":
        """
        Load a named preset: ``"sigma_6"``, ``"sigma_8"``, or ``"sigma_15"``.

        Raises ValueError for unknown names.
        """
        presets = {
            "sigma_6": {
                "labels": ["A", "C", "G", "T", "M", "K"],
                "distributions": [
                    (1.0, 0.0, 0.0, 0.0),  # A
                    (0.0, 1.0, 0.0, 0.0),  # C
                    (0.0, 0.0, 1.0, 0.0),  # G
                    (0.0, 0.0, 0.0, 1.0),  # T
                    (0.5, 0.5, 0.0, 0.0),  # M = A+C
                    (0.0, 0.0, 0.5, 0.5),  # K = G+T
                ],
            },
            "sigma_8": {
                "labels": ["A", "C", "G", "T", "R", "Y", "M", "K"],
                "distributions": [
                    (1.0, 0.0, 0.0, 0.0),  # A
                    (0.0, 1.0, 0.0, 0.0),  # C
                    (0.0, 0.0, 1.0, 0.0),  # G
                    (0.0, 0.0, 0.0, 1.0),  # T
                    (0.5, 0.0, 0.5, 0.0),  # R = A+G
                    (0.0, 0.5, 0.0, 0.5),  # Y = C+T
                    (0.5, 0.5, 0.0, 0.0),  # M = A+C
                    (0.0, 0.0, 0.5, 0.5),  # K = G+T
                ],
            },
            "sigma_15": {
                "labels": ["A", "C", "G", "T", "R", "Y", "S", "W", "K", "M", "B", "D", "H", "V", "N"],
                "distributions": [
                    (1.00, 0.00, 0.00, 0.00),  # A
                    (0.00, 1.00, 0.00, 0.00),  # C
                    (0.00, 0.00, 1.00, 0.00),  # G
                    (0.00, 0.00, 0.00, 1.00),  # T
                    (0.50, 0.00, 0.50, 0.00),  # R = A+G
                    (0.00, 0.50, 0.00, 0.50),  # Y = C+T
                    (0.00, 0.50, 0.50, 0.00),  # S = C+G
                    (0.50, 0.00, 0.00, 0.50),  # W = A+T
                    (0.00, 0.00, 0.50, 0.50),  # K = G+T
                    (0.50, 0.50, 0.00, 0.00),  # M = A+C
                    (0.00, 1/3,  1/3,  1/3 ),  # B = C+G+T
                    (1/3,  0.00, 1/3,  1/3 ),  # D = A+G+T
                    (1/3,  1/3,  0.00, 1/3 ),  # H = A+C+T
                    (1/3,  1/3,  1/3,  0.00),  # V = A+C+G
                    (0.25, 0.25, 0.25, 0.25),  # N = A+C+G+T
                ],
            },
        }

        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets.keys())}")

        preset = presets[name]
        return cls.from_distributions(preset["distributions"], preset["labels"])

    @classmethod
    def from_distributions(
        cls,
        distributions: list[tuple[float, float, float, float]],
        labels: list[str],
    ) -> "CompositeAlphabet":
        """Build a custom alphabet from explicit (A, C, G, T) tuples and letter labels."""
        return cls(distributions, labels)

    @property
    def size(self) -> int:
        """Number of letters in the alphabet."""
        return len(self._labels)

    @property
    def bits_per_letter(self) -> float:
        """Information content per position in bits (log2 of alphabet size)."""
        return np.log2(self.size)

    @property
    def letters(self) -> list[str]:
        """Human-readable letter labels, e.g. ``["A", "C", "G", "T", "R", "Y", "M", "K"]``."""
        return self._labels

    @property
    def distributions(self) -> np.ndarray:
        """(A, C, G, T) probability vectors for all letters. Shape: ``(size, 4)``."""
        return self._distributions

    def get_distribution(self, letter_index: int) -> np.ndarray:
        """Return the (A, C, G, T) probability vector for one letter. Shape: ``(4,)``."""
        return self._distributions[letter_index]

    def kl_divergence(self, observed: np.ndarray, letter_index: int) -> float:
        """
        KL(observed || distribution[letter_index]).
        Core of the KL decoder.
        """
        p = observed
        q = self._distributions[letter_index]
        # If p[i] > 0 and q[i] = 0, the observation is impossible under q → infinite divergence
        if np.any((p > 0) & (q == 0)):
            return float("inf")
        # When p[i] = 0, contribution is 0 by convention (0 * log(0/q) = 0)
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

    def nearest_letter(self, observed: np.ndarray) -> int:
        """
        Return the letter index with minimum KL divergence to observed.
        Implements Anavy et al.'s decoder.
        """
        divergences = [self.kl_divergence(observed, i) for i in range(self.size)]
        return int(np.argmin(divergences))
