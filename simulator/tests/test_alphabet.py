"""Tests for CompositeAlphabet (src/simulator/alphabet.py)."""

# pylint: disable=redefined-outer-name  # pytest fixture injection pattern

import numpy as np
import pytest

from simulator.alphabet import CompositeAlphabet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sigma_6():
    """Sigma-6 alphabet fixture."""
    return CompositeAlphabet.from_preset("sigma_6")

@pytest.fixture
def sigma_8():
    """Sigma-8 alphabet fixture."""
    return CompositeAlphabet.from_preset("sigma_8")

@pytest.fixture
def sigma_15():
    """Sigma-15 alphabet fixture."""
    return CompositeAlphabet.from_preset("sigma_15")


# ---------------------------------------------------------------------------
# Preset loading
# ---------------------------------------------------------------------------

def test_preset_sizes(sigma_6, sigma_8, sigma_15):
    """Each preset has the correct number of letters."""
    assert sigma_6.size == 6
    assert sigma_8.size == 8
    assert sigma_15.size == 15

def test_unknown_preset_raises():
    """Loading an unknown preset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset"):
        CompositeAlphabet.from_preset("sigma_99")

def test_preset_labels_sigma_8(sigma_8):
    """Sigma-8 letters are in the expected order."""
    assert sigma_8.letters == ["A", "C", "G", "T", "R", "Y", "M", "K"]


# ---------------------------------------------------------------------------
# Distributions sum to 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["sigma_6", "sigma_8", "sigma_15"])
def test_distributions_sum_to_one(preset):
    """Every row in the distribution matrix sums to 1.0."""
    alpha = CompositeAlphabet.from_preset(preset)
    sums = alpha.distributions.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, err_msg=f"{preset} rows do not sum to 1")

def test_distributions_shape(sigma_8):
    """Distribution matrix has shape (size, 4)."""
    assert sigma_8.distributions.shape == (8, 4)


# ---------------------------------------------------------------------------
# bits_per_letter
# ---------------------------------------------------------------------------

def test_bits_per_letter_sigma_8(sigma_8):
    """Sigma-8 has exactly 3 bits per letter."""
    assert sigma_8.bits_per_letter == pytest.approx(3.0)

def test_bits_per_letter_sigma_6(sigma_6):
    """Sigma-6 bits per letter equals log2(6)."""
    assert sigma_6.bits_per_letter == pytest.approx(np.log2(6))


# ---------------------------------------------------------------------------
# get_distribution
# ---------------------------------------------------------------------------

def test_get_distribution_returns_correct_row(sigma_8):
    """get_distribution(i) returns the same row as distributions[i]."""
    for i in range(sigma_8.size):
        np.testing.assert_array_equal(sigma_8.get_distribution(i), sigma_8.distributions[i])


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def test_kl_self_divergence_is_zero(sigma_8):
    """KL(p || p) = 0 for every letter in the alphabet."""
    for i in range(sigma_8.size):
        p = sigma_8.get_distribution(i)
        assert sigma_8.kl_divergence(p, i) == pytest.approx(0.0)

def test_kl_divergence_is_nonnegative(sigma_8):
    """KL divergence is always >= 0."""
    rng = np.random.default_rng(42)
    observed = rng.dirichlet(np.ones(4))
    for i in range(sigma_8.size):
        assert sigma_8.kl_divergence(observed, i) >= 0.0


# ---------------------------------------------------------------------------
# nearest_letter
# ---------------------------------------------------------------------------

def test_nearest_letter_perfect_observation(sigma_8):
    """A perfect (noise-free) observation decodes to the correct letter."""
    for i in range(sigma_8.size):
        p = sigma_8.get_distribution(i)
        assert sigma_8.nearest_letter(p) == i

def test_nearest_letter_noisy_observation(sigma_8):
    """A noisy but realistic observation decodes to the correct letter.

    In a v1 channel (no substitution errors), letter R = [0.5, 0, 0.5, 0]
    only produces A and G reads, so C and T frequencies are always exactly 0.
    """
    r_index = sigma_8.letters.index("R")
    noisy_r = np.array([0.47, 0.0, 0.53, 0.0])
    assert sigma_8.nearest_letter(noisy_r) == r_index


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_mismatched_lengths_raises():
    """Mismatched distributions and labels raises ValueError."""
    with pytest.raises(ValueError, match="same length"):
        CompositeAlphabet([(1.0, 0.0, 0.0, 0.0)], ["A", "C"])

def test_wrong_vector_length_raises():
    """A distribution vector that is not length 4 raises ValueError."""
    with pytest.raises(ValueError, match="4-element"):
        CompositeAlphabet([(1.0, 0.0, 0.0)], ["A"])  # type: ignore[arg-type]

def test_negative_value_raises():
    """A negative probability value raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        CompositeAlphabet([(-0.5, 1.5, 0.0, 0.0)], ["A"])

def test_does_not_sum_to_one_raises():
    """A distribution that does not sum to 1 raises ValueError."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        CompositeAlphabet([(0.5, 0.0, 0.0, 0.0)], ["A"])
