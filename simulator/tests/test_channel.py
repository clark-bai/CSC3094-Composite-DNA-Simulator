"""Tests for channel simulation (src/simulator/channel.py)."""

# pylint: disable=redefined-outer-name  # pytest fixture injection pattern

import numpy as np
import pytest

from simulator.alphabet import CompositeAlphabet
from simulator.config import SimulatorConfig
from simulator.library import generate_library
from simulator.channel import (
    apply_substitutions,
    simulate_reads,
    simulate_reads_raw,
    simulate_library,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sigma_8():
    """Sigma-8 alphabet fixture."""
    return CompositeAlphabet.from_preset("sigma_8")

@pytest.fixture
def strand(sigma_8):
    """A short random strand for testing."""
    rng = np.random.default_rng(0)
    return rng.choice(sigma_8.size, size=10)

@pytest.fixture
def small_config():
    """Small config for fast tests."""
    return SimulatorConfig(
        n_files=2,
        n_strands_per_file=3,
        strand_length=10,
        seed=42,
    )


# ---------------------------------------------------------------------------
# simulate_reads_raw
# ---------------------------------------------------------------------------

def test_raw_shape(strand, sigma_8):
    """Raw counts have shape (strand_length, 4)."""
    rng = np.random.default_rng(0)
    counts = simulate_reads_raw(strand, sigma_8, coverage=15, rng=rng)
    assert counts.shape == (len(strand), 4)

def test_raw_row_sums(strand, sigma_8):
    """Each row of raw counts sums to coverage."""
    rng = np.random.default_rng(0)
    coverage = 20
    counts = simulate_reads_raw(strand, sigma_8, coverage=coverage, rng=rng)
    np.testing.assert_array_equal(counts.sum(axis=1), coverage)

def test_raw_dtype(strand, sigma_8):
    """Raw counts are integers."""
    rng = np.random.default_rng(0)
    counts = simulate_reads_raw(strand, sigma_8, coverage=15, rng=rng)
    assert counts.dtype == int

def test_raw_nonnegative(strand, sigma_8):
    """Raw counts are non-negative."""
    rng = np.random.default_rng(0)
    counts = simulate_reads_raw(strand, sigma_8, coverage=15, rng=rng)
    assert np.all(counts >= 0)

def test_raw_pure_base_only_produces_that_base(sigma_8):
    """A pure base (e.g. A = [1,0,0,0]) only produces reads of that base."""
    # Strand of all A's (index 0)
    strand = np.zeros(5, dtype=int)
    rng = np.random.default_rng(0)
    counts = simulate_reads_raw(strand, sigma_8, coverage=50, rng=rng)
    # All reads should be in column 0 (A)
    assert np.all(counts[:, 0] == 50)
    assert np.all(counts[:, 1:] == 0)


# ---------------------------------------------------------------------------
# simulate_reads (normalised)
# ---------------------------------------------------------------------------

def test_freq_shape(strand, sigma_8):
    """Frequency matrix has shape (strand_length, 4)."""
    rng = np.random.default_rng(0)
    freqs = simulate_reads(strand, sigma_8, coverage=15, rng=rng)
    assert freqs.shape == (len(strand), 4)

def test_freq_rows_sum_to_one(strand, sigma_8):
    """Each row of the frequency matrix sums to 1."""
    rng = np.random.default_rng(0)
    freqs = simulate_reads(strand, sigma_8, coverage=15, rng=rng)
    np.testing.assert_allclose(freqs.sum(axis=1), 1.0)

def test_freq_nonnegative(strand, sigma_8):
    """All frequencies are non-negative."""
    rng = np.random.default_rng(0)
    freqs = simulate_reads(strand, sigma_8, coverage=15, rng=rng)
    assert np.all(freqs >= 0.0)

def test_freq_converges_at_high_coverage(sigma_8):
    """At very high coverage, observed frequencies converge to true distributions."""
    # Strand: one of each mixture letter (R, Y, M, K)
    strand = np.array([4, 5, 6, 7])  # R, Y, M, K in sigma_8
    rng = np.random.default_rng(0)
    freqs = simulate_reads(strand, sigma_8, coverage=100_000, rng=rng)

    for i, letter_idx in enumerate(strand):
        expected = sigma_8.get_distribution(letter_idx)
        np.testing.assert_allclose(freqs[i], expected, atol=0.01)

def test_freq_reproducible(strand, sigma_8):
    """Same seed produces the same frequency matrix."""
    freqs1 = simulate_reads(strand, sigma_8, 15, np.random.default_rng(99))
    freqs2 = simulate_reads(strand, sigma_8, 15, np.random.default_rng(99))
    np.testing.assert_array_equal(freqs1, freqs2)


# ---------------------------------------------------------------------------
# apply_substitutions
# ---------------------------------------------------------------------------

def test_no_substitution_at_zero_rate():
    """With sub_rate=0, the base is never changed."""
    rng = np.random.default_rng(0)
    for base in range(4):
        for _ in range(100):
            assert apply_substitutions(base, 0.0, rng) == base

def test_always_substitutes_at_rate_one():
    """With sub_rate=1, the base is always changed."""
    rng = np.random.default_rng(0)
    for base in range(4):
        for _ in range(50):
            result = apply_substitutions(base, 1.0, rng)
            assert result != base
            assert 0 <= result <= 3

def test_substitution_stays_in_range():
    """Substituted base is always 0–3."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        result = apply_substitutions(1, 0.5, rng)
        assert 0 <= result <= 3

def test_substitution_introduces_noise(sigma_8):
    """With sub_rate > 0, a pure base produces some off-target reads."""
    # Strand of all A's
    strand = np.zeros(20, dtype=int)
    rng = np.random.default_rng(0)
    freqs = simulate_reads(strand, sigma_8, coverage=1000, rng=rng, sub_rate=0.1)
    # Some reads should land in non-A columns
    assert np.any(freqs[:, 1:] > 0)


# ---------------------------------------------------------------------------
# simulate_library
# ---------------------------------------------------------------------------

def test_library_returns_correct_count(small_config, sigma_8):
    """simulate_library returns one matrix per strand."""
    library = generate_library(small_config, sigma_8)
    matrices = simulate_library(library, small_config)
    assert len(matrices) == len(library.strands)

def test_library_matrix_shapes(small_config, sigma_8):
    """Each matrix has the correct shape."""
    library = generate_library(small_config, sigma_8)
    matrices = simulate_library(library, small_config)
    for mat in matrices:
        assert mat.shape == (small_config.strand_length, 4)

def test_library_rows_sum_to_one(small_config, sigma_8):
    """Every row in every matrix sums to 1."""
    library = generate_library(small_config, sigma_8)
    matrices = simulate_library(library, small_config)
    for mat in matrices:
        np.testing.assert_allclose(mat.sum(axis=1), 1.0)

def test_library_reproducible(small_config, sigma_8):
    """Same config seed produces identical results."""
    library = generate_library(small_config, sigma_8)
    matrices1 = simulate_library(library, small_config)
    matrices2 = simulate_library(library, small_config)
    for m1, m2 in zip(matrices1, matrices2):
        np.testing.assert_array_equal(m1, m2)
