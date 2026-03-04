"""Tests for library generation (src/simulator/library.py)."""

# pylint: disable=redefined-outer-name  # pytest fixture injection pattern

import numpy as np
import pytest

from simulator.alphabet import CompositeAlphabet
from simulator.config import SimulatorConfig
from simulator.library import Strand, Library, generate_strand, generate_library


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sigma_8():
    """Sigma-8 alphabet fixture."""
    return CompositeAlphabet.from_preset("sigma_8")

@pytest.fixture
def random_config():
    """Small random-mode config for fast tests."""
    return SimulatorConfig(
        n_files=4,
        n_strands_per_file=10,
        strand_length=20,
        library_mode="random",
        seed=42,
    )

@pytest.fixture
def biased_config():
    """Small biased-mode config for fast tests."""
    return SimulatorConfig(
        n_files=4,
        n_strands_per_file=10,
        strand_length=20,
        library_mode="biased",
        bias_strength=0.8,
        seed=42,
    )


# ---------------------------------------------------------------------------
# generate_strand
# ---------------------------------------------------------------------------

def test_strand_shape(sigma_8):
    """Generated strand has the correct length."""
    rng = np.random.default_rng(0)
    seq = generate_strand(60, sigma_8, rng)
    assert seq.shape == (60,)

def test_strand_indices_valid(sigma_8):
    """All letter indices are within [0, alphabet.size)."""
    rng = np.random.default_rng(0)
    seq = generate_strand(100, sigma_8, rng)
    assert np.all(seq >= 0)
    assert np.all(seq < sigma_8.size)

def test_strand_uniform_distribution(sigma_8):
    """With no weights, letters are approximately uniformly distributed."""
    rng = np.random.default_rng(0)
    seq = generate_strand(10_000, sigma_8, rng)
    counts = np.bincount(seq, minlength=sigma_8.size)
    freqs = counts / len(seq)
    expected = 1.0 / sigma_8.size
    np.testing.assert_allclose(freqs, expected, atol=0.03)

def test_strand_weighted_distribution(sigma_8):
    """With explicit weights, letter frequencies match the weights."""
    rng = np.random.default_rng(0)
    weights = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    seq = generate_strand(10_000, sigma_8, rng, letter_weights=weights)
    counts = np.bincount(seq, minlength=sigma_8.size)
    freqs = counts / len(seq)
    np.testing.assert_allclose(freqs, weights, atol=0.03)

def test_strand_reproducible(sigma_8):
    """Same seed produces the same strand."""
    seq1 = generate_strand(60, sigma_8, np.random.default_rng(99))
    seq2 = generate_strand(60, sigma_8, np.random.default_rng(99))
    np.testing.assert_array_equal(seq1, seq2)


# ---------------------------------------------------------------------------
# generate_library — random mode
# ---------------------------------------------------------------------------

def test_random_library_size(random_config, sigma_8):
    """Random library has n_files * n_strands_per_file strands."""
    lib = generate_library(random_config, sigma_8)
    assert len(lib.strands) == random_config.n_files * random_config.n_strands_per_file

def test_random_library_strand_shapes(random_config, sigma_8):
    """Every strand has the correct sequence length."""
    lib = generate_library(random_config, sigma_8)
    for strand in lib.strands:
        assert strand.sequence.shape == (random_config.strand_length,)

def test_random_library_file_ids(random_config, sigma_8):
    """File IDs are sequential: first n_strands_per_file → 0, next → 1, etc."""
    lib = generate_library(random_config, sigma_8)
    file_ids = [s.file_id for s in lib.strands]
    for file_id in range(random_config.n_files):
        start = file_id * random_config.n_strands_per_file
        end = start + random_config.n_strands_per_file
        assert all(fid == file_id for fid in file_ids[start:end])

def test_random_library_indices_valid(random_config, sigma_8):
    """All letter indices are within the alphabet."""
    lib = generate_library(random_config, sigma_8)
    for strand in lib.strands:
        assert np.all(strand.sequence >= 0)
        assert np.all(strand.sequence < sigma_8.size)

def test_random_library_metadata(random_config, sigma_8):
    """Library stores the correct alphabet and config."""
    lib = generate_library(random_config, sigma_8)
    assert lib.alphabet is sigma_8
    assert lib.config is random_config

def test_random_library_reproducible(random_config, sigma_8):
    """Same seed produces identical libraries."""
    lib1 = generate_library(random_config, sigma_8)
    lib2 = generate_library(random_config, sigma_8)
    for s1, s2 in zip(lib1.strands, lib2.strands):
        np.testing.assert_array_equal(s1.sequence, s2.sequence)
        assert s1.file_id == s2.file_id


# ---------------------------------------------------------------------------
# generate_library — biased mode
# ---------------------------------------------------------------------------

def test_biased_library_size(biased_config, sigma_8):
    """Biased library has the correct total number of strands."""
    lib = generate_library(biased_config, sigma_8)
    assert len(lib.strands) == biased_config.n_files * biased_config.n_strands_per_file

def test_biased_library_file_ids(biased_config, sigma_8):
    """Biased library has correct sequential file IDs."""
    lib = generate_library(biased_config, sigma_8)
    file_ids = [s.file_id for s in lib.strands]
    for file_id in range(biased_config.n_files):
        start = file_id * biased_config.n_strands_per_file
        end = start + biased_config.n_strands_per_file
        assert all(fid == file_id for fid in file_ids[start:end])

def test_biased_library_files_differ(biased_config, sigma_8):
    """With high bias, per-file letter distributions are visibly different."""
    lib = generate_library(biased_config, sigma_8)
    n = biased_config.n_strands_per_file

    file_freqs = []
    for file_id in range(biased_config.n_files):
        start = file_id * n
        seqs = np.concatenate([s.sequence for s in lib.strands[start:start + n]])
        counts = np.bincount(seqs, minlength=sigma_8.size)
        file_freqs.append(counts / counts.sum())

    # At least one pair of files should have noticeably different distributions
    max_diff = 0.0
    for i in range(len(file_freqs)):
        for j in range(i + 1, len(file_freqs)):
            max_diff = max(max_diff, np.max(np.abs(file_freqs[i] - file_freqs[j])))
    assert max_diff > 0.05


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_library_mode_raises(sigma_8):
    """An unknown library_mode raises ValueError."""
    config = SimulatorConfig(library_mode="unknown")
    with pytest.raises(ValueError, match="Unknown library_mode"):
        generate_library(config, sigma_8)
