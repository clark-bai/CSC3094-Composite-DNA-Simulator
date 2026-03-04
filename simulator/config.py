"""Simulator configuration dataclass with scope-freeze defaults."""

import dataclasses


@dataclasses.dataclass
class SimulatorConfig:
    """All parameters for the composite DNA simulator.

    All fields have scope-freeze defaults. Override only what you need.
    """
    # Alphabet
    alphabet_name: str = "sigma_8"           # Preset name, or "custom"
    custom_distributions: list | None = None  # Required if alphabet_name == "custom"

    # Library
    n_files: int = 20
    n_strands_per_file: int = 500
    strand_length: int = 60
    library_mode: str = "random"             # "random" or "biased"
    bias_strength: float = 0.0               # Only used if library_mode == "biased"

    # Channel
    coverage: int = 15                       # Reads per position
    sub_rate: float = 0.0                    # Substitution error rate (v2)
    ins_rate: float = 0.0                    # Insertion error rate (v3)
    del_rate: float = 0.0                    # Deletion error rate (v3)

    # Dataset
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Reproducibility
    seed: int = 42
