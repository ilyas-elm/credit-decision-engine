"""Typed access to config/default.toml.

Thresholds, split proportions and experiment parameters are read from TOML, not
hardcoded, so that changing one is a visible diff with a reason attached. tomllib
is standard library on 3.11+, which keeps the Kaggle kernels free of a YAML pin.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.toml"

__all__ = ["Config", "SplitsConfig", "SamplingBiasWindows", "load_config"]


def _require_fractions_sum_to_one(name: str, fractions: dict[str, float]) -> None:
    total = sum(fractions.values())
    # Exact equality is the wrong test for floats written by hand in TOML.
    if abs(total - 1.0) > 1e-9:
        parts = ", ".join(f"{k}={v}" for k, v in fractions.items())
        raise ValueError(
            f"[{name}] fractions must sum to 1.0, got {total!r} ({parts}). "
            "Weeks would be silently dropped or double-counted otherwise."
        )


@dataclass(frozen=True)
class SplitsConfig:
    """Out-of-time split proportions, as fractions of the distinct WEEK_NUM values."""

    train_frac: float
    val_a_frac: float
    val_b_frac: float
    test_frac: float

    def __post_init__(self) -> None:
        _require_fractions_sum_to_one(
            "splits",
            {
                "train_frac": self.train_frac,
                "val_a_frac": self.val_a_frac,
                "val_b_frac": self.val_b_frac,
                "test_frac": self.test_frac,
            },
        )
        for name, value in self.as_dict().items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"[splits] {name} must lie strictly in (0, 1), got {value!r}.")

    def as_dict(self) -> dict[str, float]:
        return {
            "train_frac": self.train_frac,
            "val_a_frac": self.val_a_frac,
            "val_b_frac": self.val_b_frac,
            "test_frac": self.test_frac,
        }

    @property
    def ordered_fractions(self) -> list[tuple[str, float]]:
        """Split names in chronological order, oldest first."""
        return [
            ("train", self.train_frac),
            ("val_a", self.val_a_frac),
            ("val_b", self.val_b_frac),
            ("test", self.test_frac),
        ]


@dataclass(frozen=True)
class SamplingBiasWindows:
    """Time windows for the reject-inference experiment.

    Deliberately independent of :class:`SplitsConfig`: the experiment needs its
    own policy-fitting period, and tying the two together would make a change to
    the main split silently alter the experiment.
    """

    policy_frac: float
    dev_frac: float
    test_frac: float

    def __post_init__(self) -> None:
        _require_fractions_sum_to_one(
            "sampling_bias",
            {
                "policy_frac": self.policy_frac,
                "dev_frac": self.dev_frac,
                "test_frac": self.test_frac,
            },
        )

    @property
    def ordered_fractions(self) -> list[tuple[str, float]]:
        return [("policy", self.policy_frac), ("dev", self.dev_frac), ("test", self.test_frac)]


@dataclass(frozen=True)
class Config:
    """The whole config file, with the validated sections broken out."""

    raw: dict[str, Any] = field(repr=False)
    splits: SplitsConfig
    sampling_bias_windows: SamplingBiasWindows

    def section(self, name: str) -> dict[str, Any]:
        """One top-level table, raising a useful error when it is absent."""
        try:
            return self.raw[name]
        except KeyError:
            available = ", ".join(sorted(self.raw))
            raise KeyError(f"No [{name}] section in config. Available: {available}.") from None

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self.section(name)


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate the config file."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, "rb") as handle:
        raw = tomllib.load(handle)

    splits = SplitsConfig(**raw["splits"])
    sb = raw["sampling_bias"]
    windows = SamplingBiasWindows(
        policy_frac=sb["policy_frac"], dev_frac=sb["dev_frac"], test_frac=sb["test_frac"]
    )
    return Config(raw=raw, splits=splits, sampling_bias_windows=windows)
