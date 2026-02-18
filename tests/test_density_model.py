"""Test for some errors to be raised in density_model.py."""

from typing import override

import numpy as np
import pytest

from scenario_exposure import DensityModel


class WrongDensityModel(DensityModel):
    @override
    def fit(self, data: np.ndarray | list[float] | list[list[float]]) -> DensityModel:
        return DensityModel.fit(self, data)

    @override
    def _density(self, data: np.ndarray) -> np.ndarray:
        return DensityModel._density(self, np.array([1.0]))  # noqa: SLF001  # type: ignore[reportAbstractUsage]

    @override
    def sample(self, n: int) -> np.ndarray:
        return DensityModel.sample(self, n)  # type: ignore[reportAbstractUsage]


def test_no_density_method() -> None:
    density_model = WrongDensityModel()
    density_model.fit([1.0])
    with pytest.raises(NotImplementedError):
        density_model.density([1.0])


def test_no_sample_method() -> None:
    density_model = WrongDensityModel()
    with pytest.raises(RuntimeError):
        density_model.sample(1)
    density_model.fit([1.0])
    with pytest.raises(NotImplementedError):
        density_model.sample(1)
