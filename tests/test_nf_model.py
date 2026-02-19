"""Test scripts for NF model."""

import numpy as np
import pytest

from scenario_exposure import NFModel


@pytest.fixture(scope="module")
def example_data() -> list[float]:
    # Use a list of floats to cover more lines of the code.
    n = 100
    rng = np.random.default_rng(42)
    return [rng.normal() for _ in range(n)]


@pytest.fixture(scope="module")
def fitted_model(example_data: list[float]) -> NFModel:
    model = NFModel(seed=42)
    model.max_iterations = 1000
    model.n_tries = 2
    model.fit(example_data)
    return model


def test_nf_sample(fitted_model: NFModel) -> None:
    assert fitted_model.sample(10).shape == (10, 1)


TOLERANCE = 1e-3


def test_nf_density(fitted_model: NFModel) -> None:
    x = np.linspace(-5, 5, 100)
    y = fitted_model.density(x)
    assert -TOLERANCE <= np.trapezoid(y, x) - 1 <= TOLERANCE


def test_nf_fit_error(example_data: list[float]) -> None:
    model = NFModel(seed=42)
    model.max_iterations = 0
    with pytest.raises(RuntimeError):
        model.fit(example_data)
