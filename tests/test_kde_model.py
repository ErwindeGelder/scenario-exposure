"""Test scripts for KDE model."""

import numpy as np
import pytest

from scenario_exposure import KDEModel


def test_kde_model_1d() -> None:
    data = [1.0, 2.0, 3.0]
    kde_model = KDEModel()
    assert kde_model.fit(data, bandwidth=1.0).sample(10).shape == (10, 1)


def test_kde_model_3d_silverman() -> None:
    data = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    kde_model = KDEModel()
    kde_model.fit(data, silverman=True)


def test_warning_silverman() -> None:
    data = [1.0, 2.0, 3.0]
    kde_model = KDEModel()
    with pytest.warns(UserWarning, match="The data is not scaled."):
        kde_model.fit(data, scaling=False, silverman=True)


def test_kde_density() -> None:
    data1 = [1.0, 2.0, 3.0]
    kde_model = KDEModel()
    assert kde_model.fit(data1, silverman=True).density([1.0, 2.0, 3.0]).shape == (3,)

    data2 = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    kde_model = KDEModel()
    kde_model.fit(data2, silverman=True)
    assert kde_model.density([[1.0, 1.0, 1.0]]).shape == (1,)
    assert kde_model.density([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).shape == (2,)
    assert kde_model.density([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]).shape == (1, 2)


TOLERANCE = 1e-3


def test_kde_gss() -> None:
    kde_model = KDEModel()
    kde_model.fit(np.linspace(1, 3, 10))
    x = np.linspace(-2, 6, 100)
    y = kde_model.density(x)
    assert -TOLERANCE <= np.trapezoid(y, x) - 1 <= TOLERANCE


def test_kde_gss_warnings() -> None:
    kde_model = KDEModel()
    with pytest.warns(UserWarning, match="Only searched on right side"):
        kde_model.fit(np.linspace(1, 3, 3))
    kde_model.max_bw = 100
    kde_model.min_bw = 99
    with pytest.warns(UserWarning, match="Only searched on left side"):
        kde_model.fit(np.linspace(1, 3, 3))


def test_alternative_density_estimation() -> None:
    kde_model = KDEModel()
    kde_model.threshold_memory_saver = 100
    kde_model.fit(np.linspace(0, 3, 100))
    x = np.linspace(-5, 8, 100)
    y = kde_model.density(x)
    assert -TOLERANCE <= np.trapezoid(y, x) - 1 <= TOLERANCE


def test_very_low_loo_score() -> None:
    kde_model = KDEModel()
    kde_model.fit(np.linspace(1, 3, 10), bandwidth=1e-5)
    assert kde_model.score_leave_one_out() == -np.inf


def test_gss_error() -> None:
    kde_model = KDEModel()
    kde_model.min_bw = 2.0
    kde_model.max_bw = 1.0
    with pytest.raises(ValueError, match="Maximum bandwidth must be larger than minimum bandwidth"):
        kde_model.fit(np.linspace(1, 3, 10))
