"""Abstract class for density models."""

from abc import ABC, abstractmethod

import numpy as np


class DensityModel(ABC):
    """Abstract class for density models."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize density model.

        :param seed: set seed.
        """
        self._is_fitted = False
        self.set_seed(seed)

    @abstractmethod
    def fit(self, data: np.ndarray | list[float] | list[list[float]]) -> DensityModel:
        """Fit density model with data.

        :param data: numpy array of list of lists containing the data.
        :return: fitted model.
        """
        self._is_fitted = True
        return self

    def density(self, data: np.ndarray | list[float] | list[list]) -> np.ndarray:
        """Return estimated density for provided data.

        :param data: data for which density must be estimated.
        :return: array of estimated densities.
        """
        self._check_fitted()

        if isinstance(data, list):
            data = np.array(data)

        # If the input xdata is a 1D array, it is assumed that each entry corresponds to a
        # datapoint.
        # This might result in an error if xdata is meant to be a single (multi-dimensional)
        # datapoint.
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        if len(data.shape) == 2:  # noqa: PLR2004
            return self._density(data)

        # It is assumed that the last dimension corresponds to the dimension of the data
        # (i.e., a single datapoint).
        # Data is transformed to a 2d-array which can be used by self._density. Afterwards,
        # data is converted to input shape.
        newshape = data.shape[:-1]
        scores = self._density(data.reshape((np.prod(newshape), data.shape[-1])))
        return scores.reshape(newshape)

    @abstractmethod
    def _density(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Sample from the probability density function.

        :param n: number of samples.
        :return: n samples.
        """
        self._check_fitted()
        raise NotImplementedError

    def set_seed(self, seed: int | None) -> None:
        """Set the seed number (for sampling, possibly also for fitting)."""
        self._rng = np.random.default_rng(seed=seed)

    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            msg = "Call fit() before density() or sample()."
            raise RuntimeError(msg)
