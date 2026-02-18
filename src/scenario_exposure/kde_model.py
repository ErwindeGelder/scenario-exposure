"""Class for generate sampler using Kernel Density Estimation."""

import warnings
from dataclasses import dataclass, field
from typing import override

import numpy as np
import scipy.spatial.distance as dist
import scipy.stats

from .density_model import DensityModel


def _empty_array() -> np.ndarray:
    return np.array([])


@dataclass(slots=True)
class KDEConstants:
    """Constants that are used for the various methods."""

    const_looscore: float = 0.0
    const_score: float = 0.0
    exp_const_score: float = 0.0
    ndata: int = 0
    dim: int = 0
    invgr: float = (np.sqrt(5) - 1) / 2
    invgr2: float = (3 - np.sqrt(5)) / 2


@dataclass(slots=True)
class KDEData:
    """Data fields that are used for various methods."""

    std: np.ndarray = field(default_factory=_empty_array)
    mindists: np.ndarray = field(default_factory=_empty_array)


class KDEModel(DensityModel):
    """Density model based on Kernel Density Estimation."""

    min_bw: float = 0.001
    max_bw: float = 1.0
    tolerance: float = 1e-5
    max_iter: int = 100
    threshold_memory_saver: int = 10000

    def fit(
        self,
        data: np.ndarray | list[float] | list[list[float]],
        bandwidth: float | None = None,
        *,
        scaling: bool = True,
        silverman: bool = False,
    ) -> KDEModel:
        """Fit density model with data.

        :param data: numpy array of list of lists containing the data.
        :param bandwidth: bandwidth of the KDE; if not set, it will be computed.
        :param scaling: whether scaling of the data is used.
        :param silverman: whether to use Silverman's bandwidth (fast but less accurate).
        :return: fitted model.
        """
        self.data_helpers = KDEData()
        self.constants = KDEConstants()
        self.scaling = scaling
        self._set_data(data)

        if self.scaling:
            # Take minimum of "standard deviation" and "interquartile range / 1.349".
            self.data_helpers.std = np.minimum(
                np.std(self.data, axis=0), scipy.stats.iqr(self.data, axis=0) / 1.349
            )

        if bandwidth is None:
            if silverman:
                self._silverman()
            else:
                self._compute_bandwidth_gss()
        else:
            self.set_bandwidth(bandwidth)

        self._is_fitted = True
        return self

    def _set_data(self, data: np.ndarray | list[float] | list[list[float]]) -> None:
        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        self.constants.ndata = data.shape[0]
        self.constants.dim = data.shape[1]
        self.constants.const_looscore = -self.constants.ndata * (
            self.constants.dim / 2 * np.log(2 * np.pi) + np.log(self.constants.ndata - 1)
        )
        self.data = data

    def set_bandwidth(self, bandwidth: float) -> None:
        """Set the bandwidth of the KDE.

        :param bandwidth: float
        """
        self._bandwidth = bandwidth
        self.constants.const_score = -self.constants.dim / 2 * np.log(
            2 * np.pi
        ) - self.constants.dim * np.log(self._bandwidth)
        self.constants.const_score -= np.log(self.constants.ndata)
        if self.scaling:
            self.constants.const_score -= np.sum(np.log(self.data_helpers.std))
        self.constants.exp_const_score = np.exp(self.constants.const_score)

    def _silverman(self) -> None:
        if not self.scaling:
            warnings.warn(
                "The data is not scaled. This produces an incorrect value for Silverman's "
                "bandwidth!",
                UserWarning,
                stacklevel=2,
            )

        self.set_bandwidth(
            (4 / (self.constants.dim + 2)) ** (1 / (self.constants.dim + 4))
            * self.constants.ndata ** (-1 / (self.constants.dim + 4))
        )

    def _compute_bandwidth_gss(self) -> None:
        # Use Golden-section search to find optimal bandwidth.
        difference = self.max_bw - self.min_bw
        datapoints = np.array([self.min_bw, 0, 0, self.max_bw], dtype=float)
        datapoints[1] = datapoints[0] + self.constants.invgr2 * difference
        datapoints[2] = datapoints[0] + self.constants.invgr * difference
        if difference <= 0:
            msg = (
                "Maximum bandwidth must be larger than minimum bandwidth. Now "
                f"min_bw={self.min_bw}, max_bw={self.max_bw}."
            )
            raise ValueError(msg)

        # required steps to achieve tolerance
        n_iter = int(np.ceil(np.log(self.tolerance / difference) / np.log(self.constants.invgr)))
        n_iter = max(1, min(n_iter, self.max_iter))

        scores = [
            self.score_leave_one_out(bandwidth=datapoints[1]),
            self.score_leave_one_out(bandwidth=datapoints[2]),
        ]
        at_boundary_min = False  # Check if we only search at one side as this could indicate ...
        at_boundary_max = False  # ... wrong values of min_bw and max_bw.
        for _ in range(n_iter):
            if scores[0] > scores[1]:
                at_boundary_min = True
                datapoints[3] = datapoints[2]
                datapoints[2] = datapoints[1]
                scores[1] = scores[0]
                difference = self.constants.invgr * difference
                datapoints[1] = datapoints[0] + self.constants.invgr2 * difference
                scores[0] = self.score_leave_one_out(bandwidth=datapoints[1])
            else:
                at_boundary_max = True
                datapoints[0] = datapoints[1]
                datapoints[1] = datapoints[2]
                scores[0] = scores[1]
                difference = self.constants.invgr * difference
                datapoints[2] = datapoints[0] + self.constants.invgr * difference
                scores[1] = self.score_leave_one_out(bandwidth=datapoints[2])

        self.set_bandwidth(
            (datapoints[0] + datapoints[2]) / 2
            if scores[0] < scores[1]
            else (datapoints[3] + datapoints[1]) / 2
        )

        # Check if we only searched on one side.
        if not at_boundary_min:
            warnings.warn(
                "Only searched on right side. Might need to increase max_bw.", stacklevel=2
            )
        if not at_boundary_max:
            warnings.warn(
                "Only searched on left side. Might need to decrease min_bw.", stacklevel=2
            )

    def score_leave_one_out(self, bandwidth: float | None = None) -> float:
        """Return the leave-one-out score.

        :param bandwidth: Optional bandwidth to be used when computing the score.
        :return: Leave-one-out score.
        """
        # Check if the distance matrix is defined. If not, create it (this takes some time).
        if not self.data_helpers.mindists.size:
            distances = dist.pdist(self.data, metric="sqeuclidean")
            self.data_helpers.mindists = -dist.squareform(distances) / 2

        # Compute the one-leave-out score.
        bandwidth = self._bandwidth if bandwidth is None else bandwidth
        if self.constants.ndata >= self.threshold_memory_saver:
            # Do this to save memory.
            score_vector = np.zeros(self.constants.ndata)
            for i in range(self.constants.ndata):
                score_vector += np.exp(self.data_helpers.mindists[i, :] / bandwidth**2)
            score_vector -= 1
            score = np.sum(np.log(score_vector))
            score -= self.constants.ndata * self.constants.dim * np.log(bandwidth)
        else:
            score = np.sum(np.exp(self.data_helpers.mindists / bandwidth**2), axis=0) - 1
            if np.any(score <= 0.0):
                score = -np.inf
            else:
                score = np.sum(np.log(score)) - self.constants.ndata * self.constants.dim * np.log(
                    bandwidth
                )
        return score + self.constants.const_looscore

    @override
    def sample(self, n: int) -> np.ndarray:
        self._check_fitted()
        uniform_vars = self._rng.uniform(0, 1, size=n)
        i = (uniform_vars * self.constants.ndata).astype(int)
        selected_data = self.data[i]
        samples = self._rng.normal(selected_data, self._bandwidth)
        if self.scaling:
            samples *= self.data_helpers.std
        return samples

    @override
    def _density(self, data: np.ndarray) -> np.ndarray:
        # Compute the distance of the datapoints in x to the datapoints of the KDE
        # Let x have M datapoints, then the result is a (self.constants.n-by-M)-matrix
        if self.scaling:
            data = data / self.data_helpers.std
        eucl_dist = dist.cdist(self.data, data, metric="sqeuclidean")

        # Note that we have f(x,n) = sum [ (2pi)^(-d/2)/(n h^d) * exp{-(x-xi)^2/(2h**2)} ]
        #                          = (2pi)^(-d/2)/(n h^d) * sum_{i=1}^n [ exp{-(x-xi)^2/(2h**2)} ]
        # We first compute the sum. Then the log of f(x,n) is computed:
        # log(f(x,n)) = -d/2*log(2pi) - log(n) - d*log(h) + log(sum)
        sum_kernel = np.zeros(eucl_dist.shape[1])
        for dimension in eucl_dist:
            sum_kernel += np.exp(-dimension / (2 * self._bandwidth**2))
        return self.constants.exp_const_score * sum_kernel
