"""Class for generate sampler using Normalizing Flows."""

from copy import deepcopy
from typing import TYPE_CHECKING, override

import numpy as np
import torch
from nflows import distributions, flows, transforms
from sklearn.model_selection import train_test_split

from .density_model import DensityModel

if TYPE_CHECKING:
    from collections.abc import Callable


def flow_model(seed: int | np.int64, dim: int) -> flows.Flow:
    """Create the standard flow model as described in [1].

    [1]: de Gelder, E.; Buermann, M. & Op den Camp, O.
         Comparing Normalizing Flows with Kernel Density Estimation in Estimating
         Risk of Automated Driving Systems
         IEEE International Automated Vehicle Validation Conference, 2025

    :param seed: seed for initializing the weights.
    :param dim: dimension of the data.
    :return: flows model.
    """
    nlayers = 4

    torch.manual_seed(seed)
    transform = []
    for _ in range(nlayers):
        transform.append(
            transforms.MaskedAffineAutoregressiveTransform(
                features=dim, hidden_features=dim * 2, dropout_probability=0.2
            )
        )
        transform.append(transforms.BatchNorm(features=dim, momentum=0.05))
        transform.append(transforms.RandomPermutation(features=dim))
    transform = transforms.CompositeTransform(transform)
    base_distribution = distributions.StandardNormal(shape=[dim])
    return flows.Flow(transform=transform, distribution=base_distribution)


class NFModel(DensityModel):
    """Density model based on Normalizing Flows."""

    max_iterations = 10000
    training_partition = 0.8
    max_patience = 100
    n_tries = 4
    n_samples = 20000  # Sample many numbers at the same time for speed purposes.
    samples: np.ndarray | None = None
    i_sample = 0

    def __init__(
        self,
        func_flow_model: Callable[[int, int], flows.Flow] = flow_model,
        seed: int | None = None,
    ) -> None:
        """Initialize NFModel.

        :param func_flow_model: function to be used to create the flows model.
        :param seed: set seed for training (and possibly sampling).
        """
        self.func_flow_model = func_flow_model
        DensityModel.__init__(self, seed=seed)

    @override
    def fit(self, data: np.ndarray | list[float] | list[list[float]]) -> NFModel:
        def _fit_single_model() -> tuple[float, dict]:
            best_state = None
            flow = self.func_flow_model(int(self._rng.integers(0, 1 << 32)), ndim)
            optimizer = torch.optim.Adam(flow.parameters())
            best_loss = np.inf
            patience = 0
            for _ in range(self.max_iterations):
                flow.train()
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=ttraining).mean()
                loss.backward()
                optimizer.step()

                flow.eval()
                loss_test = float(-flow.log_prob(inputs=ttest).mean().detach())
                if loss_test < best_loss:
                    best_loss = loss_test
                    best_state = deepcopy(flow.state_dict())
                    patience = 0
                else:
                    patience += 1
                    if patience == self.max_patience:
                        break

            if best_state is None:
                msg = "Loss remained infinite. Have a look at the data."
                raise RuntimeError(msg)

            return best_loss, best_state

        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        ndim = data.shape[1]

        training, test = train_test_split(
            data, train_size=self.training_partition, random_state=self._rng.integers(0, 1 << 32)
        )
        ttraining = torch.tensor(training, dtype=torch.float32)
        ttest = torch.tensor(test, dtype=torch.float32)
        best_state = min([_fit_single_model() for _ in range(self.n_tries)])[1]
        self.flow = self.func_flow_model(0, data.shape[1])  # Seed not important.
        self.flow.load_state_dict(best_state)
        self.flow.eval()
        self._is_fitted = True
        self.samples = None
        self.i_sample = 0
        return self

    @override
    def sample(self, n: int) -> np.ndarray:
        self._check_fitted()
        return np.array([self._sample() for _ in range(n)])

    def _sample(self) -> np.ndarray:
        if self.samples is None or self.i_sample % self.n_samples == 0:
            torch.manual_seed(self._rng.integers(0, 1 << 32))
            self.samples = self.flow.sample(self.n_samples).detach().numpy()
        self.i_sample += 1
        return self.samples[self.i_sample % self.n_samples]

    @override
    def _density(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.exp(self.flow.log_prob(data.astype(np.float32)).detach().numpy())
