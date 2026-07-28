"""Demonstrate density estimation with KDE and Normalizing Flows (NF).

A standard 2D distribution (a two-component Gaussian mixture) is sampled with an
increasing number of points (100, 500, 2000). For each sample size, both a
KDEModel and an NFModel are fit on the samples. The true density is plotted
alongside the KDE and NF estimates so the effect of the sample size on
estimation quality can be compared.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

import scenario_exposure as se

# Ground-truth distribution: a two-component Gaussian mixture.
WEIGHTS = np.array([0.4, 0.6])
MEANS = np.array([[-1.5, -1.0], [1.5, 1.0]])
COVS = np.array([[[0.6, 0.3], [0.3, 0.6]], [[0.5, -0.2], [-0.2, 0.5]]])


def true_density(xy: np.ndarray) -> np.ndarray:
    """Evaluate the true mixture density at the given points.

    :param xy: array of shape (n, 2) with the points to evaluate.
    :return: array of shape (n,) with the true density at each point.
    """
    density = np.zeros(xy.shape[0])
    for weight, mean, cov in zip(WEIGHTS, MEANS, COVS, strict=True):
        density += weight * multivariate_normal(mean=mean, cov=cov).pdf(xy)
    return density


def sample_true(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from the true mixture distribution.

    :param n: number of samples to draw.
    :param rng: random number generator to use.
    :return: array of shape (n, 2) with the samples.
    """
    components = rng.choice(len(WEIGHTS), size=n, p=WEIGHTS)
    samples = np.empty((n, 2))
    for i, component in enumerate(components):
        samples[i] = rng.multivariate_normal(MEANS[component], COVS[component])
    return samples


if __name__ == "__main__":
    sample_sizes = [100, 500, 2000]
    seed = 42
    rng = np.random.default_rng(seed)

    # Grid on which the densities are evaluated for plotting.
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(x, y)
    grid = np.array([xx.ravel(), yy.ravel()]).T
    z_true = true_density(grid).reshape(xx.shape)

    fig, axes = plt.subplots(len(sample_sizes), 3, figsize=(12, 4 * len(sample_sizes)))

    for row, n in enumerate(sample_sizes):
        data = sample_true(n, rng)

        kde = se.KDEModel(seed=seed).fit(data)
        z_kde = kde.density(grid).reshape(xx.shape)

        nf = se.NFModel(seed=seed)
        nf.max_iterations = 1000
        nf.n_tries = 1
        nf.fit(data)
        z_nf = nf.density(grid).reshape(xx.shape)

        for col, (title, z) in enumerate(
            [
                (f"True density (n={n})", z_true),
                (f"KDE estimate (n={n})", z_kde),
                (f"NF estimate (n={n})", z_nf),
            ]
        ):
            ax = axes[row, col]
            ax.contourf(x, y, z, levels=20)
            ax.plot(data[:, 0], data[:, 1], "r.", markersize=2, alpha=0.5)
            ax.set_title(title)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(y[0], y[-1])

    fig.tight_layout()
    plt.show()
