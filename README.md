<p align="center">
  <img src="logo.png" alt="Scenario exposure logo" width="400" />
</p>

# Scenario exposure

Library for calculating probability density function using various ways.
There is also functionality to sample from the distributions.

## Example

An example script demonstrating how to use the `KDEModel` and `NFModel` classes is
provided in [`example_kde_nf_comparison.py`](example_kde_nf_comparison.py). It samples
points (100, 500, and 2000) from a standard 2D distribution and estimates the density
using both Kernel Density Estimation (KDE) and Normalizing Flows (NF), plotting the
results for comparison. Run it with:

```bash
uv run example_kde_nf_comparison.py
```

## Contributing

- **Run linter:**  

  ```bash
  uv run ruff format . --check  # Remove --check flag to reformat files.
  uv run ruff check .

- **Run typechecking:**

  ```bash
  uv run pyright
  uv run mypy src tests
  ```

- **Run tests:**  

  ```bash
  uv run pytest -s --cov=scenario_exposure --cov-report=term-missing --typeguard-packages=scenario_exposure
  ```
