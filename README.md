# Scenario exposure

Library for calculating probability density function using various ways.
There is also functionality to sample from the distributions.

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
