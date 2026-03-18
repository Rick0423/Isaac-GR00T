# Repository Guidelines

## Project Structure & Module Organization
`gr00t/` is the main Python package. Keep configuration trees in `gr00t/configs/`, data and collators in `gr00t/data/`, model code in `gr00t/model/`, training launchers in `gr00t/experiment/`, evaluation entry points in `gr00t/eval/`, and serving code in `gr00t/policy/`. Mirror source paths under `tests/gr00t/` when adding tests. Use `examples/` and `getting_started/` for runnable guides, `scripts/` for deployment utilities, `docker/` for container setup, and avoid casual edits in `external_dependencies/`.

## Build, Test, and Development Commands
Use Python 3.10 and `uv`.

- `uv sync --python 3.10` creates or refreshes the pinned environment from `uv.lock`.
- `uv pip install -e .[dev]` installs the package in editable mode with `pytest`, `ruff`, and `pre-commit`.
- `ruff format .` formats the codebase; `ruff check --fix .` applies lint and import-order fixes.
- `uv run pytest -v tests -m "not gpu"` runs the CPU-safe test suite.
- `uv run pytest -v tests/gr00t/model/test_variable_image_size.py` is the pattern for targeted regression runs.
- `uv run python -m build` builds the wheel and sdist when packaging changes.

## Coding Style & Naming Conventions
Follow `pyproject.toml`: 4-space indentation, double quotes, and a 100-character line limit. Use `snake_case` for modules, functions, CLI flags, and test files, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Let Ruff manage formatting and import sorting. Keep embodiment-specific logic close to its package instead of introducing broad helpers.

## Testing Guidelines
Tests use `pytest` with `--import-mode=importlib`; GPU-only coverage should be marked with `@pytest.mark.gpu`. No fixed coverage percentage is published, but `CONTRIBUTING.md` expects most changes to include unit tests. Name files `test_<behavior>.py`, keep reusable sample assets in `tests/fixtures/`, and add a regression test for every bug fix. Run focused tests first, then the wider CPU-safe suite.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects, sometimes with a scope prefix or issue reference, such as `CI: Migrate...` or `fix issue #541 ...`. Keep each commit focused on one logical change. Pull requests should follow `.github/pull_request_template.md`: include `Fixes #<id>`, summarize the change, update relevant docstrings or docs, and confirm appropriate tests were added for bug fixes or new features. Include logs or benchmark notes when training, evaluation, or deployment behavior changes.

## Assets & Configuration Tips
Initialize submodules with `git submodule update --init --recursive` after cloning. Do not commit checkpoints, datasets, `.venv*`, `wandb/`, or generated outputs under `demo_data/` or `models/`. Keep machine-specific paths and secrets out of tracked files.
