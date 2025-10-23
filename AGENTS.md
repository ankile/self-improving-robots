# Repository Guidelines

## Project Structure & Module Organization
Core package code lives under `sir/`, organized by task domain: `teleoperation/` for SpaceMouse entry points, `training/` for policy learning utilities, `envs/` for simulation wrappers, and `tools/` for data and logging helpers. Tests mirror runtime modules under `sir/tests/`. Assets (robot meshes, calibration files) sit in `sir/assets/`, while experiment artifacts, recorded demos, and videos persist in `data/`, `checkpoints/`, and `videos/`. High-level docs and quick-reference guides are in `docs/` and the repository root.

## Build, Test, and Development Commands
Install dependencies in editable mode via `pip install -e .[dev]` to pull in Black, Ruff, and pytest. Rapid teleop testing uses `python -m sir.teleoperation -e PegInsertionSide-v1`; swap `-e` for other ManiSkill task names. Hardware-free smoke tests live in `python -m sir.tests.test_maniskill_env` and `python -m sir.tests.test_spacemouse` (mocks SpaceMouse input). Training scripts for SLURM clusters are launched with `scripts/slurm_submit_training.sh`; update job params before dispatch.

## Coding Style & Naming Conventions
Python code targets 3.11+ with 100-character lines. Run `ruff check sir` and `black sir` before committing; both respect the configuration in `pyproject.toml`. Use descriptive module and function names tied to the robot skill (e.g., `collect_corrections`, `apply_dagger_update`). Keep files snake_case, classes PascalCase, and constants upper snake case. Prefer type hints on public APIs and document tricky controller math inline.

## Testing Guidelines
Follow pytest discovery: place new suites in `sir/tests/` and name files `test_*.py`. Co-locate fixtures with modules under test. Integration demos that require hardware should be guarded behind explicit flags so CI-only runs stay virtual. Target parity with teleoperation flows: every new control path should add a simulator regression test plus a mocked SpaceMouse input case.

## Commit & Pull Request Guidelines
Recent history favors imperative, single-line commit subjects (`Fix maniskill teleop running again`). Bundle related changes and reference issues with `[#123]` where relevant. Pull requests should outline motivation, enumerate behavioral changes, list test commands run, and attach short videos or screenshots when altering teleoperation UX. Flag breaking API updates in the title and request review from teleop and training maintainers.
