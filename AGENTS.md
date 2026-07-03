# AGENTS.md - Guide for AI Coding Agents

## Project Overview

**veropt** is a user-friendly Bayesian Optimization library designed for expensive optimization problems. It's built around PyTorch, GPyTorch, and BoTorch for Gaussian Process modeling and acquisition function optimization.

### Core Workflow
The library follows this optimization loop:
1. **Initial Phase**: Generate and evaluate initial random points
2. **Bayesian Phase**: Build GP surrogate model → optimize acquisition function → evaluate suggested points
3. **Visualization**: Plot predictions, acquisition functions, and evaluated points with Plotly

**Key Maximization Convention**: veropt always maximizes objectives. Minimization requires negating objectives.

## Architecture Overview

### Three Main Modules

#### `veropt/optimiser/` - Core Optimization Engine
- **`optimiser.py`**: `BayesianOptimiser` class - main entry point managing the optimization loop
- **`constructors.py`**: `bayesian_optimiser()` factory function with TypedDict-based configuration
- **`objective.py`**: Abstract `Objective` base class with `CallableObjective` and `InterfaceObjective` variants
- **`model.py`**: GPyTorch-based GP models (`GPyTorchFullModel`, `GPyTorchSingleModel`)
- **`acquisition.py`**: Acquisition functions (qLogEHVI, UCB) using BoTorch
- **`acquisition_optimiser.py`**: Dual annealing optimizer for finding next points
- **`prediction.py`**: `BotorchPredictor` wrapper for posterior sampling
- **`normalisation.py`**: Input/output normalization (StandardNormaliser, RobustNormaliser)
- **`saver_loader_utility.py`**: Serialization via `SavableClass` interface - all core objects implement this

#### `veropt/interfaces/` - External Simulation Integration
- **`experiment.py`**: `Experiment` class - recommended high-level interface orchestrating simulations with batch management and state persistence
- **`simulation.py`**: Abstract `SimulationRunner` for executing objective functions (local or SLURM-based)
- **`batch_manager.py`**: Submit jobs locally or to SLURM clusters
- **`result_processing.py`**: `ResultProcessor` to extract objectives from simulation outputs
- **`experiment_utility.py`**: State management (`ExperimentalState`) and configuration (`ExperimentConfig`)

#### `veropt/graphical/` - Interactive Visualization
- **`visualisation.py`**: Main entry point - creates Plotly dashboards
- **`_model_visualisation.py`**: Plots GP predictions with uncertainty
- **`_pareto_front.py`**: Multi-objective Pareto front visualization

### Data Flow

```
Objective (bounds, n_variables, n_objectives)
    ↓
Normaliser (StandardNormaliser/RobustNormaliser)
    ↓
Predictor (GP surrogate model)
    ↓
AcquisitionFunction (qLogEHVI/UCB)
    ↓
AcquisitionOptimiser (Dual Annealing)
    ↓
Experiment/Simulator (LocalSimulation/SlurmSimulation)
    ↓
ResultProcessor (extract objectives)
    ↓
BayesianOptimiser.run_optimisation_step()
```

## Critical Developer Patterns

### 0. Code Design Priorities
Follow this hierarchy when generating code - **always start minimal**:
1. **KISS (Keep It Simple, Stupid)** - Primary goal. Start with minimal implementation.
2. **YAGNI (You Aren't Gonna Need It)** - Avoid speculation. If edge case not demonstrated, add `assert` instead.
3. **SRP (Single Responsibility Principle)** - One clear purpose per function/class
4. **Rest of SOLID principles**

Example: Don't build a general strategy pattern if a simple `if/else` works. Add assertions for unsupported cases rather than over-engineering.

### 1. Natural Naming Conventions
- **Loops**: Use descriptive names, not single letters. Examples:
  - `for point_number in range(n_points):` not `for i in range(n_points):`
  - `for objective_index, obj_value in enumerate(objectives):` not `for i, val in enumerate(objectives):`
  - `for step in range(max_steps):` not `for i in range(max_steps):`
- **Variables**: Name for clarity even if verbose. `evaluated_objectives_normalised` is better than `evals`.
- **Boolean flags**: Prefix with verb (is_, has_, can_). Example: `should_normalize_inputs` not `normalize`.

### 2. Formatting
- **Bug fix comments**: When fixing a bug, document the fix in `CHANGELOG.md`, not as a long comment in the code. Short inline comments (one line) are acceptable only when the code would otherwise be non-obvious.
- **`type: ignore` usage**: Only use `# type: ignore` when fixing the error properly would substantially hurt code simplicity (e.g. fighting untyped third-party stubs). Every `type: ignore` must be accompanied by a short inline comment explaining why it is necessary. Example: `# type: ignore[assignment]  # os.environ stubs don't accept dict[str, Any] from json`
- **Line length**: keep lines within 120 characters (PyCharm right-margin guide).
- **Multi-line calls and dicts**: when a call or dict literal must be split across lines, put *every* argument/key on its own line — never mix some arguments on the opening line and others below. Either the whole call fits on one line, or each argument gets its own line.

```python
# No — mixed style
marker=dict(symbol='circle-open', size=11, color=src_colours,
            line=dict(width=2, color=src_colours)),

# Yes — fully expanded
marker=dict(
    symbol='circle-open',
    size=11,
    color=src_colours,
    line=dict(
        width=2,
        color=src_colours,
    ),
),
```

### 3. Configuration via TypedDict
The library uses `TypedDict` for flexible, validated configuration dictionaries rather than explicit arguments:

```python
optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective,
    model={'training_settings': {'max_iter': 50}},  # TypedDict
    acquisition={'parameters': {'beta': 0.2}},       # TypedDict
    acquisition_optimiser={'optimiser': 'dual_annealing'}
)
```

See `optimiser_utility.py` for `OptimiserSettingsInputDict` definition.

### 3. Savable/Loadable Architecture
All core classes inherit from `SavableClass` (abstract base in `saver_loader_utility.py`):
- Implement `gather_dicts_to_save()` to return serializable dict
- Implement `from_saved_state(saved_state: dict)` for deserialization
- For dataclasses, inherit from `SavableDataClass` instead - automatically implements both methods via `asdict()`
- This enables checkpointing: `save_to_json(optimiser, path)` / `load_optimiser_from_state(path)`

### 4. Normalization as Wrapper
Normalisation is transparent - objectives/variables can be in real or normalized space:
- Stored in `_normaliser_variables` and `_normaliser_objectives` (optional)
- Properties like `bounds_normalised`, `evaluated_variables_normalised` cache normalized versions
- Always work in real units internally, normalize only when needed by GP

### 5. Torch as Default Numerics
- `torch.set_default_dtype(torch.float64)` set at module load
- All numeric operations use PyTorch tensors
- `numpy` compatibility handled explicitly (see `check_incoming_objective_dimensions_fix_1d`)

### 6. Decorator Pattern for Input Validation
Functions use `@_check_input_dimensions` decorator (in `acquisition.py`, `model.py`) to:
- Enforce consistent variable/objective dimensions
- Support flexible positional/keyword arguments via `**kwargs`
- See `utility.py` for `enforce_amount_of_positional_arguments`

### 7. Multi-Objective & Visualization Reference Points
- Single-objective: `n_objectives=1`
- Multi-objective: Nadir point automatically generated via `get_nadir_point()` for qLogEHVI acquisition
- **BoTorch Reference Point**: Mathematical entity used internally in acquisition function calculations. Do not confuse with veropt's reference points.
- **veropt Reference Points** (visualization only): User-supplied points for comparison in plots - **distinct from BoTorch's reference point**
  - Contains both `variable_values` and `objective_values` (stored as `torch.Tensor`)
  - Interfaces aim to support automatic evaluation of user variable values to extract objective values for visualization
  - Cannot be auto-created; requires explicit user input

## Testing & Validation

### Run Tests
```bash
cd /lustre/hpc/ocean/aster07/PycharmProjects/veropt
python local_workflows/tests.py
# or directly: pytest
```

### Type Checking
```bash
python local_workflows/type_checking.py
# or: mypy veropt tests examples
```

### Linting
```bash
python local_workflows/linting.py
# or: flake8 . (ignores E402 - see setup.cfg)
```

**Type Checking Rules** (`mypy.ini`):
- `disallow_untyped_defs = True` - strict typing enforced
- `follow_untyped_imports = False` - external libs may be untyped
- `ignore_missing_imports = True` - for torch/gpytorch stubs

### Test Structure
- Test files match source structure: `tests/test_*.py` for `veropt/*.py`
- Use practice objectives from `veropt.optimiser.practice_objectives` for testing: `Hartmann` (3/4/6-d), `VehicleSafety` (5-d, multi-objective), `DTLZ1` (configurable)
- Example: `test_run_optimisation_step()` in `test_optimiser.py` runs 5 steps with reduced iterations

## Examples & Interfaces

### Quick Start
- `examples/example_single_objective.py` - Basic single-objective optimization
- `examples/example_multi_objective.py` - Multi-objective with Pareto front
- `examples/example_visualisation.py` - Interactive dashboard

### External Simulations
- `examples/interfaces/example_local_veros_experiment.py` - Local VEROS ocean model
- `examples/interfaces/example_slurm_veros_simulation.py` - VEROS on SLURM cluster
- Config files in `examples/` show JSON configuration patterns

## Key Files for Different Tasks

| Task | Files |
|------|-------|
| Add acquisition function | `optimiser/acquisition.py`, `optimiser/constructors.py` |
| Add surrogate model | `optimiser/model.py`, `optimiser/prediction.py` |
| Add normalisation | `optimiser/normalisation.py`, `optimiser/optimiser.py` |
| Add simulator interface | `interfaces/simulation.py`, `interfaces/batch_manager.py` |
| Visualization features | `graphical/visualisation.py`, `graphical/_*.py` |
| Configuration schema | `optimiser/constructors.py` (TypedDict definitions) |

## Important Notes for AI Agents

1. **Float64 Requirement**: PyTorch defaults to float32. Always preserve `torch.set_default_dtype(torch.float64)` in `__init__.py`.

2. **Flake8 Exception**: E402 (module import not at top) is ignored in `setup.cfg` because torch dtype must be set before importing submodules.

3. **NaN Handling**: Multi-objective experiments mask NaN objectives with `_mask_nans()` in `experiment.py` - edge case for robust simulations.

4. **Subclass Registry Pattern**: `get_all_subclasses()` in `saver_loader_utility.py` enables polymorphic deserialization by class name - used for all acquisition functions, models, normalizers.

5. **Experiment State Machine**: `ExperimentalState` in `experiment_utility.py` tracks `next_point` index, pending simulations, and results - critical for resuming interrupted optimizations.

6. **Default Settings**: `veropt/optimiser/default_settings.json` contains hardcoded defaults - modify via TypedDict overrides in constructor, not by editing JSON directly.

7. **Pytest + Python 3.13**: Requires `python >3.13` per `setup.py`. Codebase uses modern syntax (PEP 695 type aliases with `type Foo = ...`).

8. **JSON Schema Versioning**: Saved optimiser JSON files carry a `schema_version` integer. The current version is defined as `CURRENT_SCHEMA_VERSION` in `optimiser_saver_loader.py`. When making changes that alter the JSON structure, increment the version and add a `_migrate_vX_to_vY` function. Migration runs automatically when `allow_automatic_json_updates=True` (creates a `.bak` before writing).

9. **API Surface Audit on Ownership-Shifting Refactors**: When a refactor moves where a
   concept *lives* (e.g. noise configuration moving from kernel settings to `objective.noise_std`),
   explicitly audit whether the old location's fields are still meaningful. Fields that existed
   to approximate what the new mechanism now provides properly should be removed or deprecated
   **in the same PR** — not left as dead/misleading knobs.

   Checklist when shifting concept ownership:
   - Does each old field still have a unique, clearly-defined purpose in the new architecture?
   - Is it expressed in units the user can actually reason about (e.g. physical, not normalised)?
   - Would a user who doesn't know the internal history be confused by seeing it?
   - If answers are no/yes: remove it, or add an assert that raises if set to a non-default value.

   Example: `noise` and `noise_lower_bound` in `NoiseSettingsInputDict` were valid when noise
   lived entirely in kernel config. After V1 noise added `noise_std` on the objective (physical
   units), these became dead/misleading and should have been removed in the same PR.

## Changelog Reports

For significant changes, detailed implementation notes live in `changelog_reports/<version>/`.
Each version folder contains one markdown per feature or fix area, plus a `README.md` index.

**When to create a report**: for changes that are non-trivial to understand from the
code and commit message alone. Good candidates:
- New features that touch multiple files and have a design rationale worth explaining
- Refactors where the "why" and the before/after structure are not obvious
- Bug fixes with a non-trivial root-cause story

**When NOT to create a report**: small, self-contained changes are better described
inline in `CHANGELOG.md` only. Examples: a single-function API extension, a one-line
bug fix, a parameter rename, adding a test for existing behaviour.

**Keeping reports accurate is critical.** A misleading report is worse than no report.
Rules:
- Update the relevant report immediately whenever the implementation changes during a branch.
- If a design decision is reversed (e.g. a reload path is changed), edit the report in the
  same commit — do not leave the old description in place.
- Mark anything speculative or outdated clearly, or delete it.
- Reports live in `changelog_reports/`, **not** in `thinking_files/` (which holds planning
  and design notes that are not expected to stay current).

**Structure:**
```
changelog_reports/
└── v1.3.0/
    ├── README.md                    ← index table
    ├── noise_v1_implementation.md   ← one file per major topic
    ├── noise_settings_refactor.md
    ├── rollback_implementation.md
    ├── visualisation_improvements.md
    └── bug_fixes.md
```

`CHANGELOG.md` (root) contains the concise per-version summary and links to the relevant
reports for anyone who wants deeper detail.

