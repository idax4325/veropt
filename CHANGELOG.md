# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.3.0] - 2026-04-21

### Added
- **Observation noise support (V1)**: constant per-objective noise std can now be set on any
  `Objective` via `noise_std: dict[str, float]`. The noise is pinned in the GP likelihood,
  automatically selects `qLogNoisyEHVI` for multi-objective problems, and shown as
  uncertainty ellipses (or error bars) on Pareto front plots.
- **Trainable noise bounds on `Objective`**: `train_noise: bool`, `noise_std_min: dict[str, float]`,
  and `noise_std_max: dict[str, float]` are now first-class fields on all `Objective` subclasses.
  All noise configuration lives on the objective in physical units — no internal noise knobs
  are exposed to the user. See `changelog_reports/v1.3.0/noise_settings_refactor.md`.
- **Noise-aware Pareto front**: `get_pareto_optimal_points` now accepts
  `noise_std_per_objective` and applies ε-dominance (Laumanns et al. 2002, IEEE TEC 6(3))
  with ε_j = 1σ margin. `plot_pareto_front` / `plot_pareto_front_grid` pass noise through
  automatically and accept `uncertainty_style='ellipse'` (default) or `'error_bars'`.
- **Noisy Pareto front example**: `examples/example_noisy_pareto_front.py`.
- **Experiment rollback utility**: `veropt.interfaces.rollback.rollback_experiment` rolls
  an experiment back to a previous batch boundary, restoring JSON state and optionally
  renaming or deleting simulation result folders. CLI available via
  `veropt.interfaces.cli`. See `changelog_reports/v1.3.0/rollback_implementation.md`.
- **Schema versioning and auto-migration**: optimiser JSON files are now stamped with
  `schema_version`. Migrations v1→v2→v3→v4 are applied automatically when
  `allow_automatic_json_updates=True` (with backup). The flag can be overridden at the
  call site via `experiment(..., allow_automatic_json_updates=True)` without editing
  the JSON.
- **Normaliser `transform_scale` / `inverse_transform_scale`**: new methods that apply
  only the scale part of normalisation (no mean shift), used for transforming noise std
  into model space.
- **String-based point selection in prediction plots**: `plot_prediction_grid`,
  `plot_prediction_surface`, and `plot_prediction_surface_grid` now accept string
  selectors for `evaluated_point`: `"best"`, `"best {objective_name}"`, `"suggested N"`.
  `evaluated_point` on `plot_prediction_surface` is now optional (defaults to `None`).
- **`allow_automatic_json_updates` exposed to experiment constructors**: pass the flag
  directly to `experiment()` / `experiment_with_new_version()` to trigger a one-off JSON
  migration without editing the file by hand.
- **Fake SLURM batch manager for testing**: `MockBatchManager` in
  `tests/interfaces/test_slurm_experiment.py` enables integration tests of the experiment
  submission loop without a real SLURM queue.
- **`noise_std`, `train_noise`, `noise_std_min`, `noise_std_max` in `ExperimentConfig`**:
  all noise settings are configurable via the experiment config JSON, keyed by objective
  name. No changes to the optimiser settings JSON needed.

### Changed
- **`run_experiment_step_submitted` refactored into three phases** (collect, optimise,
  submit), each with an explicit skip condition. `run_experiment_step_direct` follows the
  same pattern (optimise, run-and-collect). Phase logic is extracted into private helpers
  `_collect_previous_batch`, `_submit_next_batch`, and `_run_and_collect_batch_direct`.
- **Noise architecture**: all user-facing noise configuration (`noise_std`, `train_noise`,
  `noise_std_min`, `noise_std_max`) lives exclusively on `Objective` in physical units.
  The internal `NoiseSettingsInputDict` TypedDict and `NoiseParameters` dataclass have been
  removed. The model layer takes a plain `train_noise: bool` for reconstruction only.
- **`noise` and `noise_lower_bound` removed from model layer**: these were in normalised
  (model) units and not meaningful to users. The constraint floor is now a hardcoded `1e-8`
  constant, never saved to JSON. Existing JSON files are migrated via schema v4.
- **Kernel noise fields removed**: `noise`, `noise_lower_bound`, and `train_noise` have been
  fully removed from individual kernel settings dataclasses. Passing these keys inside
  `kernel_settings` raises immediately. `_set_up_noise_constraints()` in the
  `GPyTorchSingleModel` base class replaces 5 identical per-kernel blocks.
- **Default plot evaluation point**: `choose_plot_point` now defaults to the **best
  evaluated point** (highest weighted objective sum) instead of the first suggested point.
- **`CURRENT_SCHEMA_VERSION = 4`** (was 1 before schema versioning; incremented for each
  structural JSON change in this release).

### Fixed
- **`continue_with_new_version` phantom points**: `run_experiment_step_submitted` was
  registering a next batch in state before the "don't submit on last step" guard, leaving
  `n_evals_per_step` phantom points in state after a completed experiment. New-version
  indices now start at the correct offset.
- **`continue_with_new_version` double-load**: the first step of a new version was
  calling `run_optimisation_step()` with `just_rebuilt=True`, which re-read
  `evaluated_objectives.json` (still holding the last replayed batch) and loaded those
  points into the GP a second time. Fixed by calling `suggest_and_save_candidates()`
  instead — model is already trained, only the suggest+save half is needed.
  See `changelog_reports/v1.3.0/bug_fixes.md`.
- **NumPy 2.4 compatibility** (`TypeError: only 0-dimensional arrays can be converted to
  Python scalars`): two sites fixed.
  `TorchNumpyWrapper.__call__` returned a shape-`[1]` array to `scipy.optimize.dual_annealing`,
  which expects a scalar — fixed with `.detach().item()`, return type updated to `float`.
  `ProximityPunishmentSequentialOptimiser._sample_acq_func` had the same pattern assigning
  to a numpy scalar slot — fixed with `.detach().item()`. (closes issue #22)
- **Noisy multi-objective reload crash** (`UnsupportedError: Models with multiple batch
  dims`): `gather_dicts_to_save` was saving `model_with_data.train_inputs` as a tuple,
  producing shape `[1, n_points, n_vars]` on reload. Fixed by unwrapping the tuple on save.
- **Stale noise state on reload**: `_apply_physical_noise` and `_apply_noise_bounds` are
  now called on the `train=False` reload path so that `objective.noise_std` / bounds are
  always the authoritative source of truth. Previously, the reload path trusted whatever
  `raw_noise` was stored in the JSON. `_apply_physical_noise` also explicitly resets the
  GPyTorch constraint lower bound to `_NOISE_CONSTRAINT_FLOOR` (1e-8) before setting the
  noise value — preventing a stale v3 lower bound (`0.99 * physical_variance`) from
  surviving the reload and being re-saved to the next checkpoint.
  The `_check_noise_desync_on_reload` guard has been removed since it is no longer needed.

## [1.2.0] - 10-12-2025
Major improvements of visual tools and added new kernels.

### Added
- New built-in kernels
- 3d prediction plot
- Ability to run all graphs without normalisation (now default)
- Template for writing a new experiment
- Two visualisation examples
- General improvement to most visual tools
- 

### Changed
- Moved some internal visual methods to their own folders
  - 'visualisation.py' is now where users should go to find visual methods
- Changed naming of public visual methods
- Suggested points are now reset after loading new data instead of 
  after saving suggestions
- Learning rate setting has been fixed and is now
  - a) functional and
  - b) residing in the model optimiser where it belongs
- Model optimiser has been cleaned up and now follows same system as similar objects
- Fixed issue from pydantic with saving nan's to json
- jsons are pretty-printed
- Objective values will not be re-calculated if they're already in exp state
- Fixed minor bug when saving suggested steps

## [1.1.2] - 17-11-2025
Added the ability to use existing run with a new objective

### Added
- New experiment constructor that will use new ability to create 
  new version of existing experiment.

### Changed
- Name and location of optimiser and experiment state jsons

## [1.1.0] - 25-10-2025
Updated interfaces to allow pausing and resuming runs.

### Added
- Ability to resume runs that have been stopped
- Support for optimiser configuration
- Experiment will save optimiser state and reload it

### Changed
- Some internal refactoring on the Experiment class

## [1.0.0] - 15-07-2025
Refactor of the entire project! 

### Added
- New interfaces folder for setting up optimisation problems on e.g. slurm
- It is possible to save the optimiser again, now in a readable, stable json file
- New setting file (also json) where optimiser configuration can be saved
- New constructor functions that can be called instead of creating classes directly
- veropt is now typed and checked by mypy

### Changed
- Internal structure
- Interfaces
- Examples

### Removed
- The GUI is not currently available but will hopefully return in the future

## [0.6.0] - 28-02-2025
### Added
- Changelog :))
- New visualisation tools in plotly
- Test folder!
  - First tests added for normalisation, more will follow in veropt 1.0

### Changed
- Dependencies are updated to newest versions
- Normalisation
  - Should work more correctly and be more robust now
- Sequence optimiser (proximity punish)
  - Now measuring scale globally to remove assumption of acq func range from 0 to a positive number
  - Furthermore checking for multiple disjoint distributions of acquisition function values and (if found) uses std of 
    top distribution
  - All this should ensure correct behaviour and avoid bugs that caused optimisation to 1) choose the same point 
    multiple times or 2) making the punishment dominate the landscape

### Removed
- Temporary:
  - Saver (will come back in later release!)
- Possibly permanent:
  - UCB with noise