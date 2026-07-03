# Rollback Implementation

A utility for recovering experiments that crashed mid-optimisation by rolling back
the experiment state, optimiser JSON, and simulation result directories to a previous
batch boundary — then wiring the state so the experiment can be cleanly resumed.

---

## Files

| File | Role |
|------|------|
| `veropt/interfaces/rollback.py` | Core logic |
| `veropt/interfaces/cli.py` | Command-line interface |
| `tests/interfaces/test_rollback.py` | Integration tests |

---

## Resuming after a rollback

The submitted-experiment loop (`run_experiment_step_submitted`) always:
1. Waits for the *previous* batch's jobs → collects results.
2. Calls `run_optimisation_step` → `_load_latest_points` (reads `evaluated_objectives.json`) → suggests next candidates.

After rollback to `target_point`, the state is wired as follows so the loop resumes correctly:

| Component | Value after rollback |
|-----------|----------------------|
| `experimental_state.just_rebuilt` | `True` (skip step 1 — no pending jobs) |
| Optimiser evaluated points | `target_point - n_evals_per_step` |
| `evaluated_objectives.json` | last batch (points `target_point - n_evals` .. `target_point - 1`) |
| `suggested_parameters.json` | same last batch |

On the first resumed step:
- `just_rebuilt` skips wait/collect.
- `_load_latest_points` adds the last batch → optimiser reaches `target_point`.
- Model trains → new candidates are suggested → next batch submitted.

**Special case: `target_point == 0`**
- Optimiser has 0 evaluated points; messenger files are empty.
- `just_rebuilt = False` (the `current_step == 0` check is sufficient).

---

## Public API

### `rollback_experiment`

```python
from veropt.interfaces.rollback import rollback_experiment

rollback_experiment(
    experiment_config_path="path/to/experiment_config.json",  # or ExperimentConfig object
    target_point=4,                          # must be a multiple of n_evaluations_per_step
    simulation_folder_handling="rename",     # "rename" | "delete" | "force_delete"
    allow_automatic_json_updates=None,       # passed to load_optimiser_from_state
)
```

`target_point` must satisfy:
- `target_point >= 0`
- `target_point <= n_points_evaluated`
- `target_point % n_evaluations_per_step == 0`

### CLI

```bash
python -m veropt.interfaces.cli rollback \
    --config path/to/experiment_config.json \
    --target-point 4 \
    [--simulation-folders rename|delete|force_delete]
```

---

## Step-by-step walkthrough of `rollback_experiment`

```
1. Load config → build PathManager
2. Load ExperimentalState + BayesianOptimiser from disk
3. Validate target_point (≥0, ≤evaluated, on batch boundary)
4. Back up state JSON and optimiser JSON with timestamp suffix
5. Handle simulation result directories (rename / delete / force_delete)
6. Patch ExperimentalState   → write atomically
7. Patch optimiser JSON      → write atomically
8. Write messenger files     → write atomically
9. Verify (reload and assert)
```

---

## Internal helpers

### `_truncate_optimiser_dict`

Edits the raw optimiser save dict in-place:

```python
def _truncate_optimiser_dict(
        optimiser_dict: dict,
        n_points_to_keep: int,
        n_evaluations_per_step: int,
) -> None:
    opt = optimiser_dict["optimiser"]

    # Slice evaluated_variables and evaluated_objectives
    for key in ("evaluated_variables", "evaluated_objectives"):
        entry = opt[key]
        if isinstance(entry.get("values"), list):
            entry["values"] = entry["values"][:n_points_to_keep]

    # Keep only the batches that fit within n_points_to_keep
    history: list = opt.get("suggested_points_history", [])
    if history:
        if n_points_to_keep == 0:
            opt["suggested_points_history"] = []
        else:
            n_batches_to_keep = n_points_to_keep // n_evaluations_per_step
            opt["suggested_points_history"] = history[:n_batches_to_keep]

    # Clear stale pending suggestions
    opt["suggested_points"] = {}

    if n_points_to_keep == 0:
        # Reset GP state_dicts so model_has_been_trained = False
        model_dicts = opt["predictor"]["state"]["model"]["state"]["model_dicts"]
        for model_key in model_dicts:
            model_dicts[model_key]["state"]["state_dict"] = {}
            model_dicts[model_key]["state"]["train_inputs"] = None
            model_dicts[model_key]["state"]["train_targets"] = None
        # Clear normalisers so the optimiser starts fully fresh
        opt["normaliser_variables"] = None
        opt["normaliser_objectives"] = None
```

> **Note on normalisers for `n_points > 0`:** Normalisers are left as-is (they may be
> slightly stale) because they are needed by `from_saved_state` when `model_has_been_trained`
> is `True`. They are re-fitted automatically on the first resumed step when
> `renormalise_each_step=True`.

### `_write_last_batch_messenger_files`

Reconstructs `evaluated_objectives.json` and `suggested_parameters.json` from the
last kept batch stored in the `ExperimentalState`:

```python
last_batch_start = target_point - n_evaluations_per_step
last_batch_indices = range(last_batch_start, target_point)

for point_no in last_batch_indices:
    point = state.points[point_no]
    for obj_name in objective_names:
        evaluated_objectives[obj_name].append(point.objective_values[obj_name])
    for param_name in parameter_names:
        suggested_parameters[param_name].append(point.parameters[param_name])
```

### `_backup_file`

Creates a timestamped copy before any mutation:

```python
backup_path = path.replace(".json", f"_backup_{timestamp}.json")
shutil.copy2(path, tmp_path)
os.replace(tmp_path, backup_path)
```

### `_atomic_write_json`

All writes go through a temp-file + `os.replace` to avoid leaving corrupt JSON on disk:

```python
with tempfile.NamedTemporaryFile(mode="w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
    json.dump(data, tmp, indent=2, default=_json_default)
    tmp_path = tmp.name
os.replace(tmp_path, path)
```

### `_handle_simulation_directories`

```python
SimulationFolderHandling = Literal["rename", "delete", "force_delete"]
```

| Mode | Behaviour |
|------|-----------|
| `"rename"` | Appends `_rolled_back` suffix. Default. |
| `"delete"` | Prompts for confirmation, then deletes. |
| `"force_delete"` | Deletes without prompting. |

---

## Tests (`tests/interfaces/test_rollback.py`)

All tests use `FakeSubmitBatchManager` and `MockSimulationRunner` — no real SLURM
queue is needed.

### Shared fixtures

```python
def _make_experiment_config(tmp_dir: str) -> ExperimentConfig: ...
def _make_optimiser_config() -> dict: ...      # n_initial=4, n_bayesian=4, n_evals_per_step=2
def _make_simulation_runner(tmp_dir) -> MockSimulationRunner: ...
def _make_result_processor() -> MockResultProcessor: ...  # fixed_objective=False
```

### Test inventory

| Test | What it checks |
|------|---------------|
| `test_rollback_state_matches_earlier_snapshot` | Run 4 steps, roll back to step 2. State/optimiser point counts match snapshot. Extra points gone. |
| `test_rollback_to_zero` | Roll back to 0. Empty state, `just_rebuilt=False`, optimiser has 0 points. |
| `test_rollback_creates_backups` | At least 2 timestamped `_backup_*.json` files exist after rollback. |
| `test_rollback_renames_simulation_directories` | Kept dirs still exist; rolled-back dirs have `_rolled_back` suffix. |
| `test_rollback_force_deletes_simulation_directories` | Rolled-back dirs are gone entirely. |
| `test_rollback_invalid_target_point_not_on_batch_boundary` | `target_point=1` with `n_evals=2` raises `AssertionError` matching "batch boundary". |
| `test_rollback_forward_raises` | `target_point=1000` raises `AssertionError`. |
| `test_rollback_sets_just_rebuilt_flag` | After rollback to non-zero target, `state.just_rebuilt is True`. |
| `test_rollback_messenger_files_contain_last_batch` | `evaluated_objectives.json` has exactly `n_evals_per_step` entries. |
| `test_experiment_can_continue_after_rollback` | After rollback, `Experiment.continue_if_possible` + one step runs without error and submits new points. |

### Example: core round-trip test

```python
def test_rollback_state_matches_earlier_snapshot() -> None:
    n_evals_per_step = 2
    target_point = 4  # 2 steps × 2 evals

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=_make_experiment_config(tmp_dir),
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=2)   # → 4 evaluated
        _run_n_steps(experiment, n_steps=2)   # → 8 evaluated

        rollback_experiment(experiment_config_path, target_point=4, simulation_folder_handling="rename")

        restored_state = ExperimentalState.load(path_manager.experimental_state_json)
        assert restored_state.next_point == 4
        assert len(restored_state.points) == 4

        restored_optimiser = load_optimiser_from_state(path_manager.optimiser_state_json)
        assert restored_optimiser.n_points_evaluated == 2  # last batch in messenger
```

### Example: resume after rollback

```python
def test_experiment_can_continue_after_rollback() -> None:
    ...
    rollback_experiment(experiment_config_path, target_point=4, simulation_folder_handling="rename")

    resumed_experiment = Experiment.continue_if_possible(
        simulation_runner=simulation_runner,
        result_processor=result_processor,
        experiment_config=experiment_config,
        optimiser_config=_make_optimiser_config(),
        batch_manager_class=FakeSubmitBatchManager,
    )
    resumed_experiment.run_experiment_step()

    assert resumed_experiment.n_points_submitted > 4
```

