# Bug Fixes

---

## NumPy 2.4 compatibility — `ProximityPunishmentSequentialOptimiser`

**File:** `veropt/optimiser/acquisition_optimiser.py`, line ~478
**Closes:** [issue #22](https://github.com/aster-stoustrup/veropt/issues/22)

### Symptom
Running with NumPy ≥ 2.4 raised:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

in `ProximityPunishmentSequentialOptimiser._sample_acq_func()`.

### Root cause
NumPy 2.4 tightened the rule that assigning to a scalar slot of a numpy array
(`samples[coord_ind] = value`) requires `value` to be 0-dimensional.
`sample.detach().numpy()` returned a shape-`[1]` array — no longer accepted.

### Fix
```python
# Before
samples[coord_ind] = sample.detach().numpy()

# After
samples[coord_ind] = sample.detach().item()
```

`.item()` extracts a plain Python float, which is always a valid assignment target
in numpy regardless of version. It also implicitly detaches from the computation
graph, so the memory-leak protection is preserved.

---

## `continue_with_new_version` — phantom points and double-load

**Files:** `veropt/interfaces/experiment.py`, `veropt/optimiser/optimiser.py`

Two related bugs affecting `continue_with_new_version` when the old experiment had
been run to completion via the submitted (SLURM) workflow.

### Bug 1 — Phantom points in state after completion

**Symptom:** after `continue_with_new_version`, `state.n_points` was 2 larger than
`n_points_evaluated`, and new-version point indices started at the wrong offset, leaving
a gap in the state.

**Root cause:** `run_experiment_step_submitted` always called `get_parameters_from_optimiser()`
(which registers new `Point` objects in `state`) before the `if not last_step` guard that
decided whether to actually submit them.  On the final step the registration happened but
the submit was skipped, leaving `n_evals_per_step` phantom points in state with no job,
no result and no objective values.

**Fix:** `is_last_step` is now computed before `_submit_next_batch()` (which contains
`get_parameters_from_optimiser()`), so no points are ever registered for a batch that
will not be submitted.

### Bug 2 — Double-load of last replayed batch

**Symptom:** after `continue_with_new_version` + 2 new steps, `n_points_evaluated`
overshot by `n_evals_per_step`, and point indices were non-contiguous.

**Root cause:** the replay loop in `continue_with_new_version` ends by writing the last
replayed batch to `evaluated_objectives.json`.  The first call to
`run_experiment_step_submitted` on the new version (with `just_rebuilt=True`) skips the
wait/collect phase but still called `run_optimisation_step()`, which starts with
`_load_latest_points()` — re-reading that file and adding the last batch to the GP a
second time.

**Fix:** when `just_rebuilt=True`, the optimise phase calls
`suggest_and_save_candidates()` instead of `run_optimisation_step()`.  The model was
already trained by `continue_with_new_version`'s `train_model()` call; this just runs
the suggest+save half without the load, cleanly producing the first new batch of
candidates.

### Refactor

Both fixes motivated restructuring `run_experiment_step_submitted` into three named
phases — collect, optimise, submit — each with an explicit skip condition:

```python
has_previous_batch = self.current_step > 0 and not self.state.just_rebuilt
if has_previous_batch:
    self._collect_previous_batch()

is_last_step = self.current_step == self.n_total_steps - 1

if not self.state.just_rebuilt:
    self.optimiser.run_optimisation_step()
else:
    self.optimiser.suggest_and_save_candidates()

self._save_optimiser()

if not is_last_step:
    self._submit_next_batch()
```

`run_experiment_step_direct` received the same treatment (optimise + run-and-collect).

---

## Noisy multi-objective reload crash — `batch_shape=[1]`

**File:** `veropt/optimiser/model.py`, `veropt/optimiser/prediction.py`
**Full details:** see `noise_settings_refactor.md` → *Schema v3* section.

### Symptom
Reloading a saved noisy multi-objective optimiser raised:

```
botorch.exceptions.errors.UnsupportedError: Models with multiple batch dims are
currently unsupported by `prune_inferior_points_multi_objective`.
```

The optimiser ran fine during training — the crash only appeared on reload.

### Root cause
`gather_dicts_to_save` saved `model_with_data.train_inputs` (a gpytorch tuple),
which on reload produced tensors with shape `[1, n_points, n_vars]` instead of
`[n_points, n_vars]`, giving `batch_shape = [1]`.

### Fix
- Unwrap the tuple on save: `model_with_data.train_inputs[0]` (→ correct shape).
- Also save the live noise constraint lower bound so the pinned noise is correctly
  restored before `load_state_dict`.
- Removed the `_apply_physical_noise` call on the reload path — the state dict
  already encodes the correct `raw_noise`.
- Added schema v3 migration (`_migrate_v2_to_v3`) to fix existing JSON files.
