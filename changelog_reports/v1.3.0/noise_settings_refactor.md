# Noise Architecture Evolution (Schema v1 → v4)

## Overview

This document traces the evolution of noise configuration through four schema versions
in v1.3.0. The end state is deliberately simple: **all noise configuration lives on
`Objective` in physical units**. Nothing noise-related is exposed to the user below
that layer.

---

## Final architecture (schema v4 — current)

### User-facing API

All noise is set on the `Objective`:

```python
# Fixed noise (exact pinning)
objective = VehicleSafety(noise_std={'VeSa 1': 0.05, 'VeSa 2': 0.1, 'VeSa 3': 0.05})

# Trained noise (GP learns noise level)
objective = VehicleSafety(train_noise=True)

# Trained noise with bounds
objective = VehicleSafety(
    train_noise=True,
    noise_std_min={'VeSa 1': 0.01, 'VeSa 2': 0.01, 'VeSa 3': 0.01},
    noise_std_max={'VeSa 1': 0.5,  'VeSa 2': 0.5,  'VeSa 3': 0.5},
)
```

Invalid combinations raise at construction time with a clear message:
- `noise_std` + `train_noise=True` → error (conflicting)
- `noise_std_min`/`noise_std_max` + `train_noise=False` → error (bounds with no training)

### What changed from v3

| v3 | v4 |
|---|---|
| `NoiseSettingsInputDict` TypedDict in `model.py` | **Removed** |
| `NoiseParameters` dataclass in `model.py` | **Removed** |
| `_noise_settings: NoiseParameters` on `GPyTorchSingleModel` | **Removed** |
| `noise_settings` param in kernel `__init__` and `from_n_variables_and_settings` | Replaced with `train_noise: bool = False` |
| `noise_settings` in `GPytorchModelChoice` constructor TypedDict | **Removed** |
| `noise` and `noise_lower_bound` saved in model JSON state | **Removed** (schema v4 migration drops them) |
| `_apply_physical_noise` modifies constraint lower bound to `0.99 * physical_variance` | Resets constraint to `_NOISE_CONSTRAINT_FLOOR` (1e-8) then sets noise value |
| `_check_noise_desync_on_reload` in `BayesianOptimiser.from_saved_state` | **Removed** — `_apply_physical_noise` / `_apply_noise_bounds` called on reload instead |
| `train=False` reload path skips noise re-application | `_apply_physical_noise` / `_apply_noise_bounds` now called on `train=False` path — `objective.noise_std` is always the source of truth |
| `train_noise` / `noise_std_min` / `noise_std_max` in `ExperimentConfig` | Added |

### JSON structure (v4)

Each `model_N` in the saved JSON now contains only:

```json
{
  "state": {
    "settings": { "lengthscale_lower_bound": 0.1, ... },
    "train_noise": false,
    "state_dict": { ... },
    "train_inputs": [ [...] ],
    "train_targets": [ [...] ]
  }
}
```

`noise` and `noise_lower_bound` fields are gone. On every reload, `_apply_physical_noise`
resets the constraint to `_NOISE_CONSTRAINT_FLOOR` (1e-8) and sets the noise value from
`objective.noise_std` — so no stale values from the JSON can survive.

### Why this is safer

The v3 architecture had noise spread across three places: the objective, the model state
JSON (`noise`, `noise_lower_bound`), and the GPyTorch likelihood state dict (`raw_noise`
calibrated against a `0.99 * physical_variance` constraint lower bound).

On reload, if the normaliser re-fit on wider data, the stale constraint lower bound caused
a spurious `ValueError`. By:
1. removing `noise` and `noise_lower_bound` from the JSON,
2. calling `_apply_physical_noise` on the `train=False` reload path (previously it was only
   called during training), and
3. having `_apply_physical_noise` reset the constraint to `_NOISE_CONSTRAINT_FLOOR` before
   setting the noise value (so `raw_noise` is always calibrated against 1e-8, preventing
   stale constraint lower bounds from surviving a save/reload cycle),

the failure modes are eliminated at the root.

Note: GPyTorch saves `raw_noise_constraint.lower_bound` **inside** `state_dict`, not just
`raw_noise`. A v3 JSON therefore contains a stale `lower_bound ≈ 0.99 * physical_variance`
which `load_state_dict` restores. Without the explicit constraint reset in
`_apply_physical_noise`, that stale value would persist in memory and be re-saved on the
next checkpoint — reproducing a v3-style state_dict. The reset prevents this.

---

## Earlier schema versions (historical context)

### Schema v1 → v2: noise moved out of kernel settings

`noise`, `noise_lower_bound`, and `train_noise` were originally repeated fields inside
each of the 6 kernel `Parameters` dataclasses. They were moved into a new
`NoiseSettingsInputDict` / `NoiseParameters` owned by the `GPyTorchSingleModel` base
class, and `_set_up_noise_constraints()` in the base class replaced 5 identical per-kernel
blocks.

`_migrate_v1_to_v2` moves `{noise, noise_lower_bound, train_noise}` from
`model_N.state.settings` to `model_N.state`.

### Schema v2 → v3: train_inputs tuple unwrap

`gather_dicts_to_save` was saving `model_with_data.train_inputs` as a tuple, producing
shape `[1, n_points, n_vars]` on reload and `batch_shape=[1]`. BoTorch's
`prune_inferior_points_multi_objective` (used inside `qLogNoisyEHVI`) then raised
`UnsupportedError: Models with multiple batch dims`. Fixed by unwrapping the tuple.

`_migrate_v2_to_v3` squeezes the extra batch dimension from saved `train_inputs`.

### Schema v3 → v4: noise architecture cleanup

As described in the **Final architecture** section above.

`_migrate_v3_to_v4` drops `noise` and `noise_lower_bound` from each `model_N.state`.

---

## Migration guide

Run `migrate_json` explicitly, or pass `allow_automatic_json_updates=True` once:

```python
from veropt.interfaces.constructors import experiment

user_experiment = experiment(
    ...,
    allow_automatic_json_updates=True  # migrates v1/v2/v3 → v4 on load, creates .bak
)
```

After migration the JSON is at schema v4 and the flag is no longer needed.
