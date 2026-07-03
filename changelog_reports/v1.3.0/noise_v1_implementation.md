# V1 Noise Feature — Implementation Reference

Documents the code changes introduced for V1 observation noise support.
See `noise_implementation_plan.md` for the full design rationale.

---

## Step 1 — `normalisation.py`: `transform_scale` / `inverse_transform_scale`

### What changed
Two abstract methods added to `Normaliser` and concrete implementations in
`NormaliserZeroMeanUnitVariance`.

### Why
`transform(x)` subtracts the mean — wrong for magnitudes like noise std.
`transform_scale` applies only the `/sqrt(var)` factor (no mean shift).

### Code (lines ~24–51 in `normalisation.py`)
```python
# Abstract methods added to Normaliser:
@abc.abstractmethod
def transform_scale(self, tensor: torch.Tensor) -> torch.Tensor:
    """Apply only the scale part of the normalisation (no mean shift)."""
    pass

@abc.abstractmethod
def inverse_transform_scale(self, tensor: torch.Tensor) -> torch.Tensor:
    """Undo only the scale part of the normalisation."""
    pass

# Concrete in NormaliserZeroMeanUnitVariance:
def transform_scale(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.sqrt(self.variances)

def inverse_transform_scale(self, tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.sqrt(self.variances)
```

---

## Step 2 — `objective.py` and `practice_objectives.py`: `noise_std` field

### What changed
- `Objective.__init__` gained `noise_std: Optional[dict[str, float]] = None`
- `gather_dicts_to_save()` includes `noise_std`
- All three practice objectives (`Hartmann`, `VehicleSafety`, `DTLZ1`) now accept and pass
  through `noise_std`, and their `from_saved_state` reads `saved_state.get('noise_std', None)`
  for backward compatibility with old JSON files

### Why
Physical noise belongs on the `Objective` (it describes the problem, not the model settings).
The dict is keyed by `objective_names` so it naturally stays aligned with the objective.

### Code — `objective.py` (lines ~15–62)
```python
class Objective(SavableClass, metaclass=abc.ABCMeta):

    def __init__(
            self,
            ...
            noise_std: Optional[dict[str, float]] = None
    ):
        ...
        if noise_std is not None:
            assert set(noise_std.keys()) == set(objective_names), (
                f"noise_std keys must match objective_names."
            )
        self.noise_std = noise_std

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                ...,
                'noise_std': self.noise_std,   # ← new
            }
        }
```

### Code — `practice_objectives.py` (example, `Hartmann`)
```python
class Hartmann(BotorchPracticeObjective):
    def __init__(self, n_variables: Literal[3, 4, 6], noise_std: Optional[dict[str, float]] = None):
        ...
        super().__init__(..., noise_std=noise_std)

    @classmethod
    def from_saved_state(cls, saved_state: dict) -> Self:
        return cls(
            n_variables=saved_state['n_variables'],
            noise_std=saved_state.get('noise_std', None)   # ← backward-compat
        )
```

---

## Step 3 — `optimiser.py`: conflict checks, properties, `_update_predictor` wiring

### What changed
Three additions to `BayesianOptimiser`:

1. `_check_noise_configuration()` — called at end of `__init__`
2. `_noise_std_tensor` property — converts `noise_std` dict to tensor
3. `_noise_std_in_model_space` property — applies normalisation if available
4. `_update_predictor()` passes `noise_std_in_model_space` to predictor

### Code
```python
# End of __init__:
self._verify_set_up()
self._set_up_settings()
self._check_noise_configuration()   # ← new

# Conflict check (raises if train_noise=True with noise_std set):
def _check_noise_configuration(self) -> None:
    if self.objective.noise_std is None:
        return
    if not hasattr(self.predictor, 'model') or not hasattr(self.predictor.model, '_model_list'):
        return
    for objective_name, single_model in zip(
        self.objective.objective_names,
        self.predictor.model._model_list
    ):
        if single_model._noise_settings.train_noise:
            raise ValueError(
                f"train_noise=True for the kernel of objective '{objective_name}', but "
                f"noise_std is set on the objective. ..."
            )

# Properties (end of class):
@property
def _noise_std_tensor(self) -> Optional[torch.Tensor]:
    if self.objective.noise_std is None:
        return None
    return torch.tensor(
        [self.objective.noise_std[name] for name in self.objective.objective_names]
    )

@property
def _noise_std_in_model_space(self) -> Optional[torch.Tensor]:
    """Returns normalised noise when normaliser is fitted, physical units otherwise.
    None when noise_std is not set.  'Physical units' is also correct when normalisation
    is disabled — in that case the GP trains on physical-unit data."""
    noise_std_tensor = self._noise_std_tensor
    if noise_std_tensor is None:
        return None
    if self._normaliser_objectives is None:
        return noise_std_tensor
    return self._normaliser_objectives.transform_scale(noise_std_tensor)

# _update_predictor now passes noise:
def _update_predictor(self, train: bool = True) -> None:
    ...
    self.predictor.update_with_new_data(
        variable_values=...,
        objective_values=...,
        train=train,
        noise_std_in_model_space=self._noise_std_in_model_space   # ← new
    )
```

### Desync protection
`_noise_std_in_model_space` is recomputed every time `_update_predictor` is called,
including after renormalisation (`renormalise_each_step=True`). On JSON reload,
`_update_predictor(train=False)` applies the physical noise to the loaded model,
overriding any tampered state_dict value.

---

## Step 4 — `model.py`: `train_model` + `_apply_physical_noise`

### What changed
- `GPyTorchFullModel.train_model` gains `noise_std_in_model_space: Optional[torch.Tensor] = None`
- After `initialise_model()`, calls `_apply_physical_noise()` if noise is provided
- New method `_apply_physical_noise()` validates and pins noise

### Important ordering note
gpytorch derives `raw_noise` via `inverse_transform(noise - lower_bound)`. The constraint
must be registered **before** the noise value is set; otherwise changing the constraint
after setting the value shifts the effective noise. `_apply_physical_noise` always does:
```python
single_model.set_noise_constraint(lower_bound=physical_variance * 0.99)  # 1. constraint first
single_model.set_noise(physical_variance)                                  # 2. value second
```

### Code (lines ~780–825 in `model.py`)
```python
@check_variable_objective_values_matching
@_check_input_dimensions
def train_model(self, *, variable_values, objective_values,
                noise_std_in_model_space: Optional[torch.Tensor] = None) -> None:
    self.initialise_model(variable_values=variable_values, objective_values=objective_values)
    if noise_std_in_model_space is not None:
        self._apply_physical_noise(noise_std_in_model_space=noise_std_in_model_space)
    self._set_mode_train()
    ...

def _apply_physical_noise(self, noise_std_in_model_space: torch.Tensor) -> None:
    for objective_index, single_model in enumerate(self._model_list):
        if single_model.model_with_data is None:
            continue
        physical_variance = float(noise_std_in_model_space[objective_index] ** 2)
        lower_bound = float(single_model.likelihood.noise_covar.raw_noise_constraint.lower_bound)
        if physical_variance < lower_bound:
            raise ValueError(f"Physical noise variance {physical_variance:.2e} ...")
        single_model.set_noise_constraint(lower_bound=physical_variance * 0.99)
        single_model.set_noise(physical_variance)
```

---

## Step 5 — `prediction.py`: noise threading in `update_with_new_data`

### What changed
- `Predictor.update_with_new_data` (abstract) gains `noise_std_in_model_space` param
- `BotorchPredictor.update_with_new_data` forwards it to `train_model` on `train=True`,
  and calls `_apply_physical_noise` directly on `train=False` (reload path)

### Code (lines ~299–326 in `prediction.py`)
```python
@check_variable_objective_values_matching
@_check_input_dimensions
def update_with_new_data(self, *, variable_values, objective_values,
                          train: bool = True,
                          noise_std_in_model_space: Optional[torch.Tensor] = None) -> None:
    if train:
        self.model.train_model(
            variable_values=variable_values,
            objective_values=objective_values,
            noise_std_in_model_space=noise_std_in_model_space
        )
    else:
        # On reload: re-apply physical noise to override any state_dict tampering
        if noise_std_in_model_space is not None:
            self.model._apply_physical_noise(noise_std_in_model_space)
    self.acquisition_function.refresh(...)
```

---

## Step 6 — `default_settings.json`: noisy acquisition defaults

```json
"acquisition": {
    "multi_objective":        "qlogehvi",
    "single_objective":       "ucb",
    "noisy_multi_objective":  "qlogneHVI",
    "noisy_single_objective": "ucb"
}
```

---

## Step 7 — `acquisition.py`: `QLogNoisyExpectedHypervolumeImprovement`

New class (lines ~254–300 in `acquisition.py`):

```python
class QLogNoisyExpectedHypervolumeImprovement(BotorchAcquisitionFunction):
    name = 'qlogneHVI'
    multi_objective = True

    def _refresh(self, model, variable_values, objective_values) -> None:
        nadir_point = get_nadir_point(variable_values=variable_values, objective_values=objective_values)
        self.function = botorch.acquisition.multi_objective.logei \
            .qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=nadir_point,
                X_baseline=variable_values,
                prune_baseline=True
            )
```

---

## Step 8 — `constructors.py`: `is_noisy` flag + auto-select

### What changed
- `AcquisitionOptions` now includes `'qlogneHVI'`
- `ProblemInformation` has an optional `is_noisy: bool` field (split TypedDict to keep existing
  code backward-compatible — old callers without `is_noisy` default to `False`)
- `botorch_predictor` passes `is_noisy=problem_information.get('is_noisy', False)` to
  `botorch_acquisition_function`
- `botorch_acquisition_function` accepts `is_noisy: bool = False` and picks the right
  default from `default_settings.json`
- `bayesian_optimiser` sets `is_noisy=objective.noise_std is not None` in `problem_information`

### Code (key fragment in `constructors.py`)
```python
class ProblemInformationRequired(TypedDict):
    n_variables: int
    n_objectives: int
    n_evaluations_per_step: int
    bounds: list[list[float]]

class ProblemInformation(ProblemInformationRequired, total=False):
    is_noisy: bool   # optional — defaults to False in botorch_predictor

def botorch_acquisition_function(
        n_variables: int,
        n_objectives: int,
        is_noisy: bool = False,
        function: Optional[AcquisitionOptions] = None,
        parameters: Optional[AcquisitionSettings] = None
) -> BotorchAcquisitionFunction:
    if function is None:
        defaults = _load_defaults()
        if n_objectives > 1:
            default_key = 'noisy_multi_objective' if is_noisy else 'multi_objective'
            return botorch_acquisition_function(..., function=defaults['acquisition'][default_key])
        ...
```

---

## User-facing API summary

### Basic usage (physical constant noise):

```python
from veropt.optimiser.practice_objectives import Hartmann
from veropt.optimiser.constructors import bayesian_optimiser

objective = Hartmann(
    n_variables=6,
    noise_std={'Hartmann': 0.05}   # std in objective units
)

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective,
    # acquisition auto-selects UCB (single-obj) — no changes needed
)
```

### Multi-objective (auto-selects qlogneHVI):
```python
from veropt.optimiser.practice_objectives import VehicleSafety

objective = VehicleSafety(
    noise_std={'VeSa 1': 0.1, 'VeSa 2': 0.1, 'VeSa 3': 0.05}
)
# acquisition function will automatically be qlogneHVI
```

### Constraints / what NOT to do:
```python
# ❌ train_noise=True with noise_std raises ValueError at init time
objective = Hartmann(n_variables=6, noise_std={'Hartmann': 0.05})
bayesian_optimiser(..., model={'noise_settings': {'train_noise': True}})  # ValueError!

# ❌ noise_lower_bound above physical variance raises ValueError at first train
bayesian_optimiser(..., model={'noise_settings': {'noise_lower_bound': 1.0}},
                   objective=objective)  # ValueError on first train_model call!
```

---

---

## Testing

### Test file
`tests/test_noise.py` — 33 tests organised in seven classes, one per implementation step (plus two new classes added after initial implementation).

### What is currently covered

| Class | Tests | What it checks |
|---|---|---|
| `TestNormaliserTransformScale` | 3 | `transform_scale` ignores mean; round-trip inverse; differs from full `transform` for non-zero mean |
| `TestObjectiveNoiseStd` | 7 | `noise_std` stored; None by default; persisted in `gather_dicts_to_save`; key-mismatch assert; `from_saved_state` for Hartmann + VehicleSafety; backward-compat when key absent |
| `TestOptimiserNoiseConfiguration` | 6 | `train_noise=True` conflict raises; no conflict without noise; `_noise_std_tensor` value; None when unset; physical units before normaliser fit; correct shape after normaliser fit |
| `TestApplyPhysicalNoise` | 3 | Noise value pinned correctly; lower bound is `variance * 0.99`; raises when variance is below existing floor |
| `TestPredictorNoiseThreading` | 2 | Smoke-test full training loop with noise; save+reload re-applies correct variance to model |
| `TestNoisyAcquisitionAutoSelection` | 6 | `qlogneHVI` for noisy multi-obj; UCB for noisy single-obj; `qlogehvi` for non-noisy multi-obj; end-to-end optimiser wires correct acquisition; non-noisy optimiser unchanged; full noisy multi-obj run |
| `TestModelPosteriorVariance` | 4 | Noiseless GP has ≈0 variance at training points; noisy GP has non-trivial variance; noisy > noiseless; noise variance is exactly pinned after a full Adam training round |
| `TestJsonNoiseDesyncDetection` | 2 | Clean JSON reloads without error; manually edited `raw_noise` in JSON raises `ValueError("Noise desync detected")` |

### TestModelPosteriorVariance — design note
These tests access the posterior directly via `model_with_data.likelihood(model_with_data(x))`,
where `x` is taken from `model_with_data.train_inputs[0]` (GPyTorch's `ExactGP` stores the
training data here). The variance at a training point is the most sensitive check — it is
nearly zero for a noiseless interpolating GP and nonzero when likelihood noise is pinned.

The exact posterior variance is *not* asserted to equal `noise_std²` analytically (it also
depends on the kernel value K(x,x) and the full Gram matrix), but the four tests together
establish that: (a) the constraint and value are both set correctly, (b) they survive Adam
training, and (c) the effect is directionally correct.

### TestJsonNoiseDesyncDetection — design note
On reload, `BayesianOptimiser.from_saved_state` calls `_check_noise_desync_on_reload()`
*before* `_update_predictor(train=False)`. The check reads the current `likelihood.noise`
value (which was loaded from the JSON state_dict) and compares it to the value that
`objective.noise_std` + the fitted normaliser implies. A 2 % relative tolerance accommodates
normalisation rounding; anything larger is treated as a user edit and raises.

The tamper test navigates the JSON at:
```
data['optimiser']['predictor']['state']['model']['state']
    ['model_dicts']['model_0']['state']['state_dict']
    ['likelihood.noise_covar.raw_noise']
```
and sets `raw_noise = [100.0]` (softplus(100) ≈ 100, i.e. variance ≈ 100 — orders of
magnitude above the expected ~0.0025 for `noise_std=0.05`).

### What is not yet covered / worth adding

- **`DTLZ1` noise round-trip**: `VehicleSafety` is tested but `DTLZ1` is not.
- **Normalised noise value assertion**: `test_noise_std_in_model_space_with_normaliser_returns_scaled`
  only checks shape, not the actual scaled value. A tighter assertion comparing against
  `normaliser.transform_scale(noise_std_tensor)` directly would be more rigorous.
- **Model noise after renormalisation**: No test runs several bayesian steps, then checks that
  the model noise tracks the updated normaliser (i.e. variance pinned in the likelihood matches
  `_noise_std_in_model_space` after renormalisation).
- **Multi-objective noise save+load**: The reload test uses Hartmann (single objective). An
  equivalent test for VehicleSafety / three objectives is missing.
- **Noisy acquisition with explicit override**: No test checks that passing
  `acquisition={'function': 'qlogehvi'}` explicitly on a noisy optimiser is honoured.

### Performance note — why are the tests slow?

The main time consumers in order of impact:

1. **Acquisition function optimisation (dual annealing)** — every call to
   `run_optimisation_step()` that has passed the initial phase runs the dual-annealing
   acquisition optimiser. This is the dominant cost per step even with `max_iter=5`, because
   dual annealing has its own budget independent of the GP training iterations.
2. **GP model training** — Adam / L-BFGS-B iterations over the kernel hyperparameters.
   With `max_iter=5` this is kept minimal, but it still involves forward/backward passes.
3. **Repeated full runs** — several tests call `run_optimisation_step()` 4–6 times.

The quickest wins to speed things up:
- Reduce the dual-annealing budget via `acquisition_optimiser={'maxiter': N}` (small N like 50).
- Use Hartmann-3 instead of Hartmann-6 for tests that only need to confirm wiring — fewer
  variables means a smaller GP and faster acquisition optimisation.

---

---

## Step 9 — Interfaces: `noise_std` in `ExperimentConfig` and `ExperimentObjective`

### What changed
- `ExperimentConfig` (pydantic model in `experiment_utility.py`) gained an optional
  `noise_std: Optional[dict[str, float]] = None` field — fully backward-compatible
  since it defaults to `None`.
- `ExperimentObjective.__init__` gained the same `noise_std` parameter, which it
  forwards to the `InterfaceObjective` / `Objective` base class.
- `ExperimentObjective.from_saved_state` reads it with `.get("noise_std", None)` for
  backward compat with saved JSONs that predate this change.
- `Experiment._make_fresh_optimiser` passes `experiment_config.noise_std` to the
  `ExperimentObjective` it constructs, so the noise flows all the way into the
  `BayesianOptimiser`.
- `examples/interfaces/template_experiment.py` has a documentation comment showing
  how to add `noise_std` to the experiment config JSON.

### Why `ExperimentConfig` and not the optimiser settings JSON
`noise_std` is a property of the **problem** (how noisy are the simulation outputs),
not of the GP algorithm.  It lives alongside `parameter_bounds` in the experiment
config, which is conceptually the right home.  The optimiser settings JSON remains
algorithm-only (kernel choice, training iterations, etc.).

### How a user sets noise in an experiment
Add `"noise_std"` to the experiment config JSON keyed by objective name:

```json
{
    "experiment_name": "my_run",
    "parameter_names": ["param1", "param2"],
    "parameter_bounds": {"param1": [0.0, 1.0], "param2": [-1.0, 1.0]},
    "path_to_experiment": "/path/to/results",
    "experiment_mode": "local_slurm",
    "output_filename": "output.nc",
    "noise_std": {"rmse_temp": 0.05, "rmse_salt": 0.02}
}
```

No changes to the optimiser settings JSON are needed.  veropt then:
- Pins the GP likelihood noise to those values (via the existing Steps 3–5 wiring)
- Auto-selects `qLogNoisyEHVI` for multi-objective problems (Step 8)
- Shows uncertainty ellipses on Pareto plots (Step 10 below)

---

## Step 10 — Graphical: Noise-aware Pareto front

### What changed
- `_pareto_front.py`: `_add_pareto_traces_2d` and both `_plot_pareto_front` /
  `_plot_pareto_front_grid` functions accept `noise_std_per_objective: Optional[torch.Tensor]`
  and `uncertainty_style: UncertaintyStyle = 'ellipse'`.
- **Ellipse style (default)**: a single batched `go.Scatter` trace draws one ±1σ
  ellipse per point (width = σ_x, height = σ_y).  All ellipses share one legend entry.
- **Error bars style**: conventional `error_x` / `error_y` on the scatter traces
  (no extra trace).
- `visualisation.py`: both public functions `plot_pareto_front_grid` and
  `plot_pareto_front` accept `uncertainty_style` and now also pass
  `noise_std_per_objective` to `get_pareto_optimal_points` so that the **highlighted
  Pareto front uses noise-aware dominance** (Step 10b below).

### Step 10b — Noise-aware dominance wired into visualisation
`get_pareto_optimal_points` uses ε-dominance (Laumanns et al. 2002, IEEE TEC 6(3))
with ε_j = epsilon_n_sigma · σ_j: point B dominates A only if B_j > A_j + ε_j for **all** j.
The tolerance is set in units of the noise std; default epsilon_n_sigma=1 gives a 1σ margin.
The visualisation functions pass `noise_std_per_objective` to this function so
borderline points near the noise floor are not prematurely excluded from the highlighted
front.

### Code (key fragments in `visualisation.py`)
```python
def plot_pareto_front_grid(
        optimiser: BayesianOptimiser,
        normalised: bool = False,
        uncertainty_style: str = 'ellipse'
) -> go.Figure:
    ...
    noise_std_per_objective = optimiser._noise_std_tensor
    pareto_optimal_indices = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        noise_std_per_objective=noise_std_per_objective,
    )['index']
    figure = _plot_pareto_front_grid(
        ...,
        noise_std_per_objective=noise_std_per_objective,
        uncertainty_style=uncertainty_style,
        return_figure=True
    )
```

### User-facing API
```python
# Default: ellipses
fig = plot_pareto_front_grid(optimiser=optimiser)

# Alternative: error bars
fig = plot_pareto_front(
    optimiser=optimiser,
    plotted_objective_indices=[0, 1],
    uncertainty_style='error_bars',
)
```
See `examples/example_noisy_pareto_front.py` for a complete runnable example.

---

## Files changed (updated)

| File | Lines (approx.) | Change |
|---|---|---|
| `normalisation.py` | 24–51 | `transform_scale`, `inverse_transform_scale` |
| `objective.py` | 15–62 | `noise_std` param + assert + save |
| `practice_objectives.py` | full | `noise_std` through all 3 practice objectives |
| `model.py` | 720–760, 780–825 | `train_model` noise param; `_apply_physical_noise` |
| `prediction.py` | 122–130, 299–326 | `noise_std_in_model_space` param in abstract + concrete |
| `optimiser.py` | 80–85, 721–779, 1155–1175 | `_check_noise_configuration`; noise properties; `_update_predictor` wiring |
| `acquisition.py` | 254–300 | `QLogNoisyExpectedHypervolumeImprovement` |
| `constructors.py` | 25–26, 34–46, 118–168, 376–420 | `is_noisy` flag; split `ProblemInformation`; auto-select |
| `default_settings.json` | 6–9 | `noisy_multi_objective`, `noisy_single_objective` keys |
| `interfaces/experiment_utility.py` | `ExperimentConfig`, `ExperimentObjective` | `noise_std` field + backward-compat `from_saved_state` |
| `interfaces/experiment.py` | `_make_fresh_optimiser` | passes `experiment_config.noise_std` to `ExperimentObjective` |
| `graphical/_pareto_front.py` | full | `noise_std_per_objective` + `uncertainty_style` params; ellipse + error-bar rendering |
| `graphical/visualisation.py` | `plot_pareto_front`, `plot_pareto_front_grid` | `uncertainty_style` param; noise-aware dominance wired |
| `examples/example_noisy_pareto_front.py` | new | runnable noisy Pareto front example |
| `examples/interfaces/template_experiment.py` | comment block | documents `noise_std` JSON field |
| `tests/test_noise.py` | new | 33 tests covering Steps 1–8 |
| `tests/test_graphical.py` | `TestUncertainParetoFront` | 9 tests covering Steps 9–10 |
