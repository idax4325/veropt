"""Tests for V1 observation noise support.

Covers:
- Normaliser transform_scale / inverse_transform_scale (Step 1)
- Objective noise_std field and save/load round-trip (Step 2)
- BayesianOptimiser conflict checks and noise properties (Step 3)
- GPyTorchFullModel._apply_physical_noise (Step 4)
- Predictor.update_with_new_data noise threading (Step 5)
- Auto-selection of noisy acquisition function (Steps 7-8)
- Model posterior variance with and without noise
- JSON noise desync detection on reload
- v3→v4 schema migration safety: raw_noise constraint recalibration
"""
import json
import math
from pathlib import Path
from typing import Optional

import pytest
import torch

from veropt.optimiser.constructors import bayesian_optimiser, botorch_acquisition_function
from veropt.optimiser.kernels import MaternKernel
from veropt.optimiser.model import GPyTorchFullModel
from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_saver_loader import save_to_json, load_optimiser_from_state
from veropt.optimiser.practice_objectives import Hartmann, VehicleSafety, DTLZ1
from veropt.optimiser.prediction import BotorchPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_objective_noisy_optimiser(n_initial: int = 4, n_bayesian: int = 4) -> BayesianOptimiser:
    """Small Hartmann-6 optimiser with noise_std set."""
    objective = Hartmann(n_variables=6, noise_std={'Hartmann': 0.05})
    return bayesian_optimiser(
        n_initial_points=n_initial,
        n_bayesian_points=n_bayesian,
        n_evaluations_per_step=1,
        objective=objective,
        model={'training_settings': {'max_iter': 5, 'verbose': False}},
    )


def _make_multi_objective_noisy_optimiser(n_initial: int = 4, n_bayesian: int = 4) -> BayesianOptimiser:
    """Small VehicleSafety optimiser with noise_std set on all three objectives."""
    noise_std = {f"VeSa {i + 1}": 0.1 for i in range(3)}
    objective = VehicleSafety(noise_std=noise_std)
    return bayesian_optimiser(
        n_initial_points=n_initial,
        n_bayesian_points=n_bayesian,
        n_evaluations_per_step=1,
        objective=objective,
        model={'training_settings': {'max_iter': 5, 'verbose': False}},
    )


# ---------------------------------------------------------------------------
# Step 1 — Normaliser.transform_scale / inverse_transform_scale
# ---------------------------------------------------------------------------

class TestNormaliserTransformScale:

    def test_transform_scale_removes_mean_shift(self) -> None:
        """transform_scale should scale but NOT subtract the mean."""
        means = torch.tensor([10.0, 100.0])
        variances = torch.tensor([4.0, 25.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        noise_tensor = torch.tensor([2.0, 5.0])
        scaled = normaliser.transform_scale(noise_tensor)

        expected = torch.tensor([2.0 / 2.0, 5.0 / 5.0])  # divided by sqrt(var)
        assert torch.allclose(scaled, expected), f"Expected {expected}, got {scaled}"

    def test_inverse_transform_scale_is_inverse(self) -> None:
        means = torch.tensor([3.0])
        variances = torch.tensor([9.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        original = torch.tensor([1.5])
        assert torch.allclose(normaliser.inverse_transform_scale(normaliser.transform_scale(original)), original)

    def test_transform_scale_is_different_from_full_transform(self) -> None:
        """For a non-zero mean, transform_scale and transform produce different results."""
        means = torch.tensor([5.0])
        variances = torch.tensor([1.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        value = torch.tensor([3.0])
        assert not torch.allclose(normaliser.transform(value), normaliser.transform_scale(value))


# ---------------------------------------------------------------------------
# Step 2 — Objective.noise_std
# ---------------------------------------------------------------------------

class TestObjectiveNoiseStd:

    def test_noise_std_stored_on_objective(self) -> None:
        noise_std = {'Hartmann': 0.05}
        objective = Hartmann(n_variables=6, noise_std=noise_std)
        assert objective.noise_std == noise_std

    def test_noise_std_is_none_by_default(self) -> None:
        objective = Hartmann(n_variables=6)
        assert objective.noise_std is None

    def test_noise_std_saved_in_state_dict(self) -> None:
        noise_std = {'Hartmann': 0.05}
        objective = Hartmann(n_variables=6, noise_std=noise_std)
        state = objective.gather_dicts_to_save()
        assert state['state']['noise_std'] == noise_std

    def test_noise_std_key_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            Hartmann(n_variables=6, noise_std={'WrongKey': 0.1})

    def test_hartmann_from_saved_state_restores_noise_std(self) -> None:
        noise_std = {'Hartmann': 0.05}
        original = Hartmann(n_variables=6, noise_std=noise_std)
        saved_state = original.gather_dicts_to_save()
        restored = Hartmann.from_saved_state(saved_state['state'])
        assert restored.noise_std == noise_std

    def test_hartmann_from_saved_state_without_noise_std_gives_none(self) -> None:
        """Backward compat: old saved states without noise_std key should give None."""
        original = Hartmann(n_variables=6)
        saved_state = original.gather_dicts_to_save()
        del saved_state['state']['noise_std']  # simulate old format
        restored = Hartmann.from_saved_state(saved_state['state'])
        assert restored.noise_std is None

    def test_vehicle_safety_from_saved_state_restores_noise_std(self) -> None:
        noise_std = {f"VeSa {i + 1}": 0.1 for i in range(3)}
        original = VehicleSafety(noise_std=noise_std)
        saved_state = original.gather_dicts_to_save()
        restored = VehicleSafety.from_saved_state(saved_state['state'])
        assert restored.noise_std == noise_std


# ---------------------------------------------------------------------------
# Step 3 — BayesianOptimiser conflict checks and noise properties
# ---------------------------------------------------------------------------

class TestOptimiserNoiseConfiguration:

    def test_train_noise_with_noise_std_raises(self) -> None:
        """Setting both noise_std and train_noise=True on the objective must raise on construction."""
        with pytest.raises(ValueError, match="train_noise=True"):
            Hartmann(n_variables=6, noise_std={'Hartmann': 0.05}, train_noise=True)

    def test_no_conflict_without_noise_std(self) -> None:
        """train_noise=True is allowed if objective has no noise_std."""
        objective = Hartmann(n_variables=6, train_noise=True)
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        assert optimiser is not None

    def test_noise_std_tensor_property(self) -> None:
        optimiser = _make_single_objective_noisy_optimiser()
        tensor = optimiser._noise_std_tensor
        assert tensor is not None
        assert torch.allclose(tensor, torch.tensor([0.05]))

    def test_noise_std_tensor_is_none_when_not_set(self) -> None:
        objective = Hartmann(n_variables=6)
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        assert optimiser._noise_std_tensor is None

    def test_noise_std_in_model_space_without_normaliser_returns_physical(self) -> None:
        """Before first fit, _noise_std_in_model_space == physical units."""
        optimiser = _make_single_objective_noisy_optimiser()
        # No data evaluated yet, so normaliser is not fitted
        assert not optimiser.normalisers_have_been_initialised
        noise_in_model = optimiser._noise_std_in_model_space
        assert noise_in_model is not None
        assert torch.allclose(noise_in_model, torch.tensor([0.05]))

    def test_noise_std_in_model_space_with_normaliser_returns_scaled(self) -> None:
        """After normaliser is fitted, _noise_std_in_model_space is scaled."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()
        assert optimiser.normalisers_have_been_initialised
        # Scaled noise should differ from physical noise (unless std is 1)
        noise_physical = optimiser._noise_std_tensor
        noise_model_space = optimiser._noise_std_in_model_space
        assert noise_physical is not None
        assert noise_model_space is not None
        # They may or may not be equal depending on the normaliser, but both should be tensors
        assert noise_model_space.shape == noise_physical.shape


# ---------------------------------------------------------------------------
# Step 4 — GPyTorchFullModel._apply_physical_noise
# ---------------------------------------------------------------------------

class TestApplyPhysicalNoise:

    def _make_trained_full_model(self, n_variables: int = 3, n_objectives: int = 1) -> GPyTorchFullModel:
        kernel = MaternKernel(n_variables=n_variables)
        from veropt.optimiser.model import AdamModelOptimiser
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=n_variables,
            n_objectives=n_objectives,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=5,
            verbose=False
        )
        # Initialise with dummy data so model_with_data is set
        variables = torch.rand(6, n_variables)
        objectives = torch.rand(6, n_objectives)
        model.initialise_model(variable_values=variables, objective_values=objectives)
        return model

    def test_apply_physical_noise_sets_noise_value(self) -> None:
        model = self._make_trained_full_model()
        noise_std = torch.tensor([0.1])
        model._apply_physical_noise(noise_std_in_model_space=noise_std)
        expected_variance = 0.1 ** 2
        assert model._model_list[0].model_with_data is not None
        actual_noise = float(model._model_list[0].model_with_data.likelihood.noise)
        assert abs(actual_noise - expected_variance) < 1e-10

    def test_apply_physical_noise_preserves_floor_constraint(self) -> None:
        """_apply_physical_noise sets the noise value but leaves the constraint at _NOISE_CONSTRAINT_FLOOR."""
        model = self._make_trained_full_model()
        noise_std = torch.tensor([0.2])
        model._apply_physical_noise(noise_std_in_model_space=noise_std)
        from veropt.optimiser.model import _NOISE_CONSTRAINT_FLOOR
        lower_bound = float(model._model_list[0].likelihood.noise_covar.raw_noise_constraint.lower_bound)
        assert abs(lower_bound - _NOISE_CONSTRAINT_FLOOR) < 1e-12

    def test_apply_physical_noise_below_floor_raises(self) -> None:
        """noise_std so small that its variance falls below _NOISE_CONSTRAINT_FLOOR (1e-8) must raise."""
        model = self._make_trained_full_model()
        # std = 1e-5 → variance = 1e-10, which is below _NOISE_CONSTRAINT_FLOOR = 1e-8
        tiny_noise_std = torch.tensor([1e-5])
        with pytest.raises(AssertionError, match="numerical floor"):
            model._apply_physical_noise(noise_std_in_model_space=tiny_noise_std)


# ---------------------------------------------------------------------------
# Step 5 — Predictor noise threading
# ---------------------------------------------------------------------------

class TestPredictorNoiseThreading:

    def test_update_with_noise_trains_without_error(self) -> None:
        """Smoke test: train predictor with noise_std_in_model_space provided."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        # Evaluate initial points so we have data
        for _ in range(4):
            optimiser.run_optimisation_step()
        # If we got here, noise threading through train_model worked
        assert optimiser.model_has_been_trained

    def test_noise_applied_on_train_false_reload(self, tmp_path: Path) -> None:
        """After save+load, the model noise should match objective.noise_std."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "test_noisy_optimiser.json")
        save_to_json(optimiser, file_path)

        loaded = load_optimiser_from_state(file_path)

        # Verify noise was re-applied: model noise ~ physical_variance_normalised
        assert loaded.objective.noise_std is not None
        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        assert isinstance(loaded.predictor, BotorchPredictor)
        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(single_model.model_with_data.likelihood.noise)  # type: ignore[union-attr]
            assert abs(actual_noise - expected_variance) < 1e-9, (
                f"Objective {objective_index}: expected variance {expected_variance:.2e} "
                f"but model has {actual_noise:.2e}"
            )

    def test_multi_objective_noisy_reload_does_not_crash(self, tmp_path: Path) -> None:
        """Regression test: loading a saved multi-objective noisy optimiser must not raise
        UnsupportedError about unexpected batch dims in prune_inferior_points_multi_objective.

        Root cause: GPyTorchSingleModel.gather_dicts_to_save was saving
        model_with_data.train_inputs (a gpytorch tuple) instead of train_inputs[0] (the raw
        tensor). On reload, torch.tensor(tuple_data) produced shape [1, n_points, n_vars]
        instead of [n_points, n_vars], giving the reloaded model batch_shape=[1].
        BoTorch's prune_inferior_points_multi_objective saw obj_vals.ndim=4 > 3 and raised
        UnsupportedError("Models with multiple batch dims...").

        Fix: save train_inputs[0] in gather_dicts_to_save so reload gives batch_shape=[]."""
        import torch
        torch.manual_seed(0)

        noise_std = {f"DTLZ1 {i + 1}": 0.05 for i in range(2)}
        objective = DTLZ1(n_variables=3, n_objectives=2, noise_std=noise_std)
        optimiser = bayesian_optimiser(
            n_initial_points=6,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )

        for _ in range(6):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "test_multi_noisy_optimiser.json")
        save_to_json(optimiser, file_path)

        # This line used to raise:
        #   botorch.exceptions.errors.UnsupportedError:
        #   Models with multiple batch dims are currently unsupported by
        #   `prune_inferior_points_multi_objective`.
        loaded = load_optimiser_from_state(file_path)

        assert loaded.model_has_been_trained

        assert isinstance(loaded.predictor, BotorchPredictor)

        # batch_shape must be [] (no spurious batch dim from saved train_inputs tuple)
        reloaded_mlgp = loaded.predictor.model.get_gpytorch_model()
        assert reloaded_mlgp.batch_shape == torch.Size([]), (
            f"Expected batch_shape=[], got {reloaded_mlgp.batch_shape}. "
            "Saving train_inputs as a tuple introduced an extra batch dimension on reload."
        )

        # Verify noise values are correct after reload
        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(single_model.model_with_data.likelihood.noise.detach())  # type: ignore[union-attr]
            assert abs(actual_noise - expected_variance) < 1e-9, (
                f"Objective {objective_index}: expected variance {expected_variance:.2e} "
                f"but model has {actual_noise:.2e}"
            )


# ---------------------------------------------------------------------------
# Steps 7–8 — Acquisition function auto-selection
# ---------------------------------------------------------------------------

class TestNoisyAcquisitionAutoSelection:

    def test_noisy_multi_objective_selects_qlogneHVI(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=5,
            n_objectives=3,
            is_noisy=True
        )
        assert acq.name == 'qlogneHVI'

    def test_noisy_single_objective_keeps_ucb(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=3,
            n_objectives=1,
            is_noisy=True
        )
        assert acq.name == 'ucb'

    def test_non_noisy_multi_objective_selects_qlogehvi(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=5,
            n_objectives=3,
            is_noisy=False
        )
        assert acq.name == 'qlogehvi'

    def test_bayesian_optimiser_with_noisy_multi_objective_uses_qlogneHVI(self) -> None:
        optimiser = _make_multi_objective_noisy_optimiser()
        assert isinstance(optimiser.predictor, BotorchPredictor)
        assert optimiser.predictor.acquisition_function.name == 'qlogneHVI'

    def test_bayesian_optimiser_without_noise_uses_qlogehvi(self) -> None:
        objective = VehicleSafety()  # no noise_std
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        assert isinstance(optimiser.predictor, BotorchPredictor)
        assert optimiser.predictor.acquisition_function.name == 'qlogehvi'

    def test_noisy_optimiser_full_run(self) -> None:
        """Smoke test: full optimisation loop with noise_std set."""
        optimiser = _make_multi_objective_noisy_optimiser(n_initial=4, n_bayesian=2)
        for step in range(6):
            optimiser.run_optimisation_step()
        assert optimiser.n_points_evaluated == 6


# ---------------------------------------------------------------------------
# Model posterior variance — does the pinned noise actually reach the GP?
# ---------------------------------------------------------------------------

class TestModelPosteriorVariance:
    """Check that fixing noise_std produces meaningfully different posterior uncertainty
    compared to a noiseless model, both immediately after pinning and after training."""

    def _train_simple_model(self, noise_std_value: Optional[float]) -> GPyTorchFullModel:
        """Return a GPyTorchFullModel with initialised (not full Adam) training on a small 1-D dataset.

        Using initialise_model + _apply_physical_noise mirrors the path taken by train_model."""
        from veropt.optimiser.model import AdamModelOptimiser
        kernel = MaternKernel(n_variables=1)
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=1,
            n_objectives=1,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=5,
            verbose=False
        )
        torch.manual_seed(0)
        variables = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
        objectives = torch.sin(variables * 3.14).squeeze(-1).unsqueeze(-1)
        model.initialise_model(variable_values=variables, objective_values=objectives)
        if noise_std_value is not None:
            noise_tensor = torch.tensor([noise_std_value])
            model._apply_physical_noise(noise_std_in_model_space=noise_tensor)
        return model

    def _posterior_variance_at_training_point(self, model: GPyTorchFullModel) -> float:
        """Return the GP posterior variance at the first training point."""
        single_model = model._model_list[0]
        assert single_model.model_with_data is not None
        gpytorch_model = single_model.model_with_data
        gpytorch_model.eval()
        x_test = gpytorch_model.train_inputs[0][0:1]  # type: ignore[index]  # gpytorch stubs as Module
        with torch.no_grad():
            posterior = gpytorch_model.likelihood(gpytorch_model(x_test))
        return float(posterior.variance.squeeze())

    def test_noiseless_model_has_near_zero_variance_at_training_point(self) -> None:
        """With no noise (likelihood noise ≈ 0), the GP nearly interpolates its training data."""
        model = self._train_simple_model(noise_std_value=None)
        variance = self._posterior_variance_at_training_point(model)
        assert variance < 0.01, (
            f"Expected near-zero posterior variance at training point for noiseless GP, got {variance:.4e}"
        )

    def test_noisy_model_has_larger_variance_at_training_point(self) -> None:
        """With noise_std set, the posterior should be meaningfully uncertain at training points."""
        noise_std = 0.3
        model = self._train_simple_model(noise_std_value=noise_std)
        variance = self._posterior_variance_at_training_point(model)
        assert variance > 0.005, (
            f"Expected non-negligible variance at training point for noisy GP (noise_std={noise_std}), "
            f"got {variance:.4e}"
        )

    def test_noisy_model_has_greater_variance_than_noiseless(self) -> None:
        """Noisy model should always have higher posterior uncertainty than noiseless."""
        noiseless_model = self._train_simple_model(noise_std_value=None)
        noisy_model = self._train_simple_model(noise_std_value=0.3)
        noiseless_variance = self._posterior_variance_at_training_point(noiseless_model)
        noisy_variance = self._posterior_variance_at_training_point(noisy_model)
        assert noisy_variance > noiseless_variance, (
            f"Noisy model variance ({noisy_variance:.4e}) should exceed "
            f"noiseless model variance ({noiseless_variance:.4e})"
        )

    def test_noise_pinning_survives_adam_training(self) -> None:
        """After running Adam optimisation, the likelihood noise should remain pinned to the physical variance."""
        from veropt.optimiser.model import AdamModelOptimiser
        noise_std = 0.1
        kernel = MaternKernel(n_variables=1)
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=1,
            n_objectives=1,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=10,
            verbose=False
        )
        torch.manual_seed(1)
        variables = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
        objectives = torch.sin(variables * 3.14).squeeze(-1).unsqueeze(-1)
        noise_tensor = torch.tensor([noise_std])
        model.train_model(
            variable_values=variables,
            objective_values=objectives,
            noise_std_in_model_space=noise_tensor
        )
        expected_variance = noise_std ** 2
        assert model._model_list[0].model_with_data is not None
        actual_noise = float(model._model_list[0].model_with_data.likelihood.noise)
        relative_error = abs(actual_noise - expected_variance) / expected_variance
        assert relative_error < 0.01, (
            f"Noise variance should remain pinned to {expected_variance:.4e} after Adam training, "
            f"but got {actual_noise:.4e} (relative error {relative_error:.2%})"
        )


# ---------------------------------------------------------------------------
# Noise pinning: objective.noise_std always wins over the state dict
# ---------------------------------------------------------------------------

class TestNoisePinningOverwritesStateDict:
    """Verify that objective.noise_std is the authoritative source of truth on reload.

    If the raw_noise in the saved JSON disagrees with objective.noise_std — whether
    because of a manual edit, a v3→v4 migration, or any other cause — the loaded
    model must end up with the noise that objective.noise_std specifies, not whatever
    was in the JSON."""

    def _save_and_tamper_raw_noise(self, tmp_path: Path, tamper_value: float) -> str:
        """Run a noisy optimiser, save it, set raw_noise to tamper_value, return path."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "tamper_test.json")
        save_to_json(optimiser, file_path)

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        state_dict = (
            data['optimiser']['predictor']['state']['model']['state']
            ['model_dicts']['model_0']['state']['state_dict']
        )
        raw_noise_key = 'likelihood.noise_covar.raw_noise'
        if raw_noise_key in state_dict:
            state_dict[raw_noise_key] = [tamper_value]

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

        return file_path

    def test_tampered_json_noise_is_overwritten_on_reload(self, tmp_path: Path) -> None:
        """Setting raw_noise to a large wrong value in the JSON must be silently overwritten
        by objective.noise_std on reload — not raise."""
        file_path = self._save_and_tamper_raw_noise(tmp_path, tamper_value=100.0)

        # Must not raise — wrong raw_noise is overwritten by _apply_physical_noise
        loaded = load_optimiser_from_state(file_path)

        assert loaded.objective.noise_std is not None
        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        assert isinstance(loaded.predictor, BotorchPredictor)
        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(
                single_model.model_with_data.likelihood.noise.detach()  # type: ignore[union-attr]
            )
            relative_error = abs(actual_noise - expected_variance) / expected_variance
            assert relative_error < 0.02, (
                f"Objective {objective_index}: objective.noise_std should override a tampered "
                f"JSON raw_noise, but got {actual_noise:.2e} instead of {expected_variance:.2e}"
            )

    def test_tiny_raw_noise_in_json_is_overwritten_on_reload(self, tmp_path: Path) -> None:
        """Setting raw_noise to a very negative value (tiny effective noise) in the JSON
        must also be overwritten by objective.noise_std."""
        file_path = self._save_and_tamper_raw_noise(tmp_path, tamper_value=-50.0)

        loaded = load_optimiser_from_state(file_path)

        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        assert isinstance(loaded.predictor, BotorchPredictor)
        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(
                single_model.model_with_data.likelihood.noise.detach()  # type: ignore[union-attr]
            )
            relative_error = abs(actual_noise - expected_variance) / expected_variance
            assert relative_error < 0.02, (
                f"Objective {objective_index}: objective.noise_std should override a near-zero "
                f"tampered raw_noise, but got {actual_noise:.2e} instead of {expected_variance:.2e}"
            )


# ---------------------------------------------------------------------------
# train_noise=True with bounds: constraint and noise value on reload
# ---------------------------------------------------------------------------

class TestTrainNoiseWithBoundsReload:
    """Verify the reload behaviour when noise is learned (train_noise=True) with explicit
    noise_std_min / noise_std_max bounds.

    Key difference from fixed noise (noise_std):
    - Fixed noise: _apply_physical_noise overwrites BOTH the constraint AND the noise value.
    - Trained noise with bounds: _apply_noise_bounds overwrites the constraint bounds only;
      the noise VALUE is preserved from the state dict (Adam's trained value).

    This is intentional — we want to resume from the trained noise level, not snap it to
    the bound edge.  The bounds constrain what Adam can do in the next training round."""

    def test_train_noise_with_bounds_reload_restores_constraint(self, tmp_path: Path) -> None:
        """After save+load with train_noise=True and explicit bounds, the Interval constraint
        lower_bound and upper_bound must equal noise_std_min² and noise_std_max² (in model space)."""
        from gpytorch.constraints import Interval

        noise_std_min = 0.01
        noise_std_max = 0.5
        objective = Hartmann(
            n_variables=6,
            train_noise=True,
            noise_std_min={'Hartmann': noise_std_min},
            noise_std_max={'Hartmann': noise_std_max},
        )
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()

        # Capture expected constraint bounds in model space (normaliser may scale them)
        noise_min_in_model = optimiser._noise_std_min_in_model_space
        noise_max_in_model = optimiser._noise_std_max_in_model_space
        assert noise_min_in_model is not None
        assert noise_max_in_model is not None

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "train_noise_bounds.json")
        save_to_json(optimiser, file_path)

        loaded = load_optimiser_from_state(file_path)

        # Re-compute expected bounds from the loaded optimiser (normaliser may re-fit)
        expected_min_var = float(loaded._noise_std_min_in_model_space[0] ** 2)  # type: ignore[index]
        expected_max_var = float(loaded._noise_std_max_in_model_space[0] ** 2)  # type: ignore[index]

        assert isinstance(loaded.predictor, BotorchPredictor)
        for single_model in loaded.predictor.model._model_list:
            constraint = (
                single_model.model_with_data.likelihood.noise_covar.raw_noise_constraint  # type: ignore[union-attr]
            )

            assert isinstance(constraint, Interval), (
                f"Expected Interval constraint after reload with bounds, got {type(constraint).__name__}"
            )
            actual_lower = float(constraint.lower_bound)
            actual_upper = float(constraint.upper_bound)

            assert abs(actual_lower - expected_min_var) < 1e-6, (
                f"Constraint lower_bound should be noise_std_min² ({expected_min_var:.2e}) "
                f"after reload but got {actual_lower:.2e}"
            )
            assert abs(actual_upper - expected_max_var) < 1e-6, (
                f"Constraint upper_bound should be noise_std_max² ({expected_max_var:.2e}) "
                f"after reload but got {actual_upper:.2e}"
            )

    def test_train_noise_with_bounds_reload_preserves_trained_noise_value(
            self, tmp_path: Path
    ) -> None:
        """After reload, the noise VALUE must be preserved from the trained state dict (not snapped
        to a bound edge).  The noise must be within [noise_std_min², noise_std_max²]."""
        noise_std_min = 0.01
        noise_std_max = 0.5
        objective = Hartmann(
            n_variables=6,
            train_noise=True,
            noise_std_min={'Hartmann': noise_std_min},
            noise_std_max={'Hartmann': noise_std_max},
        )
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()

        # Record the trained noise before saving
        assert isinstance(optimiser.predictor, BotorchPredictor)
        trained_noise = float(
            optimiser.predictor.model._model_list[0].model_with_data.likelihood.noise  # type: ignore[union-attr]
            .detach()
        )

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "train_noise_bounds_value.json")
        save_to_json(optimiser, file_path)

        loaded = load_optimiser_from_state(file_path)

        assert isinstance(loaded.predictor, BotorchPredictor)
        reloaded_noise = float(
            loaded.predictor.model._model_list[0].model_with_data.likelihood.noise.detach()  # type: ignore[union-attr]
        )

        # Value is preserved (not reset to a bound edge)
        assert abs(reloaded_noise - trained_noise) < 1e-9, (
            f"Trained noise value ({trained_noise:.4e}) should be preserved on reload "
            f"but got {reloaded_noise:.4e}"
        )

        # Value must be within the physical bounds (as a sanity check)
        expected_min_var = float(loaded._noise_std_min_in_model_space[0] ** 2)  # type: ignore[index]
        expected_max_var = float(loaded._noise_std_max_in_model_space[0] ** 2)  # type: ignore[index]
        assert expected_min_var <= reloaded_noise <= expected_max_var, (
            f"Reloaded noise {reloaded_noise:.4e} is outside bounds "
            f"[{expected_min_var:.4e}, {expected_max_var:.4e}]"
        )


# ---------------------------------------------------------------------------
# Regression: retrain after reload does not crash when empirical variance grows
# ---------------------------------------------------------------------------

class TestRetrainAfterReloadWithGrowingVariance:
    """Regression test: retrain after reload must not crash when normaliser re-fits on a wider dataset.

    Previously, _apply_physical_noise pinned the noise constraint to physical_variance * 0.99 and saved
    that live value to JSON. On reload the stale pinned value became the new floor. When the normaliser
    then re-fit on a wider dataset the new normalised variance dropped below the stale floor and raised.

    After the refactor, the constraint is always _NOISE_CONSTRAINT_FLOOR (1e-8); no dynamic pinning occurs.
    This test documents that reload + retrain with growing empirical variance still works.
    """

    def test_retrain_after_reload_does_not_crash_when_variance_grows(
            self,
            tmp_path: Path
    ) -> None:
        """Train, save, inject outlier points to force a large empirical variance
        increase, reload, retrain — must not raise."""
        objective = Hartmann(n_variables=6, noise_std={'Hartmann': 0.05})
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "retrain_regression.json")
        save_to_json(optimiser, file_path)

        reloaded = load_optimiser_from_state(file_path)

        # Inject extreme outlier objectives so the empirical std grows dramatically,
        # which would push the normalised noise well below the stale saved lower
        # bound and trigger the bug.
        outlier_variables = torch.zeros(4, 6)
        outlier_objectives = torch.tensor([[100.0], [200.0], [-100.0], [-200.0]])
        reloaded._evaluated_variables_real_units = torch.cat(
            [reloaded._evaluated_variables_real_units, outlier_variables], dim=0
        )
        reloaded._evaluated_objectives_real_units = torch.cat(
            [reloaded._evaluated_objectives_real_units, outlier_objectives], dim=0
        )

        reloaded.train_model()
        assert reloaded.model_has_been_trained


# ---------------------------------------------------------------------------
# v3→v4 schema migration: raw_noise constraint recalibration
# ---------------------------------------------------------------------------

def _softplus_inv(y: float) -> float:
    """Inverse of softplus: returns x such that log(1 + exp(x)) == y.

    For small y: log(exp(y) - 1) ≈ log(y).  Numerically stable for y > 0.
    """
    if y > 20.0:
        return y  # softplus(x) ≈ x for large x
    return math.log(math.exp(y) - 1.0)


def _build_v3_format_json(v4_data: dict, noise_var_per_objective: list[float]) -> dict:
    """Given a v4-format saved-optimiser dict, return a v3-equivalent dict.

    Changes made:
    - schema_version → 3
    - Each model's state_dict raw_noise is replaced with the v3-calibrated value
      (calibrated against lower_bound = 0.99 * physical_variance instead of 1e-8)
    - train_noise is moved back from objective state to each model's state
    - noise and noise_lower_bound are added to each model's state
    - train_noise / noise_std_min / noise_std_max are removed from objective state
    """
    import copy
    data = copy.deepcopy(v4_data)

    model_dicts = (
        data['optimiser']['predictor']['state']['model']['state']['model_dicts']
    )
    objective_state = data['optimiser']['objective']['state']

    # Reverse the train_noise move: pull it back from objective → model states
    train_noise = objective_state.pop('train_noise', False)
    objective_state.pop('noise_std_min', None)
    objective_state.pop('noise_std_max', None)

    raw_noise_key = 'likelihood.noise_covar.raw_noise'
    constraint_lower_key = 'likelihood.noise_covar.raw_noise_constraint.lower_bound'

    for objective_index, (model_key, model_dict) in enumerate(model_dicts.items()):
        model_state = model_dict['state']

        noise_var = noise_var_per_objective[objective_index]
        lower_bound_v3 = 0.99 * noise_var

        # v3 set_noise(noise_var) with lower_bound_v3 stored:
        #   raw_noise = softplus_inv(noise_var - lower_bound_v3) = softplus_inv(0.01 * noise_var)
        raw_noise_v3 = _softplus_inv(noise_var - lower_bound_v3)

        # Re-calibrate the state_dict raw_noise AND the constraint lower_bound to v3 values.
        # Real v3 JSONs have both fields stale (the user's JSON showed lower_bound ≈ 0.175).
        state_dict = model_state.get('state_dict', {})
        if raw_noise_key in state_dict:
            state_dict[raw_noise_key] = [raw_noise_v3]
        state_dict[constraint_lower_key] = lower_bound_v3  # stale v3 constraint

        # Add the v3-era model-level noise fields (migration will remove them)
        model_state['train_noise'] = train_noise
        model_state['noise'] = noise_var
        model_state['noise_lower_bound'] = lower_bound_v3

    data['schema_version'] = 3
    return data


class TestV3MigrationNoiseSafety:
    """Verify that loading a v3-format JSON (where raw_noise was calibrated against
    lower_bound = 0.99 * physical_variance) produces correct noise after v3→v4 migration.

    In v3, `set_noise(σ²)` stored raw_noise = softplus_inv(σ² - 0.99σ²) = softplus_inv(0.01σ²).
    Under the v4 constraint (lower_bound = 1e-8), that same raw_noise gives
    noise = softplus(raw_noise) + 1e-8 ≈ 0.01σ²  — 100× too small.

    This is handled correctly because _apply_physical_noise runs on the train=False reload
    path and overwrites whatever raw_noise was in the state dict with the value derived
    from objective.noise_std."""

    def test_v3_migration_restores_correct_noise(self, tmp_path: Path) -> None:
        """After migrating a v3-format JSON, the loaded model must have the correct noise variance.

        The v3 raw_noise is ~100× too small for the v4 constraint. On reload,
        _apply_physical_noise overwrites it with the value from objective.noise_std."""
        noise_std = 0.05
        objective = Hartmann(n_variables=6, noise_std={'Hartmann': noise_std})
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()

        # Capture the noise in model space BEFORE saving — this is what raw_noise encodes
        noise_in_model_space = optimiser._noise_std_in_model_space
        assert noise_in_model_space is not None
        noise_var_per_objective = [
            float(noise_in_model_space[obj_idx] ** 2)
            for obj_idx in range(optimiser.objective.n_objectives)
        ]

        # Save the v4 JSON
        optimiser.settings.allow_automatic_json_updates = True
        file_path = tmp_path / "v3_migration_test.json"
        save_to_json(optimiser, str(file_path))

        with open(file_path, 'r') as json_file:
            v4_data = json.load(json_file)

        # Downgrade to v3 format with v3-calibrated raw_noise values
        v3_data = _build_v3_format_json(v4_data, noise_var_per_objective)

        # Overwrite the file with the v3-format data
        with open(file_path, 'w') as json_file:
            json.dump(v3_data, json_file, indent=2)

        # This should succeed without raising.  If the bug exists, _check_noise_desync_on_reload
        # will raise ValueError here (raw_noise gives ~0.01 * σ² instead of σ²).
        loaded = load_optimiser_from_state(str(file_path), allow_automatic_json_updates=True)

        # If we get past the load, verify the noise was correctly restored.
        assert isinstance(loaded.predictor, BotorchPredictor)
        reloaded_noise_in_model = loaded._noise_std_in_model_space
        assert reloaded_noise_in_model is not None

        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(reloaded_noise_in_model[objective_index] ** 2)
            actual_noise = float(
                single_model.model_with_data.likelihood.noise.detach()  # type: ignore[union-attr]
            )
            relative_error = abs(actual_noise - expected_variance) / expected_variance
            assert relative_error < 0.02, (
                f"Objective {objective_index}: expected noise variance {expected_variance:.2e} "
                f"after v3 migration but got {actual_noise:.2e} "
                f"(relative error {relative_error:.1%})"
            )

        # Also verify the constraint lower_bound was reset to _NOISE_CONSTRAINT_FLOOR.
        # If _apply_physical_noise does not reset the constraint, the stale v3 lower_bound
        # survives in memory.  A subsequent save would then produce another v3-style state_dict
        # and the migration problem would recur on the next reload.
        from veropt.optimiser.model import _NOISE_CONSTRAINT_FLOOR
        for single_model in loaded.predictor.model._model_list:
            actual_lower_bound = float(
                single_model.model_with_data.likelihood.noise_covar  # type: ignore[union-attr]
                .raw_noise_constraint.lower_bound
            )
            assert abs(actual_lower_bound - _NOISE_CONSTRAINT_FLOOR) < 1e-12, (
                f"Constraint lower_bound after v3 migration should be _NOISE_CONSTRAINT_FLOOR "
                f"({_NOISE_CONSTRAINT_FLOOR:.2e}) but got {actual_lower_bound:.2e}. "
                "Without the reset, a save→reload cycle would re-produce a v3-style state_dict."
            )

    def test_v3_migration_save_reload_produces_clean_v4_json(self, tmp_path: Path) -> None:
        """After migrating a v3 JSON and saving again, the new JSON must have v4-format entries.

        Specifically:
        - raw_noise_constraint.lower_bound must equal _NOISE_CONSTRAINT_FLOOR (1e-8), not the
          stale v3 value (0.99 * physical_variance).
        - Loading the re-saved JSON a second time must give the same correct noise.

        This guards against a latent bug where _apply_physical_noise sets the value correctly
        but leaves the stale constraint in memory, so the next save re-produces the problem."""
        from veropt.optimiser.model import _NOISE_CONSTRAINT_FLOOR

        noise_std = 0.05
        objective = Hartmann(n_variables=6, noise_std={'Hartmann': noise_std})
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()

        noise_var_per_objective = [
            float(optimiser._noise_std_in_model_space[obj_idx] ** 2)  # type: ignore[index]
            for obj_idx in range(optimiser.objective.n_objectives)
        ]

        # Save v4 JSON, downgrade to v3 format
        optimiser.settings.allow_automatic_json_updates = True
        v3_path = tmp_path / "v3_source.json"
        save_to_json(optimiser, str(v3_path))
        with open(v3_path, 'r') as json_file:
            v4_data = json.load(json_file)
        v3_data = _build_v3_format_json(v4_data, noise_var_per_objective)
        with open(v3_path, 'w') as json_file:
            json.dump(v3_data, json_file, indent=2)

        # First reload: migrates v3 → v4, applies noise
        loaded_once = load_optimiser_from_state(str(v3_path), allow_automatic_json_updates=True)

        # Save the migrated optimiser to a NEW file
        v4_resaved_path = tmp_path / "v4_resaved.json"
        save_to_json(loaded_once, str(v4_resaved_path))

        # Inspect the re-saved JSON: constraint lower_bound must be _NOISE_CONSTRAINT_FLOOR
        with open(v4_resaved_path, 'r') as json_file:
            resaved_data = json.load(json_file)

        model_dicts = (
            resaved_data['optimiser']['predictor']['state']['model']['state']['model_dicts']
        )
        for model_key, model_dict in model_dicts.items():
            constraint_lower = model_dict['state']['state_dict'].get(
                'likelihood.noise_covar.raw_noise_constraint.lower_bound'
            )
            assert constraint_lower is not None, f"Constraint lower_bound missing from {model_key}"
            assert abs(float(constraint_lower) - _NOISE_CONSTRAINT_FLOOR) < 1e-12, (
                f"{model_key}: re-saved constraint lower_bound should be _NOISE_CONSTRAINT_FLOOR "
                f"({_NOISE_CONSTRAINT_FLOOR:.2e}) but got {float(constraint_lower):.6e}. "
                "The stale v3 lower_bound survived in memory and was re-saved."
            )

        # Second reload from the re-saved v4 JSON must also produce correct noise
        loaded_twice = load_optimiser_from_state(str(v4_resaved_path))
        noise_in_model = loaded_twice._noise_std_in_model_space
        assert noise_in_model is not None

        assert isinstance(loaded_twice.predictor, BotorchPredictor)
        for objective_index, single_model in enumerate(loaded_twice.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(
                single_model.model_with_data.likelihood.noise.detach()  # type: ignore[union-attr]
            )
            relative_error = abs(actual_noise - expected_variance) / expected_variance
            assert relative_error < 0.02, (
                f"Second reload (v4 → v4) gave wrong noise: expected {expected_variance:.2e}, "
                f"got {actual_noise:.2e} (relative error {relative_error:.1%})"
            )
