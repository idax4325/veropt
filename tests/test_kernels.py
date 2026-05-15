import pytest
import torch

from veropt import bayesian_optimiser
from veropt.optimiser.kernels import MaternKernel, DoubleMaternKernel
from veropt.optimiser.practice_objectives import Hartmann


def test_run_optimisation_step_rq_matern_kernel() -> None:

    # Just an integration test to ensure this kernel runs
    #   - Could probably make this more minimal

    n_initial_points = 4
    n_bayesian_points = 32

    n_evalations_per_step = 4

    objective = Hartmann(
        n_variables=6
    )

    optimiser = bayesian_optimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evalations_per_step,
        objective=objective,
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            },
            "kernels": "rational_quadratic_and_matern",
            "kernel_settings": {
                "alpha_upper_bound": 0.01,
                "alpha_lower_bound": 0.00001,
                "rq_lengthscale_lower_bound": 0.1,
                "rq_lengthscale_upper_bound": 2.0,
                "matern_lengthscale_lower_bound": 0.1,
                "matern_lengthscale_upper_bound": 2.0
            }
        },
        acquisition_optimiser={
            'optimiser': 'dual_annealing',
            'optimiser_settings': {
                'max_iter': 50
            }
        }
    )

    for i in range(3):
        optimiser.run_optimisation_step()


def test_train_noise_stored_on_kernel() -> None:
    """train_noise=True is stored on the kernel and accessible via .train_noise."""
    kernel = MaternKernel(n_variables=3, train_noise=True)
    assert kernel.train_noise is True


def test_train_noise_defaults_to_false() -> None:
    kernel = MaternKernel(n_variables=3)
    assert kernel.train_noise is False


def test_legacy_kernel_state_raises_error() -> None:
    """Loading a v1-format state dict (noise inside 'settings') must raise an error —
    the schema gate in load_optimiser_from_state is the only supported upgrade path."""

    legacy_state = {
        'n_variables': 2,
        'settings': {
            'lengthscale_lower_bound': 0.1,
            'lengthscale_upper_bound': 2.0,
            'nu': 2.5,
            'noise': 1e-08,
            'noise_lower_bound': 1e-08,
            'train_noise': False,
        },
        'state_dict': {},
        'train_inputs': [],
        'train_targets': [],
    }

    with pytest.raises((KeyError, AssertionError)):
        MaternKernel.from_saved_state(legacy_state)


def test_noise_in_kernel_settings_raises_error() -> None:
    """Passing noise fields inside kernel_settings must raise an AssertionError via _validate_typed_dict."""

    from veropt.optimiser.constructors import gpytorch_single_model

    with pytest.raises(AssertionError, match="noise"):
        gpytorch_single_model(
            n_variables=2,
            kernel='matern',
            settings={'noise': 1e-4}  # type: ignore[arg-type]
        )
