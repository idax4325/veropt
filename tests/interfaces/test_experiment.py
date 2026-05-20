import json
import os
import tempfile

import pytest

from veropt.optimiser.saver_loader_utility import rehydrate_object
from veropt.optimiser.objective import Objective
from veropt.interfaces.batch_manager import FakeSubmitBatchManager
from veropt.interfaces.experiment import Experiment, _mask_nans
from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.experiment_utility import (
    ExperimentConfig, ExperimentalState, PathManager, Point, ExperimentObjective
)

import numpy as np


# ── Helpers shared by version-change tests ─────────────────────────────────────

def _make_version_experiment_config(tmp_dir: str, version: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="version_test",
        version=version,
        parameter_names=["param1"],
        parameter_bounds={"param1": [0.0, 1.0]},
        path_to_experiment=tmp_dir,
        experiment_mode="local_slurm",  # type: ignore[arg-type]
        output_filename="output",
    )


def _make_version_optimiser_config(n_bayesian: int = 4) -> dict:
    return dict(
        n_initial_points=4,
        n_bayesian_points=n_bayesian,
        n_evaluations_per_step=2,
        model={"training_settings": {"max_iter": 5}},
    )


def _make_version_simulation_runner(tmp_dir: str) -> MockSimulationRunner:
    config = MockSimulationConfig()
    config.output_filename = "output"
    config.output_directory = tmp_dir
    return MockSimulationRunner(config=config)


def _make_version_result_processor() -> MockResultProcessor:
    return MockResultProcessor(
        objective_names=["obj1"],
        objectives={"obj1": 1.0},
        fixed_objective=False,
    )


def _save_config(experiment_config: ExperimentConfig, tmp_dir: str, name: str) -> str:
    config_path = os.path.join(tmp_dir, name)
    with open(config_path, "w") as config_file:
        json.dump(experiment_config.model_dump(), config_file)
    return config_path


def _run_n_steps(experiment: Experiment, n_steps: int) -> None:
    for _step in range(n_steps):
        experiment.run_experiment_step()


def test_experiment_objective() -> None:
    bounds_lower, bounds_upper = [0.1], [10.0]
    n_variables, n_objectives = 1, 1
    variable_names, objective_names = ["var1"], ["obj1"]
    suggested_parameters_json = "suggested.json"
    evaluated_objectives_json = "evaluated.json"

    experiment_objective = ExperimentObjective(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        n_variables=n_variables,
        n_objectives=n_objectives,
        objective_names=objective_names,
        variable_names=variable_names,
        suggested_parameters_json=suggested_parameters_json,
        evaluated_objectives_json=evaluated_objectives_json
    )

    saved_state = experiment_objective.gather_dicts_to_save()
    rehydrated_experiment_objective = rehydrate_object(
        superclass=Objective,
        name=saved_state["name"],
        saved_state=saved_state["state"]
    )

    assert isinstance(rehydrated_experiment_objective, ExperimentObjective)


def test_mask_nans() -> None:

    experimental_state = ExperimentalState.make_fresh_state(
        experiment_name="",
        experiment_directory=""
    )

    values = [20.0, 0.5, 17.0, -0.01, -13.0]
    initial_dict_of_objectives = {i: {"obj1": value} for i, value in enumerate(values)}

    for objective_values in initial_dict_of_objectives.values():
        point = Point(
            parameters={"param1": 0.1},
            state="",
            objective_values=objective_values  # type: ignore[arg-type]  # mypy silliness
        )

        experimental_state.update(point)

    new_values = [5.0, float("nan"), -5.0]
    dict_of_objectives = {i: {"obj1": value} for i, value in enumerate(new_values)}

    for objective_values in dict_of_objectives.values():
        point = Point(
            parameters={"param1": 0.1},
            state="",
            objective_values=objective_values  # type: ignore[arg-type]  # mypy silliness
        )

        experimental_state.update(point)

    updated_dict_of_objectives = _mask_nans(
        dict_of_objectives=dict_of_objectives,
        experimental_state=experimental_state
    )

    assert not np.isnan(updated_dict_of_objectives[1]["obj1"])

    new_experimental_state = ExperimentalState.make_fresh_state(
        experiment_name="",
        experiment_directory=""
    )

    with pytest.raises(AssertionError):
        _mask_nans(
            dict_of_objectives=dict_of_objectives,  # type: ignore[assignment]  # mypy silliness
            experimental_state=new_experimental_state
        )


def test_experiment_step() -> None:

    optimiser_config = dict(
        n_initial_points=3,
        n_bayesian_points=1,
        n_evaluations_per_step=1
    )

    experiment_config = ExperimentConfig(
        experiment_name="integration_test",
        parameter_names=["param1"],
        parameter_bounds={"param1": [0, 1]},
        path_to_experiment="path/to/experiment",
        experiment_mode="local",  # type: ignore[arg-type]  # pydantic casts the string to ExperimentMode internally
        run_script_filename="test_experiment",
        output_filename="output"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_script_root_directory = tmp_dir
        run_script_filename = "foo"
        run_script = f"{run_script_root_directory}/{run_script_filename}.txt"

        with open(run_script, "w+") as f:
            f.write("bar")

        experiment_config.path_to_experiment = tmp_dir
        experiment_config.run_script_root_directory = tmp_dir
        experiment_config.run_script_filename = run_script_filename

        objective_names = ["obj1"]
        objectives = {"obj1": 1.}

        simulation_config = MockSimulationConfig()
        simulation_config.output_filename = "output"
        simulation_config.output_directory = tmp_dir
        simulation_runner = MockSimulationRunner(config=simulation_config)

        with open(f"{tmp_dir}/output.txt", "w+") as f:
            f.write("0.1")

        result_processor = MockResultProcessor(
            objective_names=objective_names,
            objectives=objectives)

        experiment = Experiment.from_the_beginning(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=optimiser_config
        )

        experiment.run_experiment_step()

        assert experiment.state.points[0].objective_values["obj1"] == 0.1  # type: ignore


def test_experiment() -> None:

    # TODO: make a test for integrated experiment
    #       need better MockSimulation and MockResultProcessor for this

    pass


# ── continue_with_new_version tests ────────────────────────────────────────────

def test_continue_with_new_version_no_phantom_points() -> None:
    """
    After running an experiment to completion and calling continue_with_new_version,
    the resulting experiment should have no 'phantom' points: state.n_points must
    equal n_points_evaluated (no gap between submitted-but-unevaluated indices and
    the first new-version index).

    Background: the last run_experiment_step_submitted call in the old experiment
    suggests a new batch and registers it in state (via get_parameters_from_optimiser)
    but then skips submission because current_step == n_total_steps.  Those phantom
    points inflate state.next_point, so new-version points start at the wrong index.
    """
    n_evals_per_step = 2
    n_initial = 4
    n_bayesian_old = 4
    n_total_old_steps = (n_initial + n_bayesian_old) // n_evals_per_step + 1  # = 5

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        old_config = _make_version_experiment_config(tmp_dir, version="v1")
        new_config = _make_version_experiment_config(tmp_dir, version="v2")

        simulation_runner = _make_version_simulation_runner(tmp_dir)
        result_processor = _make_version_result_processor()

        experiment_v1 = Experiment.from_the_beginning(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=old_config,
            optimiser_config=_make_version_optimiser_config(n_bayesian=n_bayesian_old),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment_v1, n_steps=n_total_old_steps)

        n_old_evaluated = experiment_v1.n_points_evaluated

        experiment_v2 = Experiment.continue_with_new_version(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            old_experiment_config=old_config,
            new_experiment_config=new_config,
            optimiser_config=_make_version_optimiser_config(n_bayesian=8),
            batch_manager_class=FakeSubmitBatchManager,
        )

        # After the version change, state.n_points must equal n_points_evaluated:
        # any phantom points from the old experiment's last unsubmitted step must
        # have been stripped, so new-version indices start immediately after the
        # last evaluated point.
        assert experiment_v2.state.n_points == n_old_evaluated, (
            f"Expected state.n_points == n_old_evaluated ({n_old_evaluated}), "
            f"got state.n_points={experiment_v2.state.n_points}. "
            f"Likely cause: phantom points from the old experiment's final "
            f"unsubmitted batch are still in state."
        )

        # state.next_point must also agree
        assert experiment_v2.state.next_point == n_old_evaluated, (
            f"state.next_point={experiment_v2.state.next_point} but expected {n_old_evaluated}."
        )


def test_continue_with_new_version_contiguous_indices_after_new_steps() -> None:
    """
    After continue_with_new_version and two new steps, all point indices in state
    must be contiguous (no gaps) and n_points_evaluated must not double-count the
    last replayed batch.

    Two specific bugs this catches:
    1. Phantom-point gap: old experiment's last unsubmitted batch leaves indices 8-9
       in state; new-version points start at 10, creating a gap.
    2. Double-load: after replay, evaluated_objectives.json still holds the last
       batch; the first run_optimisation_step re-reads it, so n_points_evaluated
       overshoots by n_evals_per_step.
    """
    n_evals_per_step = 2
    n_initial = 4
    n_bayesian_old = 4
    n_total_old_steps = (n_initial + n_bayesian_old) // n_evals_per_step + 1  # = 5
    n_new_steps = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        old_config = _make_version_experiment_config(tmp_dir, version="v1")
        new_config = _make_version_experiment_config(tmp_dir, version="v2")

        simulation_runner = _make_version_simulation_runner(tmp_dir)
        result_processor = _make_version_result_processor()

        experiment_v1 = Experiment.from_the_beginning(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=old_config,
            optimiser_config=_make_version_optimiser_config(n_bayesian=n_bayesian_old),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment_v1, n_steps=n_total_old_steps)
        n_old_evaluated = experiment_v1.n_points_evaluated

        experiment_v2 = Experiment.continue_with_new_version(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            old_experiment_config=old_config,
            new_experiment_config=new_config,
            optimiser_config=_make_version_optimiser_config(n_bayesian=8),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment_v2, n_steps=n_new_steps)

        all_indices = sorted(experiment_v2.state.points.keys())

        # Indices must be contiguous starting from 0
        expected_indices = list(range(experiment_v2.state.n_points))
        assert all_indices == expected_indices, (
            f"Point indices are not contiguous. "
            f"Found gaps or duplicates: {all_indices} (expected {expected_indices}). "
            f"This can happen when phantom points from the old experiment's last "
            f"unsubmitted batch leave holes at indices {n_old_evaluated} and "
            f"{n_old_evaluated + n_evals_per_step - 1}."
        )

        # The 1-batch-lag invariant: optimiser has evaluated all but the last
        # submitted batch.  After n_new_steps steps the 1-batch lag is restored.
        expected_n_evaluated = experiment_v2.state.n_points - n_evals_per_step
        assert experiment_v2.n_points_evaluated == expected_n_evaluated, (
            f"n_points_evaluated={experiment_v2.n_points_evaluated} but expected "
            f"{expected_n_evaluated} (= state.n_points - n_evals_per_step). "
            f"If it is {n_old_evaluated + n_evals_per_step} instead of {n_old_evaluated}, "
            f"the last replayed batch was loaded twice by the first run_optimisation_step."
        )

        # New-version point directories must exist for all submitted new points.
        # The last batch in state was submitted but not collected, so we only check
        # up to (state.n_points - n_evals_per_step).
        path_manager_v2 = PathManager(new_config)
        results_dir = path_manager_v2.results_directory
        for point_no in range(n_old_evaluated, experiment_v2.state.n_points - n_evals_per_step):
            sim_id = PathManager.make_simulation_id(point_no=point_no, version="v2")
            assert os.path.isdir(os.path.join(results_dir, sim_id)), (
                f"Expected result directory '{sim_id}' for new-version point {point_no}."
            )
