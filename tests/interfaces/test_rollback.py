"""
Integration tests for the experiment rollback utility.

These tests run a fake-submission experiment for several steps, snapshot the state
at an intermediate point, run more steps, roll back, and verify that the restored
state matches the earlier snapshot.

The FakeSubmitBatchManager is used so no real SLURM queue is needed.

Notes on rollback semantics
----------------------------
After rollback to target_point:
  - experimental state has target_point points
  - optimiser has (target_point - n_evals_per_step) evaluated points
  - evaluated_objectives.json / suggested_parameters.json contain last batch data
  - state.just_rebuilt = True (if target_point > 0)

On first resumed run_experiment_step:
  - just_rebuilt skips wait/collection
  - _load_latest_points adds last batch → optimiser reaches target_point
  - model trains → new candidates suggested → next batch submitted
"""

import json
import os
import tempfile

import pytest

from veropt.interfaces.batch_manager import FakeSubmitBatchManager
from veropt.interfaces.experiment import Experiment
from veropt.interfaces.experiment_utility import ExperimentConfig, ExperimentalState, PathManager
from veropt.interfaces.local_simulation import MockSimulationConfig, MockSimulationRunner
from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.rollback import rollback_experiment
from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_experiment_config(tmp_dir: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="rollback_test",
        parameter_names=["param1"],
        parameter_bounds={"param1": [0.0, 1.0]},
        path_to_experiment=tmp_dir,
        experiment_mode="local_slurm",  # type: ignore[arg-type]
        output_filename="output",
    )


def _make_optimiser_config() -> dict:
    return dict(
        n_initial_points=4,
        n_bayesian_points=4,
        n_evaluations_per_step=2,
        model={"training_settings": {"max_iter": 5}},
    )


def _make_simulation_runner(tmp_dir: str) -> MockSimulationRunner:
    config = MockSimulationConfig()
    config.output_filename = "output"
    config.output_directory = tmp_dir
    return MockSimulationRunner(config=config)


def _make_result_processor() -> MockResultProcessor:
    # fixed_objective=False so the result processor reads from the output file written by
    # FakeSubmitBatchManager (which writes distinct float values per point), giving the GP
    # enough variance to avoid a degenerate covariance matrix.
    return MockResultProcessor(
        objective_names=["obj1"],
        objectives={"obj1": 1.0},
        fixed_objective=False,
    )


def _save_experiment_config(experiment_config: ExperimentConfig, tmp_dir: str) -> str:
    config_path = os.path.join(tmp_dir, "experiment_config.json")
    with open(config_path, "w") as config_file:
        json.dump(experiment_config.model_dump(), config_file)
    return config_path


def _run_n_steps(experiment: Experiment, n_steps: int) -> None:
    for _step in range(n_steps):
        experiment.run_experiment_step()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_rollback_state_matches_earlier_snapshot() -> None:
    """
    Run to step 2, snapshot state. Run 2 more steps. Roll back to step 2.
    Verify n_points in state and optimiser both match the snapshot.
    """
    n_evals_per_step = 2
    snapshot_after_n_steps = 2
    extra_steps = 2
    target_point = snapshot_after_n_steps * n_evals_per_step  # = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=snapshot_after_n_steps)
        n_points_at_snapshot = experiment.n_points_evaluated  # should be target_point

        _run_n_steps(experiment, n_steps=extra_steps)
        assert experiment.n_points_evaluated > n_points_at_snapshot

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=target_point,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)

        restored_state = ExperimentalState.load(path_manager.experimental_state_json)
        assert restored_state.next_point == target_point
        assert len(restored_state.points) == target_point

        # Optimiser should have target_point - n_evals_per_step points (last batch in messenger)
        restored_optimiser = load_optimiser_from_state(path_manager.optimiser_state_json)
        assert restored_optimiser.n_points_evaluated == target_point - n_evals_per_step

        # All expected point indices should still be present in state
        for point_no in range(target_point):
            assert point_no in restored_state.points

        # Points beyond target should be gone
        for point_no in range(target_point, target_point + n_evals_per_step * extra_steps):
            assert point_no not in restored_state.points


def test_rollback_to_zero() -> None:
    """Rolling back to point 0 should produce an empty state and a fresh optimiser."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=2)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=0,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)
        restored_state = ExperimentalState.load(path_manager.experimental_state_json)

        assert restored_state.next_point == 0
        assert len(restored_state.points) == 0
        assert not restored_state.just_rebuilt  # At step 0, current_step==0 is sufficient

        restored_optimiser = load_optimiser_from_state(path_manager.optimiser_state_json)
        assert restored_optimiser.n_points_evaluated == 0


def test_rollback_creates_backups() -> None:
    """Rollback should create timestamped backup files for both the state and optimiser JSONs."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=2)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=0,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)
        experiment_dir = path_manager.experiment_directory

        all_files = os.listdir(experiment_dir)
        backup_files = [f for f in all_files if "_backup_" in f and f.endswith(".json")]

        # Expect one backup for state JSON and one for optimiser JSON
        assert len(backup_files) >= 2, (
            f"Expected at least 2 backup files (state + optimiser), found: {backup_files}"
        )


def test_rollback_renames_simulation_directories() -> None:
    """With handling='rename', rolled-back point directories should be renamed."""

    n_evals_per_step = 2
    n_steps_to_run = 3
    target_point = n_evals_per_step  # keep only the first batch

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=n_steps_to_run)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=target_point,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)
        results_dir = path_manager.results_directory

        # Directories for kept points should still exist unchanged
        for point_no in range(target_point):
            simulation_id = PathManager.make_simulation_id(point_no=point_no)
            assert os.path.isdir(os.path.join(results_dir, simulation_id)), (
                f"Directory for kept point {point_no} should still exist."
            )

        # Directories for rolled-back points should have been renamed
        for point_no in range(target_point, n_evals_per_step * n_steps_to_run):
            simulation_id = PathManager.make_simulation_id(point_no=point_no)
            original_path = os.path.join(results_dir, simulation_id)
            renamed_path = original_path + "_rolled_back"

            assert not os.path.isdir(original_path), (
                f"Original directory for rolled-back point {point_no} should no longer exist."
            )
            assert os.path.isdir(renamed_path), (
                f"Renamed directory for rolled-back point {point_no} should exist."
            )


def test_rollback_force_deletes_simulation_directories() -> None:
    """With handling='force_delete', rolled-back directories should be deleted without prompting."""

    n_evals_per_step = 2
    n_steps_to_run = 2
    target_point = n_evals_per_step

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=n_steps_to_run)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=target_point,
            simulation_folder_handling="force_delete",
        )

        path_manager = PathManager(experiment_config)
        results_dir = path_manager.results_directory

        for point_no in range(target_point, n_evals_per_step * n_steps_to_run):
            simulation_id = PathManager.make_simulation_id(point_no=point_no)
            assert not os.path.isdir(os.path.join(results_dir, simulation_id)), (
                f"Deleted directory for point {point_no} should no longer exist."
            )


def test_rollback_invalid_target_point_not_on_batch_boundary() -> None:
    """Rolling back to a point that is not on a batch boundary should raise AssertionError."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=2)

        with pytest.raises(AssertionError, match="batch boundary"):
            rollback_experiment(
                experiment_config_path=experiment_config_path,
                target_point=1,  # n_evals_per_step=2, so 1 is not a valid boundary
                simulation_folder_handling="rename",
            )


def test_rollback_forward_raises() -> None:
    """Rolling forward (target_point > n_evaluated) should raise an AssertionError."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=2)

        with pytest.raises(AssertionError):
            rollback_experiment(
                experiment_config_path=experiment_config_path,
                target_point=1000,
                simulation_folder_handling="rename",
            )


def test_rollback_sets_just_rebuilt_flag() -> None:
    """After rolling back to target_point > 0, just_rebuilt should be True in the state."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=3)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=2,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)
        restored_state = ExperimentalState.load(path_manager.experimental_state_json)

        assert restored_state.just_rebuilt is True, (
            "just_rebuilt should be True after rollback to a non-zero target so the "
            "first resumed step skips wait_for_jobs."
        )


def test_rollback_messenger_files_contain_last_batch() -> None:
    """
    After rollback to target_point > 0, evaluated_objectives.json should contain
    exactly n_evaluations_per_step entries for the last kept batch.
    """
    n_evals_per_step = 2
    target_point = n_evals_per_step * 2  # = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(tmp_dir),
            result_processor=_make_result_processor(),
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=4)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=target_point,
            simulation_folder_handling="rename",
        )

        path_manager = PathManager(experiment_config)
        with open(path_manager.evaluated_objectives_json, "r") as objectives_file:
            evaluated_objectives = json.load(objectives_file)

        for obj_name, values in evaluated_objectives.items():
            assert len(values) == n_evals_per_step, (
                f"Messenger file should have {n_evals_per_step} entries for '{obj_name}', "
                f"got {len(values)}."
            )


def test_experiment_can_continue_after_rollback() -> None:
    """
    After a rollback, the experiment should be resumable and run at least one more
    step without errors. New parameters should be submitted after the first step.
    """
    n_evals_per_step = 2
    snapshot_after_n_steps = 2
    extra_steps = 2
    target_point = snapshot_after_n_steps * n_evals_per_step  # = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_config = _make_experiment_config(tmp_dir)
        experiment_config_path = _save_experiment_config(experiment_config, tmp_dir)

        with open(os.path.join(tmp_dir, "output.txt"), "w") as output_file:
            output_file.write("0.5")

        simulation_runner = _make_simulation_runner(tmp_dir)
        result_processor = _make_result_processor()

        experiment = Experiment.from_the_beginning(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        _run_n_steps(experiment, n_steps=snapshot_after_n_steps + extra_steps)

        rollback_experiment(
            experiment_config_path=experiment_config_path,
            target_point=target_point,
            simulation_folder_handling="rename",
        )

        # Resume — must not crash and must submit new parameters
        resumed_experiment = Experiment.continue_if_possible(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager,
        )

        resumed_experiment.run_experiment_step()

        # After the first resumed step, new parameters should have been submitted
        assert resumed_experiment.n_points_submitted > target_point, (
            "After resuming from rollback, new parameters should have been submitted."
        )
