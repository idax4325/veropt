import tempfile
from typing import Optional

from veropt.interfaces.batch_manager import FakeSubmitBatchManager
from veropt.interfaces.experiment import Experiment
from veropt.interfaces.experiment_utility import ExperimentConfig
from veropt.interfaces.local_simulation import MockSimulationConfig, MockSimulationRunner
from veropt.interfaces.result_processing import MockResultProcessor


# --- Helpers ---

def _make_experiment_config(
        path_to_experiment: str,
        version: Optional[str] = None
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_slurm",
        version=version,
        parameter_names=["param1"],
        parameter_bounds={"param1": [0.0, 1.0]},
        path_to_experiment=path_to_experiment,
        experiment_mode="local_slurm",  # type: ignore[arg-type]  # pydantic casts string internally
        output_filename="output"
    )


def _make_result_processor() -> MockResultProcessor:
    # fixed_objective=False so the processor reads the output file written by FakeSubmitBatchManager.
    # Each point's file contains float(point_no + 1), giving unique per-point objective values.
    return MockResultProcessor(
        objective_names=["obj1"],
        objectives={"obj1": 1.0},
        fixed_objective=False
    )


def _make_optimiser_config() -> dict:
    # Small iteration counts so tests run quickly
    return dict(
        n_initial_points=4,
        n_bayesian_points=4,
        n_evaluations_per_step=4,
        verbose=False,
        model={"training_settings": {"max_iter": 5}},
        acquisition_optimiser={
            "optimiser": "dual_annealing",
            "optimiser_settings": {"max_iter": 5}
        }
    )


def _make_simulation_runner() -> MockSimulationRunner:
    return MockSimulationRunner(config=MockSimulationConfig())


# --- Tests ---

def test_submit_experiment_submits_expected_batches() -> None:
    """A fresh submit-mode experiment should run end-to-end and submit the right number of batches."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            experiment_config=_make_experiment_config(tmp_dir),
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager
        )

        experiment.run_experiment()

        fake_batch_manager = experiment.batch_manager
        assert isinstance(fake_batch_manager, FakeSubmitBatchManager)

        # n_total_steps = (4 + 4) // 4 + 1 = 3
        # Step 0 submits initial batch, step 1 submits bayesian batch, step 2 is final (no submission)
        assert fake_batch_manager.n_batches_submitted == 2
        assert fake_batch_manager.n_jobs_submitted == 8
        assert experiment.n_points_evaluated == 8


def test_new_version_experiment_processes_pending_jobs() -> None:
    """
    After continue_with_new_version, running the experiment should:
      1. Skip waiting on the first new step (just_rebuilt=True) - no new-version jobs are pending yet.
      2. Submit its own new bayesian jobs on that first step.
      3. On the second step (just_rebuilt=False), correctly wait for and process those new jobs.

    We use n_bayesian_points=8 so the new version has 2 remaining steps after the rebuild
    (n_total_steps=4, old experiment leaves current_step=2, so n_remaining=2). This gives
    one step to submit and one step to wait.

    This test exposes the bug where just_rebuilt is never reset to False, causing
    wait_for_jobs to be skipped on every step - meaning the new version's own submitted
    jobs are never processed.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        old_config = _make_experiment_config(tmp_dir, version=None)
        new_config = _make_experiment_config(tmp_dir, version="v2")

        # n_bayesian=8 gives n_total_steps = (4+8)//4 + 1 = 4
        optimiser_config = _make_optimiser_config()
        optimiser_config["n_bayesian_points"] = 8

        # --- Build and partially run old experiment (2 of 4 steps) ---

        old_experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            experiment_config=old_config,
            optimiser_config=optimiser_config,
            batch_manager_class=FakeSubmitBatchManager
        )

        old_experiment.run_experiment_step()  # Step 0: submits 4 initial points
        old_experiment.run_experiment_step()  # Step 1: processes initial, submits 4 bayesian

        assert old_experiment.n_points_submitted == 8
        assert old_experiment.n_points_evaluated == 4

        # --- Create new version, rebuilding from the 4 evaluated initial points ---

        new_experiment = Experiment.continue_with_new_version(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            old_experiment_config=old_config,
            new_experiment_config=new_config,
            optimiser_config=optimiser_config,
            batch_manager_class=FakeSubmitBatchManager
        )

        assert new_experiment.n_points_evaluated == 4   # initial points re-evaluated
        assert new_experiment.n_points_submitted == 8   # old state carried over
        assert new_experiment.state.just_rebuilt is True

        # --- Run the 2 remaining steps of the new experiment ---
        # Step 1 (just_rebuilt=True):  skip waiting, suggest + submit 4 new bayesian points
        # Step 2 (just_rebuilt=False): wait for those new jobs, process, make final suggestions

        new_experiment.run_experiment()

        new_fake_bm = new_experiment.batch_manager
        assert isinstance(new_fake_bm, FakeSubmitBatchManager)

        # New version should have submitted exactly 1 new batch (its own step 1)
        assert new_fake_bm.n_batches_submitted == 1
        assert new_fake_bm.n_jobs_submitted == 4

        # The new version's own jobs (points 8-11) should have been waited for on step 2.
        # With the bug (just_rebuilt never resets), step 2 also skips wait_for_jobs,
        # so these points stay "Simulation started" and this assertion fails.
        assert new_experiment.state.points[8].state == "Simulation completed"
        assert new_experiment.state.points[11].state == "Simulation completed"
