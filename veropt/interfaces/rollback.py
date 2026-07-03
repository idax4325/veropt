"""
Rollback utility for recovering experiments that crashed mid-optimisation.

Rolls back the experimental state and optimiser JSON files to a requested
target point, handles simulation result directories, and writes the correct
messenger files so the experiment can be cleanly resumed.

How resuming works after rollback
----------------------------------
The submitted-experiment loop (run_experiment_step_submitted) always:
  1. Waits for the *previous* batch's jobs → collects results
  2. Calls run_optimisation_step → _load_latest_points (reads evaluated_objectives.json)
     → suggests next candidates

After a rollback to `target_point` the state is wired as follows so the loop
resumes correctly on the first call to run_experiment_step:

  * experimental_state.just_rebuilt = True  (skip step 1 — no pending jobs)
  * optimiser has (target_point - n_evals_per_step) evaluated points
  * evaluated_objectives.json / suggested_parameters.json contain the last
    batch (points target_point - n_evals .. target_point - 1)

On the first resumed step:
  → just_rebuilt skips wait/collect
  → _load_latest_points adds the last batch → optimiser reaches target_point
  → model trains → candidates suggested → next batch submitted

Special case: target_point == 0
  * optimiser has 0 evaluated points
  * evaluated_objectives.json / suggested_parameters.json are empty
  * just_rebuilt = False  (current_step == 0 check is sufficient)
"""

import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import Literal, Optional, Union

from veropt.interfaces.experiment_utility import ExperimentConfig, ExperimentalState, PathManager
from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state


SimulationFolderHandling = Literal["rename", "delete", "force_delete"]


def rollback_experiment(
        experiment_config_path: Union[str, ExperimentConfig],
        target_point: int,
        simulation_folder_handling: SimulationFolderHandling = "rename",
        allow_automatic_json_updates: Optional[bool] = None,
) -> None:
    """
    Roll back an experiment to a given target point.

    Parameters
    ----------
    experiment_config_path:
        Path to the experiment config JSON, or an ExperimentConfig object directly.
    target_point:
        The point index to roll back to. Must be a multiple of n_evaluations_per_step.
        After rollback, the experiment will resume as if only points 0..target_point-1
        have been evaluated. Pass 0 to roll back before any simulations ran.
    simulation_folder_handling:
        What to do with simulation directories for rolled-back points:
        - "rename"       : rename (e.g. point_5 → point_5_rolled_back). Default.
        - "delete"       : ask for confirmation, then delete.
        - "force_delete" : delete without asking.
    allow_automatic_json_updates:
        Passed to load_optimiser_from_state for schema migration.
    """

    # ── 1. Load config and paths ─────────────────────────────────────────────
    experiment_config = ExperimentConfig.load(experiment_config_path)  # type: ignore[arg-type]
    path_manager = PathManager(experiment_config)

    state_json_path = path_manager.experimental_state_json
    optimiser_json_path = path_manager.optimiser_state_json
    evaluated_objectives_json_path = path_manager.evaluated_objectives_json
    suggested_parameters_json_path = path_manager.suggested_parameters_json

    assert os.path.exists(state_json_path), f"Experimental state not found: {state_json_path}"
    assert os.path.exists(optimiser_json_path), f"Optimiser state not found: {optimiser_json_path}"

    # ── 2. Load existing state and validate target_point ─────────────────────
    state = ExperimentalState.load(state_json_path)
    optimiser = load_optimiser_from_state(
        file_name=optimiser_json_path,
        allow_automatic_json_updates=allow_automatic_json_updates,
    )

    n_evaluations_per_step = optimiser.n_evaluations_per_step
    n_points_evaluated = optimiser.n_points_evaluated
    n_points_submitted = state.n_points

    assert target_point >= 0, "target_point must be >= 0."
    assert target_point <= n_points_evaluated, (
        f"target_point ({target_point}) is beyond the number of evaluated points ({n_points_evaluated}). "
        f"Nothing to roll back."
    )
    assert target_point % n_evaluations_per_step == 0, (
        f"target_point ({target_point}) must be a multiple of n_evaluations_per_step "
        f"({n_evaluations_per_step}) so that the rollback lands on a complete batch boundary."
    )

    print(
        f"\nRolling back experiment '{experiment_config.experiment_name}': "
        f"{n_points_evaluated} evaluated / {n_points_submitted} submitted → target {target_point}.\n"
    )

    # ── 3. Back up all JSON files ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _backup_file(state_json_path, suffix=f"_backup_{timestamp}")
    _backup_file(optimiser_json_path, suffix=f"_backup_{timestamp}")

    # ── 4. Handle simulation result directories ───────────────────────────────
    rolled_back_point_nos = list(range(target_point, n_points_submitted))
    _handle_simulation_directories(
        path_manager=path_manager,
        point_nos=rolled_back_point_nos,
        handling=simulation_folder_handling,
        experiment_version=experiment_config.version,
    )

    # ── 5. Patch experimental state ───────────────────────────────────────────
    state.points = {
        point_no: point
        for point_no, point in state.points.items()
        if point_no < target_point
    }
    state.next_point = target_point
    # just_rebuilt = True tells run_experiment_step_submitted to skip wait_for_jobs
    # and results collection on the next call (no pending jobs after a rollback).
    # Not needed at step 0 since current_step == 0 already bypasses those checks.
    state.just_rebuilt = target_point > 0
    _atomic_write_json(state_json_path, state.model_dump())
    print(f"  ✓ Experimental state rolled back to {target_point} points.")

    # ── 6. Patch optimiser JSON ───────────────────────────────────────────────
    # The optimiser is set to (target_point - n_evals_per_step) evaluated points.
    # On the first resumed step, _load_latest_points will add the last batch from
    # evaluated_objectives.json, bringing the optimiser back to target_point.
    with open(optimiser_json_path, "r") as json_file:
        optimiser_dict = json.load(json_file)

    n_points_for_optimiser = max(0, target_point - n_evaluations_per_step)
    _truncate_optimiser_dict(
        optimiser_dict=optimiser_dict,
        n_points_to_keep=n_points_for_optimiser,
        n_evaluations_per_step=n_evaluations_per_step,
    )
    _atomic_write_json(optimiser_json_path, optimiser_dict)
    print(f"  ✓ Optimiser state rolled back to {n_points_for_optimiser} evaluated points.")

    # ── 7. Write messenger files (evaluated_objectives + suggested_parameters) ──
    if target_point == 0:
        _write_empty_messenger_files(
            evaluated_objectives_json_path=evaluated_objectives_json_path,
            suggested_parameters_json_path=suggested_parameters_json_path,
            objective_names=optimiser.objective.objective_names,
            parameter_names=experiment_config.parameter_names,
        )
        print("  ✓ Messenger JSONs reset to empty (target_point=0).")
    else:
        _write_last_batch_messenger_files(
            state=state,
            target_point=target_point,
            n_evaluations_per_step=n_evaluations_per_step,
            evaluated_objectives_json_path=evaluated_objectives_json_path,
            suggested_parameters_json_path=suggested_parameters_json_path,
        )
        print(
            f"  ✓ Messenger JSONs contain last batch "
            f"(points {target_point - n_evaluations_per_step}–{target_point - 1})."
        )

    # ── 8. Verify ─────────────────────────────────────────────────────────────
    _verify_rollback(
        state_json_path=state_json_path,
        optimiser_json_path=optimiser_json_path,
        expected_state_points=target_point,
        expected_optimiser_points=n_points_for_optimiser,
        allow_automatic_json_updates=allow_automatic_json_updates,
    )

    print(
        f"\nRollback complete. Experiment '{experiment_config.experiment_name}' "
        f"is ready to resume from point {target_point} "
        f"(batch {target_point // n_evaluations_per_step})."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _backup_file(path: str, suffix: str) -> None:
    backup_path = path.replace(".json", f"{suffix}.json")
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(mode="rb", dir=dir_name, delete=False, suffix=".tmp") as tmp:
        tmp_path = tmp.name

    shutil.copy2(path, tmp_path)
    os.replace(tmp_path, backup_path)
    print(f"  ✓ Backed up {os.path.basename(path)} → {os.path.basename(backup_path)}")


def _atomic_write_json(path: str, data: dict) -> None:
    dir_name = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=dir_name, delete=False, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        json.dump(data, tmp, indent=2, default=_json_default)
        tmp_path = tmp.name

    os.replace(tmp_path, path)


def _json_default(obj: object) -> object:
    if hasattr(obj, "tolist"):
        return obj.tolist()  # type: ignore[attr-defined]
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _truncate_optimiser_dict(
        optimiser_dict: dict,
        n_points_to_keep: int,
        n_evaluations_per_step: int,
) -> None:
    """
    Truncate evaluated data and suggested_points_history in-place on the raw
    optimiser save dict to n_points_to_keep evaluated points.
    """
    opt = optimiser_dict["optimiser"]

    for key in ("evaluated_variables", "evaluated_objectives"):
        entry = opt[key]
        if isinstance(entry.get("values"), list):
            entry["values"] = entry["values"][:n_points_to_keep]

    # Truncate suggested_points_history: keep only batches that fit within n_points_to_keep.
    history: list = opt.get("suggested_points_history", [])
    if history:
        if n_points_to_keep == 0:
            opt["suggested_points_history"] = []
        else:
            n_batches_to_keep = n_points_to_keep // n_evaluations_per_step
            opt["suggested_points_history"] = history[:n_batches_to_keep]

    # Clear current pending suggestions (stale after rollback)
    opt["suggested_points"] = {}

    if n_points_to_keep == 0:
        # When rolling back to 0, reset all single-model state_dicts to empty so that
        # GPyTorchSingleModel.from_saved_state will NOT call initialise_model_from_state_dict,
        # leaving model_with_data = None. This in turn makes GPyTorchFullModel.__init__
        # set _model = None, so model_has_been_trained = False and from_saved_state for the
        # optimiser will skip the _update_predictor call that would otherwise fail with
        # empty evaluated_variables.
        try:
            model_dicts = opt["predictor"]["state"]["model"]["state"]["model_dicts"]
            for model_key in model_dicts:
                model_dicts[model_key]["state"]["state_dict"] = {}
                model_dicts[model_key]["state"]["train_inputs"] = None
                model_dicts[model_key]["state"]["train_targets"] = None
        except (KeyError, TypeError):
            pass  # Predictor structure may vary; surface loading failures naturally.

        # Normalisers must also be cleared so the optimiser starts in a fully fresh state.
        opt["normaliser_variables"] = None
        opt["normaliser_objectives"] = None

    # Note: for n_points_to_keep > 0, normalisers are intentionally left as-is. They may be
    # slightly stale (fitted on more data than is now present) but are needed by from_saved_state
    # when model_has_been_trained is True. They will be re-fitted automatically on the first
    # resumed step since renormalise_each_step=True when normalise=True.


def _write_empty_messenger_files(
        evaluated_objectives_json_path: str,
        suggested_parameters_json_path: str,
        objective_names: list[str],
        parameter_names: list[str],
) -> None:
    _atomic_write_json(
        evaluated_objectives_json_path,
        {name: [] for name in objective_names},
    )
    _atomic_write_json(
        suggested_parameters_json_path,
        {name: [] for name in parameter_names},
    )


def _write_last_batch_messenger_files(
        state: ExperimentalState,
        target_point: int,
        n_evaluations_per_step: int,
        evaluated_objectives_json_path: str,
        suggested_parameters_json_path: str,
) -> None:
    last_batch_start = target_point - n_evaluations_per_step
    last_batch_indices = range(last_batch_start, target_point)

    first_point = state.points[last_batch_start]
    assert first_point.objective_values is not None, (
        f"Point {last_batch_start} has no objective values — "
        f"cannot write last batch to messenger file."
    )
    objective_names = list(first_point.objective_values.keys())
    parameter_names = list(first_point.parameters.keys())

    evaluated_objectives: dict[str, list[float]] = {name: [] for name in objective_names}
    suggested_parameters: dict[str, list[float]] = {name: [] for name in parameter_names}

    for point_no in last_batch_indices:
        point = state.points[point_no]
        assert point.objective_values is not None, (
            f"Point {point_no} has no objective values — cannot write last batch to messenger file."
        )
        for obj_name in objective_names:
            evaluated_objectives[obj_name].append(point.objective_values[obj_name])
        for param_name in parameter_names:
            suggested_parameters[param_name].append(point.parameters[param_name])

    _atomic_write_json(evaluated_objectives_json_path, evaluated_objectives)
    _atomic_write_json(suggested_parameters_json_path, suggested_parameters)


def _handle_simulation_directories(
        path_manager: PathManager,
        point_nos: list[int],
        handling: SimulationFolderHandling,
        experiment_version: Optional[str],
) -> None:

    if not point_nos:
        print("  ✓ No simulation directories to handle.")
        return

    dirs_to_handle = []
    for point_no in point_nos:
        simulation_id = PathManager.make_simulation_id(
            point_no=point_no,
            version=experiment_version,
        )
        dir_path = os.path.join(path_manager.results_directory, simulation_id)
        if os.path.isdir(dir_path):
            dirs_to_handle.append(dir_path)

    if not dirs_to_handle:
        print("  ✓ No simulation directories found on disk to handle.")
        return

    if handling == "delete":
        print(f"\n  The following {len(dirs_to_handle)} simulation directories will be deleted:")
        for dir_path in dirs_to_handle:
            print(f"    {dir_path}")
        answer = input("\n  Confirm deletion? [yes/no]: ").strip().lower()
        if answer != "yes":
            print("  Deletion cancelled. Directories left unchanged.")
            return
        _delete_directories(dirs_to_handle)

    elif handling == "force_delete":
        _delete_directories(dirs_to_handle)

    elif handling == "rename":
        for dir_path in dirs_to_handle:
            renamed_path = dir_path + "_rolled_back"
            os.rename(dir_path, renamed_path)
        print(f"  ✓ Renamed {len(dirs_to_handle)} simulation directories (suffix: _rolled_back).")


def _delete_directories(dirs: list[str]) -> None:
    for dir_path in dirs:
        shutil.rmtree(dir_path)
    print(f"  ✓ Deleted {len(dirs)} simulation directories.")


def _verify_rollback(
        state_json_path: str,
        optimiser_json_path: str,
        expected_state_points: int,
        expected_optimiser_points: int,
        allow_automatic_json_updates: Optional[bool],
) -> None:

    reloaded_state = ExperimentalState.load(state_json_path)
    assert reloaded_state.next_point == expected_state_points, (
        f"Rollback verification failed: experimental state next_point "
        f"is {reloaded_state.next_point}, expected {expected_state_points}."
    )
    assert len(reloaded_state.points) == expected_state_points, (
        f"Rollback verification failed: experimental state has {len(reloaded_state.points)} points, "
        f"expected {expected_state_points}."
    )

    reloaded_optimiser = load_optimiser_from_state(
        file_name=optimiser_json_path,
        allow_automatic_json_updates=allow_automatic_json_updates,
    )
    assert reloaded_optimiser.n_points_evaluated == expected_optimiser_points, (
        f"Rollback verification failed: optimiser has {reloaded_optimiser.n_points_evaluated} "
        f"evaluated points, expected {expected_optimiser_points}."
    )

    print(
        f"  ✓ Verification passed: state has {expected_state_points} points, "
        f"optimiser has {expected_optimiser_points} points."
    )
