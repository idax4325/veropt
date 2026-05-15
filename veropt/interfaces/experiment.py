import os.path
from typing import Optional, Union, Self
import json

from veropt.interfaces.simulation import SimulationRunner
from veropt.interfaces.batch_manager import make_batch_manager, DirectBatchManager, \
    SubmitBatchManager
from veropt.interfaces.result_processing import ResultProcessor, ObjectivesDict
from veropt.interfaces.experiment_utility import (
    ExperimentConfig, ExperimentMode, ExperimentalState, PathManager, Point, ParametersDict, ExperimentObjective
)
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_saver_loader import (
    bayesian_optimiser, load_optimiser_from_settings, load_optimiser_from_state, save_to_json
)

import torch
import numpy as np

torch.set_default_dtype(torch.float64)


def _mask_nans(
        dict_of_objectives: ObjectivesDict,
        experimental_state: ExperimentalState
) -> ObjectivesDict:  # TODO: Remove when veropt core supports nan imputs

    current_minima: dict[str, float] = {}
    current_stds: dict[str, float] = {}
    first_new_point = next(iter(dict_of_objectives.values()))

    assert experimental_state.points, "To clear nans, there must be at least one point saved to state."

    for objective_name in first_new_point.keys():
        objective_values = []

        for i in range(experimental_state.next_point):
            if experimental_state.points[i].objective_values is not None:  # I check if it is None right here
                objective_values.append(experimental_state.points[i].objective_values[objective_name])  # type: ignore
            else:
                continue

        assert objective_values, f'No objective values found for objective "{objective_name}".'

        current_minima[objective_name] = np.nanmin(objective_values)  # type: ignore[arg-type]  # Type checked above
        current_stds[objective_name] = np.nanstd(objective_values).astype(float)  # type: ignore[arg-type]

        assert not np.isnan(current_minima[objective_name]), (
            f'All objective values are nans for objective "{objective_name}".'
        )

    for i, objectives in dict_of_objectives.items():
        dict_of_objectives[i] = {
            name: value if not np.isnan(value) else current_minima[name] - 2 * current_stds[name]
            for name, value in objectives.items()
        }

    return dict_of_objectives


class Experiment:
    def __init__(
            self,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: ExperimentConfig,
            optimiser: BayesianOptimiser,
            path_manager: PathManager,
            batch_manager: Union[DirectBatchManager, SubmitBatchManager],
            state: ExperimentalState
    ) -> None:

        self.experiment_config = experiment_config
        self.path_manager = path_manager

        self.simulation_runner = simulation_runner
        self.batch_manager = batch_manager
        self.result_processor = result_processor

        self.state = state
        self.optimiser = optimiser

        self.n_parameters = len(self.experiment_config.parameter_names)
        self.n_objectives = len(self.result_processor.objective_names)

    @classmethod
    def from_the_beginning(
            cls,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, dict],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)
        path_manager = PathManager(experiment_config)

        state = ExperimentalState.make_fresh_state(
            experiment_name=experiment_config.experiment_name,
            experiment_directory=path_manager.experiment_directory
        )

        state_json_path = path_manager.experimental_state_json

        if os.path.exists(state_json_path):
            raise RuntimeError(
                f"Experimental state exists at {state_json_path}. Please clear all files from previous run,"
                f"unless you want to continue that run. (In that case, use .continue_if_possible instead of"
                f".from_the_beginning.)"
            )

        n_parameters = len(experiment_config.parameter_names)
        n_objectives = len(result_processor.objective_names)

        optimiser = cls._make_fresh_optimiser(
            n_parameters=n_parameters,
            n_objectives=n_objectives,
            experiment_config=experiment_config,
            result_processor=result_processor,
            path_manager=path_manager,
            optimiser_config=optimiser_config,
        )

        batch_manager = cls._make_fresh_batch_manager(
            experiment_config=experiment_config,
            simulation_runner=simulation_runner,
            path_manager=path_manager,
            batch_manager_class=batch_manager_class
        )

        experiment = cls(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser=optimiser,
            path_manager=path_manager,
            batch_manager=batch_manager,
            state=state
        )

        experiment._initialise_objective_jsons()

        return experiment

    @classmethod
    def _continue_existing(
            cls,
            state_path: str,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None,
            allow_automatic_json_updates: Optional[bool] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)
        path_manager = PathManager(experiment_config)

        state = ExperimentalState.load(state_path)

        batch_manager = cls._make_fresh_batch_manager(
            experiment_config=experiment_config,
            simulation_runner=simulation_runner,
            path_manager=path_manager,
            batch_manager_class=batch_manager_class
        )

        optimiser = load_optimiser_from_state(
            file_name=path_manager.optimiser_state_json,
            allow_automatic_json_updates=allow_automatic_json_updates
        )

        return cls(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser=optimiser,
            path_manager=path_manager,
            batch_manager=batch_manager,
            state=state
        )

    @classmethod
    def continue_if_possible(
            cls,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, dict],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None,
            state_path: Optional[str] = None,
            allow_automatic_json_updates: Optional[bool] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)

        if state_path is None:
            path_manager = PathManager(experiment_config)
            state_path = path_manager.experimental_state_json

        if os.path.exists(state_path):
            return cls._continue_existing(
                state_path=state_path,
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=experiment_config,
                batch_manager_class=batch_manager_class,
                allow_automatic_json_updates=allow_automatic_json_updates
            )
        else:
            return cls.from_the_beginning(
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=experiment_config,
                optimiser_config=optimiser_config,
                batch_manager_class=batch_manager_class
            )

    @classmethod
    def continue_with_new_version(
            cls,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            old_experiment_config: Union[str, ExperimentConfig],
            new_experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, dict],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None,
            allow_automatic_json_updates: Optional[bool] = None
    ) -> Self:

        # TODO: Change name
        #   - But allow us to stay in same directoy
        #   - So maybe add an addendum to the name?
        #       - Maybe we do versions...?

        # TODO: Add option to change parameters as well
        #   - assuming a new parameter has just been the default value up until this point,
        #   can just add it with the same value at each point
        #   - so would need to check if parameters are already there or not
        #       * and be provided a default value

        new_experiment_config = ExperimentConfig.load(new_experiment_config)
        new_path_manager = PathManager(new_experiment_config)
        new_state_path = new_path_manager.experimental_state_json

        old_experiment_config = ExperimentConfig.load(old_experiment_config)
        old_path_manager = PathManager(old_experiment_config)
        old_state_path = old_path_manager.experimental_state_json

        assert new_experiment_config.experiment_name == old_experiment_config.experiment_name, (
            f"Attempted to make new version of experiment '{old_experiment_config.experiment_name}' but name doesn't "
            f"match with the one in the new configuration file ('{new_experiment_config.experiment_name}')."
        )

        old_state = ExperimentalState.load(old_state_path)
        old_optimiser = load_optimiser_from_state(
            file_name=old_path_manager.optimiser_state_json,
            allow_automatic_json_updates=allow_automatic_json_updates
        )

        if os.path.exists(new_state_path):

            assert os.path.exists(new_path_manager.optimiser_state_json), (
                "Found existing experimental state but no existing optimiser. Please clean up your files "
                "and try again."
            )

            new_optimiser = load_optimiser_from_state(
                file_name=new_path_manager.optimiser_state_json,
                allow_automatic_json_updates=allow_automatic_json_updates
            )

            assert new_optimiser.n_points_evaluated < old_optimiser.n_points_evaluated, (
                f"Attempted to create new version of experiment {old_experiment_config.experiment_name}, but "
                f"it seems like that version has already loaded all points from the old version. "
                f"Either 1) change to a new version name, 2) erase the existing files or "
                f"3) continue running the existing version."
            )

            experiment = cls._continue_existing(
                state_path=new_state_path,
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=new_experiment_config,
                batch_manager_class=batch_manager_class
            )

            n_steps_evaluated = old_optimiser.n_points_evaluated // experiment.n_evaluations_per_step
            n_steps_loaded = new_optimiser.n_points_evaluated // experiment.n_evaluations_per_step
            n_steps_to_reevaluate = n_steps_evaluated - n_steps_loaded

        else:

            batch_manager = cls._make_fresh_batch_manager(
                experiment_config=new_experiment_config,
                simulation_runner=simulation_runner,
                path_manager=new_path_manager,
                batch_manager_class=batch_manager_class
            )

            experiment = cls(
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=new_experiment_config,
                optimiser=old_optimiser,
                path_manager=new_path_manager,
                batch_manager=batch_manager,
                state=old_state
            )

            n_steps_to_reevaluate = experiment.n_points_evaluated // experiment.n_evaluations_per_step

            experiment._reset_objective_values()

            experiment.optimiser = experiment._make_fresh_optimiser(
                n_parameters=experiment.n_parameters,
                n_objectives=experiment.n_objectives,
                experiment_config=experiment.experiment_config,
                result_processor=experiment.result_processor,
                path_manager=experiment.path_manager,
                optimiser_config=optimiser_config,
            )

        for step_no in range(n_steps_to_reevaluate):
            experiment.re_run_experiment_step_from_existing_data()

        experiment.optimiser.train_model()
        experiment._save_optimiser()

        experiment.state.just_rebuilt = True

        return experiment

    @staticmethod
    def _make_fresh_optimiser(
            n_parameters: int,
            n_objectives: int,
            experiment_config: ExperimentConfig,
            result_processor: ResultProcessor,
            path_manager: PathManager,
            optimiser_config: Union[str, dict]
    ) -> BayesianOptimiser:

        bounds_lower = [experiment_config.parameter_bounds[name][0]
                        for name in experiment_config.parameter_names]
        bounds_upper = [experiment_config.parameter_bounds[name][1]
                        for name in experiment_config.parameter_names]

        objective = ExperimentObjective(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=n_parameters,
            n_objectives=n_objectives,
            variable_names=experiment_config.parameter_names,
            objective_names=result_processor.objective_names,
            suggested_parameters_json=path_manager.suggested_parameters_json,
            evaluated_objectives_json=path_manager.evaluated_objectives_json,
            noise_std=experiment_config.noise_std,
            noise_std_min=experiment_config.noise_std_min,
            noise_std_max=experiment_config.noise_std_max,
            train_noise=experiment_config.train_noise,
        )

        if isinstance(optimiser_config, str):
            return load_optimiser_from_settings(
                file_name=optimiser_config,
                objective=objective
            )

        elif isinstance(optimiser_config, dict):
            return bayesian_optimiser(
                objective=objective,
                **optimiser_config
            )

        else:
            raise ValueError("optimiser_config must be a valid dictionary or a path to a valid configuration file")

    @staticmethod
    def _make_fresh_batch_manager(
            experiment_config: ExperimentConfig,
            simulation_runner: SimulationRunner,
            path_manager: PathManager,
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]]
    ) -> Union[DirectBatchManager, SubmitBatchManager]:

        # TODO: Refactor this (consider how to set up this little guy)
        #   - Fixed this so it works but isn't general
        #   - Probably need a constructor that makes the experiment itself...?
        #   - Or actually, might mostly need to make an interface to the batch_manager where it can take
        #     some of the things it is being bound to here?
        #       - Then it can be initialised on its own but can be linked to internal things here or something

        return make_batch_manager(
            experiment_mode=experiment_config.experiment_mode.name,  # type: ignore[arg-type]  # Silly mypy
            simulation_runner=simulation_runner,
            run_script_filename=experiment_config.run_script_filename,
            run_script_root_directory=path_manager.run_script_root_directory,
            results_directory=path_manager.results_directory,
            experimental_state_json=path_manager.experimental_state_json,
            output_filename=experiment_config.output_filename,
            check_job_status_frequency=60,  # TODO: put in server config,
            batch_manager_class=batch_manager_class,
            experiment_version=experiment_config.version
        )

    def _initialise_objective_jsons(self) -> None:

        initial_parameter_dict: dict[str, list] = {name: [] for name in self.experiment_config.parameter_names}
        initial_objectives_dict: dict[str, list] = {name: [] for name in self.result_processor.objective_names}

        with open(self.path_manager.suggested_parameters_json, "w") as f:
            json.dump(initial_parameter_dict, f)

        with open(self.path_manager.evaluated_objectives_json, "w") as f:
            json.dump(initial_objectives_dict, f)

    def _reset_objective_values(
            self
    ) -> None:

        for point_no, point in self.state.points.items():
            point.objective_values = None

    def get_parameters_from_optimiser(self) -> dict[int, dict]:

        with open(self.path_manager.suggested_parameters_json, 'r') as f:
            suggested_parameters = json.load(f)

        dict_of_parameters = {}

        for i in range(self.optimiser.n_evaluations_per_step):
            parameters = {name: value[i] for name, value in suggested_parameters.items()}
            dict_of_parameters[self.state.next_point] = parameters
            new_point = Point(
                parameters=parameters,
                state="Received parameters from core"
            )

            self.state.update(new_point)

        self.state.save_to_json(self.path_manager.experimental_state_json)

        return dict_of_parameters

    def save_objectives_to_state(
            self,
            dict_of_objectives: ObjectivesDict
    ) -> None:

        for point_no, objective_values in dict_of_objectives.items():
            self.state.points[point_no].objective_values = objective_values  # type: ignore[assignment]  # silly mypy

        self.state.save_to_json(self.path_manager.experimental_state_json)

    def send_objectives_to_optimiser(
            self,
            dict_of_objectives: ObjectivesDict
    ) -> None:

        # TODO: Remove when veropt core supports nan imputs
        dict_of_objectives = _mask_nans(
            dict_of_objectives=dict_of_objectives,
            experimental_state=self.state
        )

        evaluated_objectives = {
            name: [dict_of_objectives[point_no][name] for point_no in dict_of_objectives.keys()]
            for name in self.result_processor.objective_names
        }

        with open(self.path_manager.evaluated_objectives_json, "w") as f:
            json.dump(evaluated_objectives, f)

    def send_parameters_to_optimiser(
            self,
            dict_of_parameters: ParametersDict
    ) -> None:

        # TODO: Either change optimiser interface in general or make sure this saves the same as the ExperimentObj
        #   - Is saving/loading to/from the optimiser necessary?
        #       - Maybe refactor InterfaceObjective to allow passing directly?
        #   - Otherwise, make some shared function to save with
        #       - Currently just kind of assuming it'll be the same, oops

        evaluated_parameters = {
            name: [dict_of_parameters[point_no][name] for point_no in dict_of_parameters.keys()]
            for name in self.experiment_config.parameter_names
        }

        with open(self.path_manager.suggested_parameters_json, "w") as f:
            json.dump(evaluated_parameters, f)

    def _save_optimiser(self) -> None:

        save_to_json(
            object_to_save=self.optimiser,
            file_path=self.path_manager.optimiser_state_json
        )

    def run_experiment_step_direct(self) -> None:

        assert issubclass(type(self.batch_manager), DirectBatchManager), (
            "Batch manager must be subclassing DirectBatchManager to call this method."
        )
        self.optimiser.run_optimisation_step()

        dict_of_parameters = self.get_parameters_from_optimiser()

        results = self.batch_manager.run_batch(  # type: ignore[union-attr]  # Checked above
            dict_of_parameters=dict_of_parameters,
            experimental_state=self.state
        )

        objective_values = self.state.get_objective_values(
            start_point=self.current_batch_indices['start'],
            end_point=self.current_batch_indices['end']
        )

        dict_of_objectives = self.result_processor.process(
            results=results,
            existing_objective_values=objective_values
        )

        self.save_objectives_to_state(
            dict_of_objectives=dict_of_objectives
        )
        self.send_objectives_to_optimiser(
            dict_of_objectives=dict_of_objectives
        )

    def run_experiment_step_submitted(self) -> None:

        # Note for the future: Could consider doing two Optimiser and two Experiment classes instead of these checks
        assert issubclass(type(self.batch_manager), SubmitBatchManager), (
            "Batch manager must be subclassing SubmitBatchManager to call this method"
        )

        if not self.current_step == 0 and not self.state.just_rebuilt:

            self.batch_manager.wait_for_jobs(  # type: ignore[union-attr]  # Checked above
                experimental_state=self.state
            )

            results = self.state.get_results(
                start_point=self.current_batch_indices['start'],
                end_point=self.current_batch_indices['end']
            )

            objective_values = self.state.get_objective_values(
                start_point=self.current_batch_indices['start'],
                end_point=self.current_batch_indices['end']
            )

            dict_of_objectives = self.result_processor.process(
                results=results,
                existing_objective_values=objective_values
            )

            self.save_objectives_to_state(
                dict_of_objectives=dict_of_objectives
            )
            self.send_objectives_to_optimiser(
                dict_of_objectives=dict_of_objectives
            )

        self.optimiser.run_optimisation_step()

        dict_of_parameters = self.get_parameters_from_optimiser()

        self._save_optimiser()

        if not self.current_step == self.n_total_steps:
            self.batch_manager.submit_batch(  # type: ignore[union-attr]  # Checked above
                dict_of_parameters=dict_of_parameters,
                experimental_state=self.state
            )

        if self.state.just_rebuilt:
            self.state.just_rebuilt = False

    def re_run_experiment_step_from_existing_data(self) -> None:

        results = self.state.get_results(
            start_point=self.current_batch_indices['start'],
            end_point=self.current_batch_indices['end']
        )

        dict_of_objectives = self.result_processor.process(
            results=results,
            existing_objective_values=None
        )

        dict_of_parameters = self.state.get_parameters(
            start_point=self.current_batch_indices['start'],
            end_point=self.current_batch_indices['end']
        )

        self.save_objectives_to_state(
            dict_of_objectives=dict_of_objectives
        )

        self.send_objectives_to_optimiser(
            dict_of_objectives=dict_of_objectives
        )
        self.send_parameters_to_optimiser(
            dict_of_parameters=dict_of_parameters
        )

        self.optimiser.load_optimisation_step()

        # Note: Could wait with this until we get all data
        self._save_optimiser()

    def run_experiment_step(self) -> None:
        if self.experiment_config.experiment_mode == ExperimentMode.local:
            self.run_experiment_step_direct()

        elif self.experiment_config.experiment_mode == ExperimentMode.local_slurm:
            self.run_experiment_step_submitted()

        elif self.experiment_config.experiment_mode == ExperimentMode.remote_slurm:
            self.run_experiment_step_submitted()

        else:
            raise RuntimeError(f"Unknown experiment mode: '{self.experiment_config.experiment_mode}'")

    def run_experiment(self) -> None:

        n_remaining_steps = self.n_total_steps - self.current_step

        for i in range(n_remaining_steps):
            self.run_experiment_step()

    @property
    def current_step(self) -> int:
        return self.n_points_submitted // self.n_evaluations_per_step

    @property
    def n_total_steps(self) -> int:
        total_full_steps = (self.n_initial_points + self.n_bayesian_points) // self.n_evaluations_per_step
        return total_full_steps + 1

    @property
    def n_evaluations_per_step(self) -> int:
        return self.optimiser.n_evaluations_per_step

    @property
    def n_initial_points(self) -> int:
        return self.optimiser.n_initial_points

    @property
    def n_bayesian_points(self) -> int:
        return self.optimiser.n_bayesian_points

    @property
    def n_points_submitted(self) -> int:
        return self.state.n_points

    @property
    def n_points_evaluated(self) -> int:
        return self.optimiser.n_points_evaluated

    @property
    def current_batch_indices(self) -> dict[str, int]:
        return {
            'start': self.n_points_evaluated,
            'end': self.n_points_evaluated + self.n_evaluations_per_step - 1
        }

    def restart_experiment(self) -> None:
        raise NotImplementedError("Restarting an experiment is not implemented yet.")
