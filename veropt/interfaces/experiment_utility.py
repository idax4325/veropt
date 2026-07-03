import json
from enum import StrEnum
from typing import Optional, Self, Union
import os

import torch

from veropt.interfaces.result_processing import ObjectivesDict
from veropt.interfaces.simulation import SimulationResult, SimulationResultsDict
from veropt.interfaces.utility import Config, create_directory

from pydantic import BaseModel

from veropt.optimiser.objective import InterfaceObjective

ParametersDict = dict[int, dict[str, float]]


class Point(BaseModel):
    parameters: dict[str, float]
    state: str
    job_id: Optional[int] = None
    result: Optional[SimulationResult] = None
    objective_values: Optional[dict[str, float]] = None


class ExperimentMode(StrEnum):
    local = "local"
    local_slurm = "local_slurm"
    remote_slurm = "remote_slurm"


class ExperimentalState(Config):
    experiment_name: str
    experiment_directory: str
    points: dict[int, Point]
    next_point: int
    just_rebuilt: bool = False

    def update(
            self,
            new_point: Point
    ) -> None:

        self.points[self.next_point] = new_point
        self.next_point += 1

    @classmethod
    def make_fresh_state(
            cls,
            experiment_name: str,
            experiment_directory: str
    ) -> Self:

        return cls(
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
            points={},
            next_point=0
        )

    def get_results(
            self,
            start_point: int,
            end_point: int
    ) -> SimulationResultsDict:

        points_batch = {point_no: self.points[point_no] for point_no in range(start_point, end_point + 1)}
        results_batch = {point_no: point.result for point_no, point in points_batch.items()}

        for point_no, result in results_batch.items():
            assert result is not None, f"No result found for point {point_no}"

        return results_batch  # type: ignore[return-value] #  Type asserted just above

    def get_objective_values(
            self,
            start_point: int,
            end_point: int
    ) -> Union[ObjectivesDict, dict[int, None]]:

        points_batch = {point_no: self.points[point_no] for point_no in range(start_point, end_point + 1)}
        objective_values_batch = {point_no: point.objective_values for point_no, point in points_batch.items()}

        # mypy seems to think this will be ObjectivesDict or None. I don't see why =/
        return objective_values_batch  # type: ignore [return-value]

    def get_parameters(
            self,
            start_point: int,
            end_point: int
    ) -> ParametersDict:

        results = self.get_results(
            start_point=start_point,
            end_point=end_point
        )

        return {point_no: result.parameters for point_no, result in results.items()}

    @property
    def n_points(self) -> int:
        return len(self.points)


class ExperimentConfig(Config):
    experiment_name: str
    version: Optional[str] = None
    parameter_names: list[str]
    parameter_bounds: dict[str, list[float]]
    path_to_experiment: str
    experiment_mode: ExperimentMode
    experiment_directory_name: Optional[str] = None
    run_script_filename: Optional[str] = None
    run_script_root_directory: Optional[str] = None
    output_filename: str
    noise_std: Optional[dict[str, float]] = None
    noise_std_min: Optional[dict[str, float]] = None
    noise_std_max: Optional[dict[str, float]] = None
    train_noise: bool = False


class PathManager:
    def __init__(
            self,
            experiment_config: ExperimentConfig
    ):

        self.experiment_config = experiment_config

        create_directory(self.experiment_directory)

        if self.run_script_root_directory is not None:
            assert os.path.isdir(self.run_script_root_directory), "Run script root directory not found."

        create_directory(self.results_directory)

    @property
    def experiment_directory(self) -> str:
        if self.experiment_config.experiment_directory_name is not None:
            path = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_directory_name
            )

        else:
            path = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_name
            )

        return path

    @property
    def run_script_root_directory(self) -> Optional[str]:

        if self.experiment_config.run_script_root_directory is not None:
            return self.experiment_config.run_script_root_directory

        default_path = os.path.join(
            self.experiment_directory,
            f"{self.experiment_config.experiment_name}_setup"
        )

        if os.path.isdir(default_path):
            return default_path

        return None

    @property
    def results_directory(self) -> str:
        return os.path.join(self.experiment_directory, "results")

    @property
    def experimental_state_json(self) -> str:

        return os.path.join(
            self.experiment_directory,
            f"{self.experiment_name_with_version}_experimental_state.json"
        )

    @property
    def suggested_parameters_json(self) -> str:
        return os.path.join(
            self.results_directory,
            f"{self.experiment_name_with_version}_suggested_parameters.json"
        )

    @property
    def evaluated_objectives_json(self) -> str:
        return os.path.join(
            self.results_directory,
            f"{self.experiment_name_with_version}_evaluated_objectives.json"
        )

    @property
    def optimiser_state_json(self) -> str:

        return os.path.join(
            self.experiment_directory,
            f"{self.experiment_name_with_version}_optimiser_state.json"
        )

    @property
    def version_string(self) -> str:
        if self.experiment_config.version is not None:
            version_string = f"_{self.experiment_config.version}"
        else:
            version_string = ""

        return version_string

    @property
    def experiment_name_with_version(self) -> str:
        return f"{self.experiment_config.experiment_name}{self.version_string}"

    @staticmethod
    def make_simulation_id(
            point_no: int,
            version: Optional[str] = None
    ) -> str:

        if version is not None:
            version_string = f"_{version}"
        else:
            version_string = ""

        return f"point_{point_no}{version_string}"


class ExperimentObjective(InterfaceObjective):

    name = "experiment_objective"

    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            variable_names: list[str],
            objective_names: list[str],
            suggested_parameters_json: str,
            evaluated_objectives_json: str,
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ):

        self.suggested_parameters_json = suggested_parameters_json
        self.evaluated_objectives_json = evaluated_objectives_json

        super().__init__(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=n_variables,
            n_objectives=n_objectives,
            variable_names=variable_names,
            objective_names=objective_names,
            noise_std=noise_std,
            noise_std_min=noise_std_min,
            noise_std_max=noise_std_max,
            train_noise=train_noise
        )

    def save_candidates(
            self,
            suggested_variables: dict[str, torch.Tensor],
    ) -> None:

        suggested_variables_np = {name: value.tolist() for name, value in suggested_variables.items()}

        with open(self.suggested_parameters_json, 'w') as f:
            json.dump(suggested_variables_np, f)

    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        with open(self.suggested_parameters_json, 'r') as f:
            suggested_variables_np = json.load(f)

        with open(self.evaluated_objectives_json, 'r') as f:
            evaluated_objectives_np = json.load(f)

        suggested_variables = {name: torch.tensor(value) for name, value in suggested_variables_np.items()}
        evaluated_objectives = {name: torch.tensor(value) for name, value in evaluated_objectives_np.items()}

        return suggested_variables, evaluated_objectives

    def gather_dicts_to_save(self) -> dict:
        saved_state = super().gather_dicts_to_save()
        saved_state['state']['suggested_parameters_json'] = self.suggested_parameters_json
        saved_state['state']['evaluated_objectives_json'] = self.evaluated_objectives_json
        return saved_state

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        bounds_lower = saved_state["bounds"][0]
        bounds_upper = saved_state["bounds"][1]

        return cls(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=saved_state["n_variables"],
            n_objectives=saved_state["n_objectives"],
            variable_names=saved_state["variable_names"],
            objective_names=saved_state["objective_names"],
            suggested_parameters_json=saved_state["suggested_parameters_json"],
            evaluated_objectives_json=saved_state["evaluated_objectives_json"],
            noise_std=saved_state.get("noise_std", None),
            noise_std_min=saved_state.get("noise_std_min", None),
            noise_std_max=saved_state.get("noise_std_max", None),
            train_noise=saved_state.get("train_noise", False),
        )
