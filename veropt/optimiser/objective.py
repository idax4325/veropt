import abc
from enum import Enum
from typing import Optional, Union

import torch

from veropt.optimiser.saver_loader_utility import SavableClass
from veropt.optimiser.utility import check_incoming_objective_dimensions_fix_1d


class Objective(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            variable_names: list[str],
            objective_names: list[str],
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ):
        assert len(bounds_lower) == n_variables
        assert len(bounds_upper) == n_variables

        self.bounds = torch.tensor([bounds_lower, bounds_upper])
        self.n_variables = n_variables
        self.n_objectives = n_objectives

        self.variable_names = variable_names
        self.objective_names = objective_names

        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.train_noise = train_noise

        self._validate_noise_configuration()

    def _validate_noise_configuration(self) -> None:
        if self.noise_std is not None:
            assert set(self.noise_std.keys()) == set(self.objective_names), (
                f"noise_std keys {set(self.noise_std.keys())} must match objective_names {set(self.objective_names)}."
            )

        if self.noise_std is not None and self.train_noise:
            raise ValueError(
                "noise_std pins the noise to a fixed physical value. "
                "train_noise=True would attempt to learn it simultaneously — these cannot be combined.\n"
                "  • Use noise_std alone if you know your measurement noise (e.g. noise_std={'obj': 0.05}).\n"
                "  • Use train_noise=True alone (optionally with noise_std_min/noise_std_max) if you want "
                "the model to learn the noise level from the data."
            )

        if (self.noise_std_min is not None or self.noise_std_max is not None) and not self.train_noise:
            raise ValueError(
                "noise_std_min and noise_std_max bound the noise level during learning, so they require "
                "train_noise=True (the GP optimizer learns what noise is consistent with the data). "
                "With train_noise=False (default), the noise is fixed and there is nothing to bound.\n"
                "  • Add train_noise=True to use bounds, or\n"
                "  • Remove noise_std_min/noise_std_max and use noise_std to fix the noise to an exact value."
            )

        if self.noise_std_min is not None:
            assert set(self.noise_std_min.keys()) == set(self.objective_names), (
                f"noise_std_min keys {set(self.noise_std_min.keys())} must match "
                f"objective_names {set(self.objective_names)}."
            )

        if self.noise_std_max is not None:
            assert set(self.noise_std_max.keys()) == set(self.objective_names), (
                f"noise_std_max keys {set(self.noise_std_max.keys())} must match "
                f"objective_names {set(self.objective_names)}."
            )

    def get_bounds(
            self,
            variable: Union[int, str]
    ) -> list[float]:

        if isinstance(variable, str):
            variable_index = self.variable_names.index(variable)
        else:
            variable_index = variable

        bounds = self.bounds[:, variable_index]

        return bounds.tolist()

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'bounds': self.bounds,
                'n_variables': self.n_variables,
                'n_objectives': self.n_objectives,
                'variable_names': self.variable_names,
                'objective_names': self.objective_names,
                'noise_std': self.noise_std,
                'noise_std_min': self.noise_std_min,
                'noise_std_max': self.noise_std_max,
                'train_noise': self.train_noise,
            }
        }


class CallableObjective(Objective, metaclass=abc.ABCMeta):

    def __call__(self, parameter_values: torch.Tensor) -> torch.Tensor:

        objective_values = self._run(
            parameter_values=parameter_values
        )

        objective_values = check_incoming_objective_dimensions_fix_1d(
            objective_values=objective_values,
            n_objectives=self.n_objectives,
            function_name='__call__',
            class_name=self.__class__.__name__
        )

        return objective_values

    @abc.abstractmethod
    def _run(self, parameter_values: torch.Tensor) -> torch.Tensor:
        pass


# TODO: Consider if we want to check that var and obj names match at this level
class InterfaceObjective(Objective, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save_candidates(
            self,
            suggested_variables: dict[str, torch.Tensor]
    ) -> None:
        pass

    @abc.abstractmethod
    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pass


class ObjectiveKind(Enum):
    callable = 1
    interface = 2


def determine_objective_type(
        objective: Union[CallableObjective, InterfaceObjective]
) -> ObjectiveKind:

    if isinstance(objective, CallableObjective):
        return ObjectiveKind.callable
    elif isinstance(objective, InterfaceObjective):
        return ObjectiveKind.interface
    else:
        raise ValueError(
            f"The objective must be a subclass of either {CallableObjective.__name__} or {InterfaceObjective.__name__}."
        )
