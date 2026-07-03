import abc
from typing import Literal, Optional, Self

import botorch
import torch

from veropt.optimiser.objective import CallableObjective


class BotorchPracticeObjective(CallableObjective, metaclass=abc.ABCMeta):

    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            function: botorch.test_functions.base.BaseTestProblem,
            variable_names: Optional[list[str]] = None,
            objective_names: Optional[list[str]] = None,
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ):

        variable_names = variable_names or [f"var_{i}" for i in range(1, n_variables + 1)]
        objective_names = objective_names or [f"obj_{i}" for i in range(1, n_objectives + 1)]

        self.function = function

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

    def _run(self, parameter_values: torch.Tensor) -> torch.Tensor:

        return self.function(parameter_values)


class Hartmann(BotorchPracticeObjective):

    name = 'hartmann'

    def __init__(
            self,
            n_variables: Literal[3, 4, 6],
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ):

        assert n_variables in [3, 4, 6]

        n_objectives = 1

        function = botorch.test_functions.Hartmann(negate=True)

        super().__init__(
            bounds_lower=[0.0] * n_variables,
            bounds_upper=[1.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=['Hartmann'],
            noise_std=noise_std,
            noise_std_min=noise_std_min,
            noise_std_max=noise_std_max,
            train_noise=train_noise
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            n_variables=saved_state['n_variables'],
            noise_std=saved_state.get('noise_std', None),
            noise_std_min=saved_state.get('noise_std_min', None),
            noise_std_max=saved_state.get('noise_std_max', None),
            train_noise=saved_state.get('train_noise', False),
        )


class VehicleSafety(BotorchPracticeObjective):

    name = 'vehicle_safety'

    def __init__(
            self,
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ) -> None:
        n_variables = 5
        n_objectives = 3
        function = botorch.test_functions.VehicleSafety()
        objective_names = [f"VeSa {obj_no + 1}" for obj_no in range(n_objectives)]

        super().__init__(
            bounds_lower=[1.0] * n_variables,
            bounds_upper=[3.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=objective_names,
            noise_std=noise_std,
            noise_std_min=noise_std_min,
            noise_std_max=noise_std_max,
            train_noise=train_noise
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            noise_std=saved_state.get('noise_std', None),
            noise_std_min=saved_state.get('noise_std_min', None),
            noise_std_max=saved_state.get('noise_std_max', None),
            train_noise=saved_state.get('train_noise', False),
        )


class DTLZ1(BotorchPracticeObjective):

    name = 'dtlz_1'

    def __init__(
            self,
            n_variables: int = 10,
            n_objectives: int = 5,
            noise_std: Optional[dict[str, float]] = None,
            noise_std_min: Optional[dict[str, float]] = None,
            noise_std_max: Optional[dict[str, float]] = None,
            train_noise: bool = False
    ):

        function = botorch.test_functions.DTLZ1(
            dim=n_variables,
            num_objectives=n_objectives,
            negate=True
        )

        objective_names = [f"DTLZ1 {obj_no + 1}" for obj_no in range(n_objectives)]

        super().__init__(
            bounds_lower=[0.0] * n_variables,
            bounds_upper=[1.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=objective_names,
            noise_std=noise_std,
            noise_std_min=noise_std_min,
            noise_std_max=noise_std_max,
            train_noise=train_noise
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            n_variables=saved_state['n_variables'],
            n_objectives=saved_state['n_objectives'],
            noise_std=saved_state.get('noise_std', None),
            noise_std_min=saved_state.get('noise_std_min', None),
            noise_std_max=saved_state.get('noise_std_max', None),
            train_noise=saved_state.get('train_noise', False),
        )
