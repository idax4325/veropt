import abc
import functools
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Self, TypedDict, Unpack

import botorch
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from veropt.optimiser.optimiser_utility import get_nadir_point
from veropt.optimiser.utility import (
    DataShape, _validate_typed_dict, check_variable_and_objective_shapes, check_variable_objective_values_matching,
    enforce_amount_of_positional_arguments, unpack_variables_objectives_from_kwargs
)
from veropt.optimiser.saver_loader_utility import EmptyDataClass, SavableClass, SavableDataClass


def _check_input_dimensions[T, **P](
        function: Callable[P, T]
) -> Callable[P, T]:

    @functools.wraps(function)
    def check_dimensions(
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> T:

        enforce_amount_of_positional_arguments(
            function=function,
            received_args=args
        )

        assert isinstance(args[0], AcquisitionFunction)
        self: AcquisitionFunction = args[0]

        variable_values, objective_values = unpack_variables_objectives_from_kwargs(kwargs)

        if variable_values is None and objective_values is None:
            raise RuntimeError("This decorator was called to check input shapes but found no valid inputs.")

        check_variable_and_objective_shapes(
            n_variables=self.n_variables,
            n_objectives=self.n_objectives,
            function_name=function.__name__,
            class_name=self.__class__.__name__,
            variable_values=variable_values,
            objective_values=objective_values,
        )

        return function(
            *args,
            **kwargs
        )

    return check_dimensions


class AcquisitionFunction(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'
    multi_objective: bool

    def __init__(
            self,
            n_variables: int,
            n_objectives: int
    ) -> None:

        self.n_variables = n_variables
        self.n_objectives = n_objectives

        self.function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

        if self.multi_objective:
            assert self.n_objectives > 1, (
                f"This acquisition function ({self.__class__.__name__}) is meant for multi-objective problems but "
                f"received only {self.n_objectives} objective."
            )
        else:
            assert self.n_objectives == 1, (
                f"This acquisition function ({self.__class__.__name__}) is meant for single-objective problems but "
                f"received {self.n_objectives} objectives."
            )

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

    @abc.abstractmethod
    def get_settings(self) -> SavableDataClass:
        pass

    @classmethod
    @abc.abstractmethod
    def from_n_variables_n_objectives_and_settings(
            cls,
            n_variables: int,
            n_objectives: int,
            settings: Mapping[str, Any]
    ) -> Self:
        pass

    @_check_input_dimensions
    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        assert self.function is not None, "The acquisition function must receive a model before being called."

        return self.function(variable_values)

    def gather_dicts_to_save(self) -> dict:

        if self.get_settings() is not None:
            settings_dict = self.get_settings().gather_dicts_to_save()
        else:
            settings_dict = {}

        return {
            'name': self.name,
            'state': {
                'n_variables': self.n_variables,
                'n_objectives': self.n_objectives,
                'settings': settings_dict
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        return cls.from_n_variables_n_objectives_and_settings(
            n_variables=saved_state['n_variables'],
            n_objectives=saved_state['n_objectives'],
            settings=saved_state['settings']
        )


class BotorchAcquisitionFunction(AcquisitionFunction, metaclass=abc.ABCMeta):

    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        n_points_in_call = variable_values.shape[DataShape.index_points]

        # TODO: See if we can make this run without a for loop (botorch complains atm)
        if n_points_in_call > 1:

            acquisition_values = torch.zeros(n_points_in_call)

            for point_no in range(n_points_in_call):
                acquisition_values[point_no] = super().__call__(
                    variable_values=variable_values[point_no:point_no + 1, :]
                )

            return acquisition_values

        else:
            return super().__call__(
                variable_values=variable_values
            )

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def refresh(
            self,
            *,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        # This structure is to automatically have the decorator on all implementations of refresh
        self._refresh(
            model=model,
            variable_values=variable_values,
            objective_values=objective_values,
        )

    @abc.abstractmethod
    def _refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:
        pass


class QLogExpectedHyperVolumeImprovement(BotorchAcquisitionFunction):

    name = 'qlogehvi'
    multi_objective = True

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
    ) -> None:

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    @classmethod
    def from_n_variables_n_objectives_and_settings(
            cls,
            n_variables: int,
            n_objectives: int,
            settings: Mapping[str, Any]
    ) -> Self:
        assert len(settings) == 0, f"{cls.name} doesn't use any settings but received {settings}"

        return cls(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    def get_settings(self) -> SavableDataClass:
        return EmptyDataClass()

    def _refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        nadir_point = get_nadir_point(
            variable_values=variable_values,
            objective_values=objective_values
        )

        partitioning = FastNondominatedPartitioning(
            ref_point=nadir_point,
            Y=objective_values
        )

        self.function = botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=nadir_point,
            partitioning=partitioning  # type: ignore[arg-type]  # Seems like botorch needs to update their types?
        )


class QLogNoisyExpectedHypervolumeImprovement(BotorchAcquisitionFunction):

    name = 'qlogneHVI'
    multi_objective = True

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
    ) -> None:

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    @classmethod
    def from_n_variables_n_objectives_and_settings(
            cls,
            n_variables: int,
            n_objectives: int,
            settings: Mapping[str, Any]
    ) -> Self:
        assert len(settings) == 0, f"{cls.name} doesn't use any settings but received {settings}"

        return cls(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    def get_settings(self) -> SavableDataClass:
        return EmptyDataClass()

    def _refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        nadir_point = get_nadir_point(
            variable_values=variable_values,
            objective_values=objective_values
        )

        self.function = botorch.acquisition.multi_objective.logei.qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=nadir_point,
            X_baseline=variable_values,
            prune_baseline=True
        )


# For typing in the constructors
class UpperConfidenceBoundOptionsInputDict(TypedDict, total=False):
    beta: float


@dataclass
class UpperConfidenceBoundOptions(SavableDataClass):
    beta: float = 3.0


class UpperConfidenceBound(BotorchAcquisitionFunction):

    name = 'ucb'
    multi_objective = False

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
            **settings: Unpack[UpperConfidenceBoundOptionsInputDict]
    ):

        self.settings = UpperConfidenceBoundOptions(
            **settings
        )

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    def _refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        self.function = botorch.acquisition.analytic.UpperConfidenceBound(
            model=model,
            beta=self.settings.beta
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings

    @classmethod
    def from_n_variables_n_objectives_and_settings(
            cls,
            n_variables: int,
            n_objectives: int,
            settings: Mapping[str, Any]
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=UpperConfidenceBoundOptionsInputDict,
            object_name=cls.name
        )

        return cls(
            n_variables=n_variables,
            n_objectives=n_objectives,
            **settings
        )
