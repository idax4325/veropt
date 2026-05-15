from copy import deepcopy
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from typing import Iterator, Mapping, Optional, Self, TypedDict, Union

import torch

from veropt.optimiser.initial_points import InitialPointsChoice
from veropt.optimiser.initial_points import InitialPointsGenerationMode
from veropt.optimiser.normalisation import Normaliser
from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass
from veropt.optimiser.utility import DataShape, PredictionDict, TensorWithNormalisationFlag


class OptimisationMode(StrEnum):
    initial = auto()
    bayesian = auto()


# TODO: Write a test to make sure the arguments of this and the dict are the same? (except n_init, n_bayes, n_objs)
class OptimiserSettings(SavableClass):

    def __init__(
            self,
            n_initial_points: int,
            n_bayesian_points: int,
            n_objectives: int,
            n_evaluations_per_step: int,
            initial_points_generator: InitialPointsChoice = 'random',
            normalise: bool = True,
            verbose: bool = True,
            renormalise_each_step: Optional[bool] = None,
            n_points_before_fitting: Optional[int] = None,
            objective_weights: Optional[list[float]] = None,
            allow_automatic_json_updates: bool = False
    ):
        self.n_initial_points = n_initial_points
        self.n_bayesian_points = n_bayesian_points
        self.n_objectives = n_objectives
        self.n_evaluations_per_step = n_evaluations_per_step

        self.initial_points_generator = InitialPointsGenerationMode(initial_points_generator)

        self.normalise = normalise
        self.verbose = verbose

        if renormalise_each_step is None:

            if n_objectives > 1:
                self.renormalise_each_step = True
            else:
                self.renormalise_each_step = False

        else:
            self.renormalise_each_step = renormalise_each_step

        if n_points_before_fitting is None:

            n_points_before_fitting = self.n_initial_points - self.n_evaluations_per_step

            if n_points_before_fitting < 1:
                n_points_before_fitting = self.n_initial_points

            self.n_points_before_fitting = n_points_before_fitting
        else:
            self.n_points_before_fitting = n_points_before_fitting

        if objective_weights is None:
            self.objective_weights = torch.ones(self.n_objectives) / self.n_objectives
        else:
            self.objective_weights = torch.tensor(objective_weights)

        self.allow_automatic_json_updates = allow_automatic_json_updates

    def gather_dicts_to_save(self) -> dict:

        return self.__dict__

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        return cls(
            **saved_state
        )


class OptimiserSettingsInputDict(TypedDict, total=False):
    objective_weights: list[float]
    normalise: bool
    n_points_before_fitting: int
    verbose: bool
    renormalise_each_step: bool
    initial_points_generator: InitialPointsChoice
    allow_automatic_json_updates: bool


@dataclass
class SuggestedPoints(SavableDataClass):
    variable_values: torch.Tensor
    predicted_objective_values: Optional[PredictionDict]
    generated_at_step: int
    generated_with_mode: str
    normalised: bool

    def __getitem__(
            self,
            point_no: int
    ) -> 'SuggestedPoints':

        if self.predicted_objective_values is not None:

            predicted_objective_values = {
                prediction_type: tensor[point_no, :]  # type: ignore[index]  # should be well-defined
                for (prediction_type, tensor) in self.predicted_objective_values.items()
            }

        else:
            predicted_objective_values = None

        return SuggestedPoints(
            variable_values=self.variable_values[point_no, :],
            predicted_objective_values=predicted_objective_values,  # type: ignore[arg-type]  # Should be safe
            generated_at_step=self.generated_at_step,
            generated_with_mode=self.generated_with_mode,
            normalised=self.normalised,
        )

    def __len__(self) -> int:
        return self.variable_values.shape[DataShape.index_points]

    def __iter__(self) -> Iterator['SuggestedPoints']:
        for suggested_point_no in range(len(self)):
            yield self[suggested_point_no]

    @property
    def variable_values_flagged(self) -> TensorWithNormalisationFlag:
        return TensorWithNormalisationFlag(
            tensor=self.variable_values,
            normalised=self.normalised
        )

    def copy(self) -> 'SuggestedPoints':

        if self.predicted_objective_values is None:

            predicted_values_cloned: Optional[PredictionDict] = None

        else:
            predicted_values_cloned = {
                'mean': self.predicted_objective_values['mean'].detach().clone(),
                'lower': self.predicted_objective_values['lower'].detach().clone(),
                'upper': self.predicted_objective_values['upper'].detach().clone(),
            }

        return SuggestedPoints(
            variable_values=self.variable_values.detach().clone(),
            predicted_objective_values=predicted_values_cloned,
            generated_at_step=deepcopy(self.generated_at_step),
            generated_with_mode=deepcopy(self.generated_with_mode),
            normalised=deepcopy(self.normalised),
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        if saved_state["predicted_objective_values"] is None:
            prediction: Optional[PredictionDict] = None

        else:

            prediction = {  # type: ignore[assignment]  # mypy is silly
                prediction_type: torch.tensor(saved_state['predicted_objective_values'][prediction_type])
                for prediction_type in ('mean', 'lower', 'upper')
            }

        return cls(
            variable_values=torch.tensor(saved_state['variable_values']),
            predicted_objective_values=prediction,
            generated_at_step=saved_state['generated_at_step'],
            generated_with_mode=saved_state['generated_with_mode'],
            normalised=saved_state['normalised'],
        )

    def gather_dicts_to_save(self) -> dict:
        return asdict(self.copy())


def normalise_suggested_points(
        suggested_points: SuggestedPoints,
        normaliser_variables: Normaliser,
        normaliser_objectives: Normaliser
) -> SuggestedPoints:

    assert suggested_points.normalised is False

    normalised_variable_values = normaliser_variables.transform(
        suggested_points.variable_values
    )

    if suggested_points.predicted_objective_values is not None:

        normalised_prediction: Optional[PredictionDict] = {  # type: ignore[assignment]  # mypy silliness
            prediction_type: normaliser_objectives.transform(
                suggested_points.predicted_objective_values[prediction_type]  # type: ignore[literal-required]
            )
            for prediction_type in ('mean', 'lower', 'upper')
        }

    else:
        normalised_prediction = None

    return SuggestedPoints(
        variable_values=normalised_variable_values,
        predicted_objective_values=normalised_prediction,
        generated_at_step=suggested_points.generated_at_step,
        generated_with_mode=suggested_points.generated_with_mode,
        normalised=True
    )


def unnormalise_suggested_points(
        suggested_points: SuggestedPoints,
        normaliser_variables: Normaliser,
        normaliser_objectives: Normaliser
) -> SuggestedPoints:

    assert suggested_points.normalised

    normalised_variable_values = normaliser_variables.inverse_transform(
        suggested_points.variable_values
    )

    if suggested_points.predicted_objective_values is not None:

        normalised_prediction: Optional[PredictionDict] = {  # type: ignore[assignment]  # mypy silliness
            prediction_type: normaliser_objectives.inverse_transform(
                suggested_points.predicted_objective_values[prediction_type]  # type: ignore[literal-required]
            )
            for prediction_type in ('mean', 'lower', 'upper')
        }

    else:
        normalised_prediction = None

    return SuggestedPoints(
        variable_values=normalised_variable_values,
        predicted_objective_values=normalised_prediction,
        generated_at_step=suggested_points.generated_at_step,
        generated_with_mode=suggested_points.generated_with_mode,
        normalised=False
    )


class ReferencePointInputDict(TypedDict, total=False):
    variable_values: dict[str, float]
    objective_values: dict[str, float]


@dataclass
class ReferencePoint(SavableDataClass):
    variable_values: torch.Tensor
    objective_values: Optional[torch.Tensor]
    normalised: bool

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        raw_objective_values = saved_state.get('objective_values')

        return cls(
            variable_values=torch.tensor(saved_state['variable_values']),
            objective_values=torch.tensor(raw_objective_values) if raw_objective_values is not None else None,
            normalised=saved_state['normalised']
        )


def _format_number(
        number: float
) -> str:
    if abs(number) < 0.0001:
        return f"{number:.2e}"
    elif abs(number) < 0.01:
        return f"{number:.4f}"
    elif abs(number) < 1:
        return f"{number:.3f}"
    elif abs(number) > 100:
        return f"{number:.1f}"
    elif abs(number) >= 10_000:
        return f"{number:.3e}"
    else:
        return f"{number:.2f}"


def _single_list_with_floats_to_string(
        unformatted_list: list[float]
) -> str:

    formatted_list = "["

    for iteration, list_item in enumerate(unformatted_list):
        if iteration < len(unformatted_list) - 1:
            formatted_list += f"{_format_number(list_item)}, "
        else:
            formatted_list += f"{_format_number(list_item)}]"

    return formatted_list


def list_with_floats_to_string(
        unformatted_list: Union[list[float], list[list[float]]]
) -> str:

    if len(unformatted_list) == 0:
        return "[]"

    if type(unformatted_list[0]) is list:

        formatted_list = _nested_list_of_floats_to_string(unformatted_list)  # type: ignore[arg-type]

    else:

        formatted_list = _single_list_with_floats_to_string(unformatted_list)  # type: ignore[arg-type]

    return formatted_list


def _nested_list_of_floats_to_string(
        unformatted_list: list[list[float]]
) -> str:
    formatted_list = "["

    for list_ind, list_of_floats in enumerate(unformatted_list):
        for number_ind, number in enumerate(list_of_floats):
            if number_ind < len(list_of_floats) - 1:
                formatted_list += f"{_format_number(number)}, "
            elif list_ind < len(unformatted_list) - 1:
                formatted_list += f"{_format_number(number)}], ["
            else:
                formatted_list += f"{_format_number(number)}]"

    return formatted_list


class BestPoints(TypedDict):
    variables: torch.Tensor
    objectives: torch.Tensor
    index: int


def get_best_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: torch.Tensor,
        objectives_greater_than: Optional[float | list[float]] = None,
        best_for_objecive_index: Optional[int] = None
) -> BestPoints | None:

    if len(variable_values) == 0:
        return {
            'variables': torch.tensor([]),
            'objectives': torch.tensor([]),
            'index': -1,
        }

    assert objectives_greater_than is None or best_for_objecive_index is None, "Specifying both options not supported"

    if objectives_greater_than is None and best_for_objecive_index is None:

        max_index_tensor = (objective_values * weights).sum(dim=DataShape.index_dimensions).argmax()
        max_index = int(max_index_tensor)

    elif objectives_greater_than is not None:

        max_index_or_none = _get_points_greater_than(
            objective_values=objective_values,
            weights=weights,
            objectives_greater_than=objectives_greater_than
        )

        if max_index_or_none is None:
            return None
        else:
            max_index = max_index_or_none

    elif best_for_objecive_index is not None:
        max_index_tensor = objective_values[:, best_for_objecive_index].argmax()
        max_index = int(max_index_tensor)

    else:
        raise ValueError

    best_variables = variable_values[max_index]
    best_values = objective_values[max_index]

    return {
        'variables': best_variables,
        'objectives': best_values,
        'index': max_index
    }


def _get_points_greater_than(
        objective_values: torch.Tensor,
        weights: torch.Tensor,
        objectives_greater_than: Optional[float | list[float]] = None
) -> Union[int, None]:

    n_objs = objective_values.shape[DataShape.index_dimensions]

    if type(objectives_greater_than) is float:

        large_enough_objective_values = objective_values > objectives_greater_than

    elif type(objectives_greater_than) is list:

        large_enough_objective_values = objective_values > torch.tensor(objectives_greater_than)

    else:
        raise ValueError

    large_enough_objective_rows = large_enough_objective_values.sum(dim=DataShape.index_dimensions) == n_objs

    if large_enough_objective_rows.max() == 0:
        # Could alternatively raise exception but might be overkill
        return None

    filtered_objective_values = objective_values * large_enough_objective_rows.unsqueeze(dim=DataShape.index_dimensions)

    max_index = int((filtered_objective_values * weights).sum(dim=DataShape.index_dimensions).argmax())

    return max_index


class ParetoOptimalPoints(TypedDict):
    variables: torch.Tensor
    objectives: torch.Tensor
    index: list[int]


def get_pareto_optimal_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        sort_by_max_weighted_sum: bool = False,
        noise_std_per_objective: Optional[torch.Tensor] = None,
        epsilon_n_sigma: float = 1.0
) -> ParetoOptimalPoints:
    """Return the Pareto-optimal (non-dominated) subset of the evaluated points.

    When `noise_std_per_objective` is supplied the algorithm applies ε-dominance
    (Laumanns et al. 2002, IEEE TEC 6(3)) with ε_j = epsilon_n_sigma · σ_j: point B
    dominates A only if B_j > A_j + ε_j for **all** objectives j.  The tolerance is
    set in units of the noise std, so epsilon_n_sigma=1 (default) means dominance must
    exceed 1σ of noise per objective.  epsilon_n_sigma=0 recovers the standard noiseless
    criterion.

    Reference: Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002).
    Combining convergence and diversity in evolutionary multiobjective optimization.
    IEEE Transactions on Evolutionary Computation, 6(3), 263–276.

    Parameters
    ----------
    noise_std_per_objective:
        Tensor of shape [n_objectives] with the known noise std per objective.
        When None the standard (noiseless) Pareto criterion is used.
    epsilon_n_sigma:
        Number of noise standard deviations that define ε_j = epsilon_n_sigma · σ_j.
        Default 1.0 (1σ margin).  Larger values are more conservative (fewer points
        kept on the front).  0 reduces to the standard noiseless criterion.
    """
    margin = (
        epsilon_n_sigma * noise_std_per_objective
        if noise_std_per_objective is not None
        else torch.zeros(objective_values.shape[DataShape.index_dimensions])
    )

    pareto_optimal_booleans = torch.ones(
        size=(objective_values.shape[DataShape.index_points],),
        dtype=torch.bool
    )
    for value_index, value in enumerate(objective_values):
        if pareto_optimal_booleans[value_index]:
            # Standard:  keep p  if EXISTS j: p_j > value_j
            # Noisy:     keep p  if EXISTS j: p_j > value_j − margin_j
            pareto_optimal_booleans[pareto_optimal_booleans.clone()] = torch.any(
                input=objective_values[pareto_optimal_booleans] > value - margin,
                dim=DataShape.index_dimensions
            )
            pareto_optimal_booleans[value_index] = True

    pareto_optimal_indices_tensor = pareto_optimal_booleans.nonzero().squeeze()

    if sort_by_max_weighted_sum:

        assert weights is not None, "Must be given weights to sort by weighted sum."

        pareto_optimal_values = objective_values[pareto_optimal_indices_tensor]
        weighted_sum_values = pareto_optimal_values @ weights
        sorted_index = weighted_sum_values.argsort()
        sorted_index = torch.flip(sorted_index, dims=(0,))
        pareto_optimal_indices_tensor = pareto_optimal_indices_tensor[sorted_index]

    pareto_optimal_indices = pareto_optimal_indices_tensor.tolist()

    return {
        'variables': variable_values[pareto_optimal_indices_tensor],
        'objectives': objective_values[pareto_optimal_indices_tensor],
        'index': pareto_optimal_indices
    }


def _convert_to_tensors(
        values: Mapping[str, Union[torch.Tensor, float]],
        names: list[str],
        expected_amount_points: int
) -> dict[str, torch.Tensor]:

    converted: dict[str, torch.Tensor] = {}
    for name in names:
        value = values[name]
        if not isinstance(value, torch.Tensor):
            converted[name] = torch.tensor([value])
        elif len(value.shape) < 1:
            converted[name] = value.unsqueeze(dim=0)
        else:
            converted[name] = value
        assert len(converted[name]) in (expected_amount_points, 0)
    return converted


def named_values_to_tensor(
        new_variable_values: Mapping[str, Union[torch.Tensor, float]],
        new_objective_values: Mapping[str, Union[torch.Tensor, float]],
        variable_names: list[str],
        objective_names: list[str],
        expected_amount_points: int
) -> tuple[torch.Tensor, torch.Tensor]:

    new_variable_values = _convert_to_tensors(
        values=new_variable_values,
        names=variable_names,
        expected_amount_points=expected_amount_points
    )
    new_objective_values = _convert_to_tensors(
        values=new_objective_values,
        names=objective_names,
        expected_amount_points=expected_amount_points
    )

    new_variable_values_tensor = torch.stack(
        [new_variable_values[name] for name in variable_names],
        dim=DataShape.index_dimensions
    )

    new_objective_values_tensor = torch.stack(
        [new_objective_values[name] for name in objective_names],
        dim=DataShape.index_dimensions
    )

    return (
        new_variable_values_tensor,
        new_objective_values_tensor
    )


def _named_variables_to_tensor(
        variable_values: Mapping[str, Union[torch.Tensor, float]],
        variable_names: list[str],
) -> torch.Tensor:
    """Convert a name→value dict to a [1, n_variables] tensor (variables only)."""
    converted = _convert_to_tensors(
        values=variable_values,
        names=variable_names,
        expected_amount_points=1
    )
    return torch.stack(
        [converted[name] for name in variable_names],
        dim=DataShape.index_dimensions
    )


def get_nadir_point(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
) -> torch.Tensor:
    pareto_values = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )['objectives']

    return pareto_values.min(dim=DataShape.index_points)[0]


def format_output_for_objective(
    suggested_variables: torch.Tensor,
    variable_names: list[str]
) -> dict[str, torch.Tensor]:

    suggested_variables_dict = {
        name: suggested_variables[:, variable_number] for (variable_number, name) in enumerate(variable_names)
    }

    return suggested_variables_dict
