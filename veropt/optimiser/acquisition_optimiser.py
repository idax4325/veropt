from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Self, TypedDict, Unpack, Callable

import numpy as np
import scipy
import torch
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass, rehydrate_object
from veropt.optimiser.utility import DataShape, _validate_typed_dict


class AcquisitionOptimiser(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'
    maximum_evaluations_per_step: int

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int
    ) -> None:

        self.bounds = bounds
        self.n_evaluations_per_step = n_evaluations_per_step

        assert 'maximum_evaluations_per_step' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'maximum_evaluations_per_step'."
        )

        assert n_evaluations_per_step <= self.maximum_evaluations_per_step, (
            f"This optimiser can only find {self.maximum_evaluations_per_step} point(s) at a time "
            f"but received a setting of {n_evaluations_per_step} evaluations per step."
        )

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

    def __call__(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:
        return self.optimise(acquisition_function)

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:
        self.bounds = new_bounds

    @abc.abstractmethod
    def optimise(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:
        pass

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'bounds': self.bounds,
                'n_evaluations_per_step': self.n_evaluations_per_step,
                'settings': self.get_settings().gather_dicts_to_save()
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        return cls.from_bounds_n_evaluations_per_step_and_settings(
            bounds=saved_state['bounds'],
            n_evaluations_per_step=saved_state['n_evaluations_per_step'],
            settings=saved_state['settings']
        )

    @classmethod
    @abc.abstractmethod
    def from_bounds_n_evaluations_per_step_and_settings(
            cls,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            settings: Mapping[str, Any],
    ) -> Self:
        pass

    @abc.abstractmethod
    def get_settings(self) -> SavableDataClass:
        pass


class TorchNumpyWrapper:
    def __init__(
            self,
            acquisition_function: AcquisitionFunction,
    ):
        self.acquisition_function = acquisition_function

    def __call__(
            self,
            variable_values: np.ndarray
    ) -> float:

        # TODO: Move somewhere prettier:
        #   - And make more general etc etc
        if len(variable_values.shape) == 1:
            variable_values = variable_values.reshape(1, len(variable_values))

        output = self.acquisition_function(
            variable_values=torch.tensor(variable_values)
        )

        return output.detach().item()


class DualAnnealingSettingsInputDict(TypedDict, total=False):
    max_iter: int


@dataclass
class DualAnnealingSettings(SavableDataClass):
    max_iter: int = 1_000


class DualAnnealingOptimiser(AcquisitionOptimiser):

    name = 'dual_annealing'
    maximum_evaluations_per_step = 1

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            **settings: Unpack[DualAnnealingSettingsInputDict]
    ):
        self.settings = DualAnnealingSettings(
            **settings
        )

        super().__init__(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step
        )

    def optimise(
            self,
            acquisition_function: AcquisitionFunction
    ) -> torch.Tensor:

        wrapped_acquisition_function = TorchNumpyWrapper(
            acquisition_function=acquisition_function
        )

        optimisation_result = scipy.optimize.dual_annealing(
            func=lambda x: - wrapped_acquisition_function(x),
            bounds=self.bounds.T,
            maxiter=self.settings.max_iter
        )

        candidates = torch.tensor(optimisation_result.x)

        # TODO: Make a general version in superclass?
        if len(candidates.shape) == 1:
            candidates = candidates.unsqueeze(DataShape.index_points)

        return candidates

    @classmethod
    def from_bounds_n_evaluations_per_step_and_settings(
            cls,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            settings: Mapping[str, Any]
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=DualAnnealingSettingsInputDict,
            object_name=cls.name
        )

        return cls(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step,
            **settings
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings


class ProximityPunishAcquisitionFunction(AcquisitionFunction):
    name = 'proximity_punish'

    def __init__(
            self,
            original_acquisition_function: AcquisitionFunction,
            other_points: list[torch.Tensor],
            scaling: float,
            alpha: float,
            omega: float
    ):
        self.original_acquisition_function = original_acquisition_function
        self.other_points = other_points

        self.multi_objective = original_acquisition_function.multi_objective

        self.scaling = scaling
        self.alpha = alpha
        self.omega = omega

        super().__init__(
            n_variables=self.original_acquisition_function.n_variables,
            n_objectives=self.original_acquisition_function.n_objectives
        )

    @classmethod
    def from_n_variables_n_objectives_and_settings(
            cls,
            n_variables: int,
            n_objectives: int,
            settings: Mapping[str, Any]
    ) -> Self:
        # Not expecting to need this since this class is just constructed right before it's needed
        raise NotImplementedError()

    def get_settings(self) -> SavableDataClass:

        # Might not need this but safer to have it since it's a required method

        @dataclass()
        class ProximityPunishFunctionSettings(SavableDataClass):
            scaling: float
            alpha: float
            omega: float

        return ProximityPunishFunctionSettings(
            scaling=self.scaling,
            alpha=self.alpha,
            omega=self.omega
        )

    def _add_proximity_punishment(
            self,
            point_variable_values: torch.Tensor,
            acquisition_value: torch.Tensor,
            other_points_variable_values: list[torch.Tensor]
    ) -> torch.Tensor:

        assert self.scaling is not None, "Scaling must have been calculated before adding the proximity punishment."

        proximity_punish = torch.zeros(len(acquisition_value))
        scaling = self.omega * self.scaling

        for other_point_variables in other_points_variable_values:

            proximity_punish += scaling * torch.exp(
                -(torch.sum((point_variable_values - other_point_variables) ** 2, dim=1) / (self.alpha ** 2))
            )

        return acquisition_value.detach() - proximity_punish

    def update_points(
            self,
            new_points: list[torch.Tensor]
    ) -> None:
        self.other_points = new_points

    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        original_acquisition_value = self.original_acquisition_function(
            variable_values=variable_values
        )

        modified_acquisition_value = self._add_proximity_punishment(
            point_variable_values=variable_values,
            acquisition_value=original_acquisition_value,
            other_points_variable_values=self.other_points
        )

        return modified_acquisition_value


class ProximityPunishSettingsInputDict(TypedDict, total=False):
    alpha: float
    omega: float
    refresh_setting: Literal['simple', 'advanced']
    verbose: bool


@dataclass
class ProximityPunishSettings(SavableDataClass):
    alpha: float = 0.7
    omega: float = 1.0
    refresh_setting: Literal['simple', 'advanced'] = 'advanced'
    verbose: bool = True  # TODO: Should ideally inherit this from optimiser


class ProximityPunishmentSequentialOptimiser(AcquisitionOptimiser):

    name = 'proximity_punishment'
    maximum_evaluations_per_step = 1_000_000

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            single_step_optimiser: AcquisitionOptimiser,
            scaling: Optional[float] = None,
            **settings: Unpack[ProximityPunishSettingsInputDict]
    ):

        self.single_step_optimiser = single_step_optimiser

        self.settings = ProximityPunishSettings(
            **settings
        )

        self.scaling: Optional[float] = scaling

        super().__init__(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.single_step_optimiser.__class__.__name__})"

    def optimise(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:

        self.refresh(
            acquisition_function=acquisition_function,
        )

        assert self.scaling is not None, "'refresh' failed to set scaling attribute"

        punishing_acquisition_function = ProximityPunishAcquisitionFunction(
            original_acquisition_function=acquisition_function,
            other_points=[],
            scaling=self.scaling,
            alpha=self.settings.alpha,
            omega=self.settings.omega
        )

        candidates: list[torch.Tensor] = []

        for candidate_no in range(self.n_evaluations_per_step):

            if self.settings.verbose and candidate_no == 0:
                print(
                    "Optimising acquisition function... ",
                    end="\r"
                )

            candidates.append(self.single_step_optimiser(
                acquisition_function=punishing_acquisition_function
            ))

            punishing_acquisition_function.update_points(
                new_points=candidates
            )

            if self.settings.verbose:
                print(
                    f"Optimising acquisition function... "
                    f"Found point {candidate_no + 1} of {self.n_evaluations_per_step}.",
                    end="\r"
                )

        if self.settings.verbose:
            print("\n")

        candidates_tensor = torch.cat(
            tensors=candidates,
            dim=DataShape.index_points
        )

        return candidates_tensor

    def refresh(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        if self.settings.verbose:
            print(
                "Finding scale for the acquisition optimiser...",
                end="\r"
            )

        if self.settings.refresh_setting == 'simple':
            self._refresh_scaling_simple(acquisition_function=acquisition_function)

        elif self.settings.refresh_setting == 'advanced':
            self._refresh_scaling_advanced(acquisition_function=acquisition_function)

        if self.settings.verbose:
            print("Found scale for the acquisition optimiser. \n")

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:

        self.single_step_optimiser.update_bounds(
            new_bounds=new_bounds
        )

        super().update_bounds(
            new_bounds=new_bounds
        )

    def get_modified_acquisition_values(
            self,
            *,
            acquisition_function: AcquisitionFunction,
            variable_values: torch.Tensor,
            points_to_punish: list[torch.Tensor],
    ) -> list[torch.Tensor]:

        assert self.scaling is not None, "Must have found scaling to call this method"

        punishing_acquisition_function = ProximityPunishAcquisitionFunction(
            original_acquisition_function=acquisition_function,
            other_points=[],
            scaling=self.scaling,
            alpha=self.settings.alpha,
            omega=self.settings.omega
        )

        modified_acquisition_values: list[torch.Tensor] = []

        for last_included_point_no in range(1, len(points_to_punish)):

            punishing_acquisition_function.update_points(
                new_points=points_to_punish[0:last_included_point_no]
            )

            modified_acquisition_values.append(punishing_acquisition_function(
                variable_values=variable_values,
            ))

        return modified_acquisition_values

    def _sample_acq_func(
            self,
            acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        n_acq_func_samples = 1000
        n_params = self.bounds.shape[1]

        random_coordinates = (
            (self.bounds[1] - self.bounds[0]) * torch.rand(n_acq_func_samples, n_params) + self.bounds[0]
        )

        samples = np.zeros(n_acq_func_samples)

        for coord_ind in range(n_acq_func_samples):
            sample = acquisition_function(
                variable_values=random_coordinates[coord_ind:coord_ind + 1, :]
            )
            samples[coord_ind] = sample.detach().item()

        return samples

    def _refresh_scaling_simple(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        acq_func_samples = self._sample_acq_func(acquisition_function=acquisition_function)

        sampled_std = acq_func_samples.std()

        self.scaling = sampled_std

    def _refresh_scaling_advanced(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        acq_func_samples = self._sample_acq_func(acquisition_function=acquisition_function)
        acq_func_samples = np.expand_dims(acq_func_samples, axis=1)

        min_clusters = 1
        min_scored_clusters = 2
        max_clusters = 7

        gaussian_fitters = {
            n_clusters: GaussianMixture(n_components=n_clusters)
            for n_clusters in range(min_clusters, max_clusters + 1)
        }
        scores = {
            n_clusters: 0.0
            for n_clusters in range(min_scored_clusters, max_clusters + 1)
        }

        for n_clusters in range(min_clusters, max_clusters + 1):

            gaussian_fitters[n_clusters].fit(acq_func_samples)

            if n_clusters >= min_scored_clusters:

                predictions = gaussian_fitters[n_clusters].predict(acq_func_samples)

                if np.unique(predictions).size > 1:
                    scores[n_clusters] = silhouette_score(
                        X=acq_func_samples,
                        labels=predictions
                    )
                else:
                    # TODO: Verify that this is okay
                    scores[n_clusters] = 0.0

        # Someone please make a prettier version of this >:)
        best_score_n_clusters = list(scores.keys())[np.array(list(scores.values())).argmax()]
        best_fitter = gaussian_fitters[best_score_n_clusters]

        # TODO: Finetune and test criterion for n_c=1
        if best_fitter.covariances_.max() * 3 > gaussian_fitters[1].covariances_[0]:
            best_score_n_clusters = 1
            best_fitter = gaussian_fitters[best_score_n_clusters]

        top_cluster_ind = best_fitter.means_.argmax()

        self.scaling = 2 * float(np.sqrt(best_fitter.covariances_[top_cluster_ind]))

    def gather_dicts_to_save(self) -> dict:
        save_dict = super().gather_dicts_to_save()
        save_dict['state']['single_step_optimiser'] = self.single_step_optimiser.gather_dicts_to_save()
        save_dict['state']['scaling'] = self.scaling

        return save_dict

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        single_step_optimiser = rehydrate_object(
            superclass=AcquisitionOptimiser,
            name=saved_state['single_step_optimiser']['name'],
            saved_state=saved_state['single_step_optimiser']['state']
        )

        return cls(
            bounds=saved_state['bounds'],
            n_evaluations_per_step=saved_state['n_evaluations_per_step'],
            single_step_optimiser=single_step_optimiser,
            scaling=saved_state['scaling']
        )

    @classmethod
    def from_bounds_n_evaluations_per_step_and_settings(
            cls,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            settings: Mapping[str, Any],
    ) -> Self:
        raise RuntimeError(
            "This acquisition optimiser can't be constructed just from bounds, n_evaluations_per_step and settings."
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings


def _calculate_proximity_punished_acquisition_values(
        proximity_punish_optimiser: ProximityPunishmentSequentialOptimiser,
        acquisition_function: AcquisitionFunction,
        normaliser_variables: Callable[[torch.Tensor], torch.Tensor],
        evaluated_point: torch.Tensor,
        variable_index: int,
        variable_array: torch.Tensor,
        suggested_points_variables: torch.Tensor,
        normalised: bool
) -> list[torch.Tensor]:

    n_suggested_points = suggested_points_variables.shape[DataShape.index_points]

    full_variable_array = evaluated_point.repeat(len(variable_array), 1)
    full_variable_array[:, variable_index] = variable_array

    suggested_points_variables_list = [suggested_points_variables[i, :] for i in range(n_suggested_points)]

    if normalised is False:
        full_variable_array = normaliser_variables(full_variable_array)

    modified_acquisition_values = proximity_punish_optimiser.get_modified_acquisition_values(
        acquisition_function=acquisition_function,
        variable_values=full_variable_array,
        points_to_punish=suggested_points_variables_list
    )

    return modified_acquisition_values
