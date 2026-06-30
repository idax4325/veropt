import abc
import functools
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Optional, Self, Sequence, TypedDict, Unpack

import botorch
import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval, LessThan
from gpytorch.distributions import MultivariateNormal

from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass, rehydrate_object
from veropt.optimiser.utility import (
    _validate_typed_dict, check_variable_and_objective_shapes,
    check_variable_objective_values_matching, enforce_amount_of_positional_arguments,
    unpack_variables_objectives_from_kwargs
)


# TODO: Consider deleting this abstraction. Does it have a function at this point?
class SurrogateModel(metaclass=abc.ABCMeta):

    def __init__(
            self,
            n_variables: int,
            n_objectives: int
    ):
        self.n_variables = n_variables
        self.n_objectives = n_objectives

    @abc.abstractmethod
    def train_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:
        pass


class GPyTorchDataModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):  # type: ignore[misc]
    _num_outputs = 1

    def __init__(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            mean_module: gpytorch.means.Mean,
            kernel: gpytorch.kernels.Kernel
    ) -> None:

        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood
        )

        self.mean_module = mean_module
        self.covar_module = kernel

        self.to(tensor=train_inputs)  # making sure we're on the right device/dtype

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# TODO: Move to different file
def format_json_state_dict(
        state_dict: dict,
) -> dict:
    formatted_dict = state_dict.copy()

    for key, value in state_dict.items():

        if isinstance(value, str):

            if "inf" in value:
                formatted_dict[key] = torch.tensor(float(value))

        elif isinstance(value, list):

            for item_no, item in enumerate(value):
                if isinstance(item, str):
                    if "inf" in item:
                        value[item_no] = float(item)

            formatted_dict[key] = torch.tensor(value)

        elif isinstance(value, float):

            formatted_dict[key] = torch.tensor(value)

    return formatted_dict


_NOISE_CONSTRAINT_FLOOR: float = 1e-8


class GPyTorchSingleModel(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            mean_module: gpytorch.means.Mean,
            kernel: gpytorch.kernels.Kernel,
            n_variables: int,
            train_noise: bool = False
    ) -> None:

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.kernel = kernel

        self.n_variables = n_variables

        self.model_with_data: Optional[GPyTorchDataModel] = None

        self.trained_parameters: list[dict[str, Iterator[torch.nn.Parameter]]] = [{}]

        self.train_noise = train_noise

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

        assert 'name' != 'meta', (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"trained: {'yes' if self.model_with_data else 'no'}"
            f", settings: {self.get_settings()}"
            f")"
        )

    @classmethod
    @abc.abstractmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> Self:
        pass

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        train_noise: bool = saved_state.get('train_noise', False)

        model = cls.from_n_variables_and_settings(
            n_variables=saved_state['n_variables'],
            settings=saved_state['settings'],
            train_noise=train_noise
        )

        if len(saved_state['state_dict']) > 0:
            model.initialise_model_from_state_dict(
                train_inputs=torch.tensor(saved_state['train_inputs']),
                train_targets=torch.tensor(saved_state['train_targets']),
                state_dict=format_json_state_dict(saved_state['state_dict']),
            )

        return model

    @abc.abstractmethod
    def get_settings(self) -> SavableDataClass:
        pass

    def _set_up_trained_parameters(self) -> None:

        parameter_group_list = []

        assert self.model_with_data is not None, "Model must be initialised to use this function."

        if self.train_noise:

            parameter_group_list.append(
                {'params': self.model_with_data.parameters()}
            )

        else:

            parameter_group_list.append(
                {'params': self.model_with_data.mean_module.parameters()}
            )

            parameter_group_list.append(
                {'params': self.model_with_data.covar_module.parameters()}
            )

        self.trained_parameters = parameter_group_list

    @abc.abstractmethod
    def _set_up_kernel_specific_constraints(self) -> None:
        """Set kernel-specific parameter constraints (e.g. lengthscales, alpha).
        Called automatically by _set_up_model_constraints — do NOT call _set_up_noise_constraints here."""
        pass

    def _set_up_model_constraints(self) -> None:
        """Concrete template: applies noise constraints first, then delegates to kernel-specific constraints."""
        self._set_up_noise_constraints()
        self._set_up_kernel_specific_constraints()

    def _set_up_noise_constraints(self) -> None:
        """Set the numerical-floor noise constraint and initial near-zero noise value.
        Called automatically from _set_up_model_constraints — not to be called directly from kernels."""
        self.set_noise_constraint(lower_bound=_NOISE_CONSTRAINT_FLOOR)
        # Near-zero default; overwritten by _apply_physical_noise if noise_std is set.
        self.set_noise(_NOISE_CONSTRAINT_FLOOR)

    def initialise_model_with_data(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
    ) -> None:

        self.model_with_data = GPyTorchDataModel(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            kernel=self.kernel
        )

        self._set_up_trained_parameters()

        self._set_up_model_constraints()

    def initialise_model_from_state_dict(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
            state_dict: dict
    ) -> None:

        self.initialise_model_with_data(
            train_inputs=train_inputs,
            train_targets=train_targets
        )

        self.model_with_data.load_state_dict(  # type: ignore[union-attr]  # model initialised just before calling this
            state_dict=state_dict
        )

    def gather_dicts_to_save(self) -> dict:

        if self.model_with_data is not None:
            state_dict = self.model_with_data.state_dict()
            # train_inputs is a gpytorch tuple (X_tensor,); unpack to avoid extra batch dim on reload
            (train_inputs,) = self.model_with_data.train_inputs  # type: ignore[misc]  # gpytorch stubs as Module
            train_targets = self.model_with_data.train_targets

        else:
            state_dict = {}
            train_inputs = None
            train_targets = None

        # Always save train_noise for reconstruction; the actual noise value is re-derived at training time.
        return {
            'name': self.name,
            'state': {
                'state_dict': state_dict,
                'train_inputs': train_inputs,
                'train_targets': train_targets,
                'n_variables': self.n_variables,
                'settings': self.get_settings().gather_dicts_to_save(),
                'train_noise': self.train_noise,
            }
        }

    def set_noise_constraint(
            self,
            lower_bound: float
    ) -> None:

        # Default seems to be 1e-4
        #   - Would like to make sure we don't have noise when we try to set it to zero
        #   - Alternatively, setting it too low might risk numerical instability?

        assert self.model_with_data is not None, "Model must be initiated to change constraints"

        change_greater_than_constraint(
            lower_bound=lower_bound,
            parameter_name='raw_noise',
            module=self.likelihood.noise_covar
        )

    def set_noise(
            self,
            noise: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initiated to call this function"

        if noise < self.likelihood.noise_covar.raw_noise_constraint.lower_bound:
            noise = float(self.likelihood.noise_covar.raw_noise_constraint.lower_bound)

        self.model_with_data.likelihood.noise = torch.tensor(noise)


def change_interval_constraints(
        lower_bound: float,
        upper_bound: float,
        parameter_name: str,
        module: gpytorch.Module
) -> None:

    constraint = Interval(
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )

    module.register_constraint(
        param_name=parameter_name,
        constraint=constraint
    )


def change_greater_than_constraint(
        lower_bound: float,
        parameter_name: str,
        module: gpytorch.Module,
) -> None:

    constraint = GreaterThan(
        lower_bound=lower_bound
    )

    module.register_constraint(
        param_name=parameter_name,
        constraint=constraint
    )


def change_less_than_constraint(
        upper_bound: float,
        parameter_name: str,
        module: gpytorch.Module
) -> None:

    constraint = LessThan(
        upper_bound=upper_bound
    )

    module.register_constraint(
        param_name=parameter_name,
        constraint=constraint
    )


class TorchModelOptimiser(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self
    ) -> None:

        self.optimiser: Optional[torch.optim.Optimizer] = None

    def gather_dicts_to_save(self) -> dict:

        return {
            'name': self.name,
            'state': {
                'settings': self.get_settings().gather_dicts_to_save()
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        return cls.from_settings(
            settings=saved_state['settings']
        )

    @classmethod
    @abc.abstractmethod
    def from_settings(
            cls,
            settings: Mapping[str, Any]
    ) -> Self:
        ...

    @abc.abstractmethod
    def get_settings(self) -> SavableDataClass:
        ...

    @abc.abstractmethod
    def initialise_optimiser(
            self,
            parameters: Iterator[torch.nn.Parameter] | list[dict[str, Iterator[torch.nn.Parameter]]]
    ) -> None:
        ...


class AdamInputDict(TypedDict, total=False):
    learning_rate: float


@dataclass
class AdamParameters(SavableDataClass):
    learning_rate: float = 0.1


class AdamModelOptimiser(TorchModelOptimiser):

    name = 'adam'

    def __init__(
            self,
            **settings: Unpack[AdamInputDict]
    ) -> None:

        self.settings = AdamParameters(
            **settings
        )

        super().__init__()

    def get_settings(self) -> SavableDataClass:
        return self.settings

    def initialise_optimiser(
            self,
            parameters: Iterator[torch.nn.Parameter] | list[dict[str, Iterator[torch.nn.Parameter]]]
    ) -> None:

        self.optimiser = torch.optim.Adam(
            params=parameters,
            lr=self.settings.learning_rate
        )

    @classmethod
    def from_settings(
            cls,
            settings: Mapping[str, Any]
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=AdamInputDict,
            object_name=cls.name,
        )

        return cls(
            **settings
        )


class ModelMode(Enum):
    training = 1
    evaluating = 2


class GPyTorchTrainingParametersInputDict(TypedDict, total=False):
    loss_change_to_stop: float
    max_iter: int
    verbose: bool
    max_cholesky_size: Optional[int]


@dataclass
class GPyTorchTrainingParameters(SavableDataClass):
    loss_change_to_stop: float = 1e-6  # TODO: Find optimal value for this?
    max_iter: int = 10_000
    verbose: bool = True
    max_cholesky_size: Optional[int] = None


class GPyTorchFullModel(SurrogateModel, SavableClass):

    name = 'gpytorch_full_model'

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
            single_model_list: Sequence[GPyTorchSingleModel],
            model_optimiser: TorchModelOptimiser,
            settings: GPyTorchTrainingParameters
    ) -> None:

        self.settings = settings

        assert len(single_model_list) == n_objectives, "Number of objectives must match the length of the model list"

        self._model_list = single_model_list

        self._model: Optional[botorch.models.ModelListGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.LikelihoodList] = None

        self._marginal_log_likelihood: Optional[gpytorch.mlls.SumMarginalLogLikelihood] = None

        self._model_optimiser = model_optimiser

        if self._model_list[0].model_with_data is None:
            single_models_are_trained = False

        else:
            single_models_are_trained = True

            for model in self._model_list:
                assert model.model_with_data is not None

        if single_models_are_trained:
            self._initialise_model_likelihood_lists()

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    @classmethod
    def from_the_beginning(
            cls,
            n_variables: int,
            n_objectives: int,
            single_model_list: Sequence[GPyTorchSingleModel],
            model_optimiser: TorchModelOptimiser,
            **kwargs: Unpack[GPyTorchTrainingParametersInputDict]
    ) -> 'GPyTorchFullModel':

        training_settings = GPyTorchTrainingParameters(
            **kwargs
        )

        return cls(
            n_variables=n_variables,
            n_objectives=n_objectives,
            single_model_list=single_model_list,
            model_optimiser=model_optimiser,
            settings=training_settings
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        model_list = []

        for model_dict in saved_state['model_dicts'].values():
            model_list.append(rehydrate_object(
                superclass=GPyTorchSingleModel,
                name=model_dict['name'],
                saved_state=model_dict['state']
            ))

        assert len(model_list) == saved_state['n_objectives']

        settings = GPyTorchTrainingParameters.from_saved_state(
            saved_state=saved_state['settings']
        )

        model_optimiser = rehydrate_object(
            superclass=TorchModelOptimiser,
            name=saved_state['model_optimiser']['name'],
            saved_state=saved_state['model_optimiser']['state']
        )

        return cls(
            n_variables=saved_state['n_variables'],
            n_objectives=saved_state['n_objectives'],
            single_model_list=model_list,
            model_optimiser=model_optimiser,
            settings=settings
        )

    @staticmethod
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

            assert isinstance(args[0], GPyTorchFullModel)
            self: GPyTorchFullModel = args[0]

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

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}({[model.__class__.__name__ for model in self._model_list]})"
        )

    def __str__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"\n{''.join([f"{model} \n" for model in self._model_list])}"
            f"settings: {self.settings}\n"
            f")"
        )

    def __getitem__(
            self,
            model_no: int
    ) -> GPyTorchSingleModel:

        if model_no > len(self._model_list) - 1:
            raise IndexError()

        return self._model_list[model_no]

    def __len__(self) -> int:
        return len(self._model_list)

    @_check_input_dimensions
    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> list[MultivariateNormal]:

        previous_mode = self._mode

        self._set_mode_evaluate()

        assert self._likelihood is not None, "Model must be initiated to call it"
        assert self._model is not None, "Model must be initiated to call it"

        if self.settings.max_cholesky_size is not None:
            evaluate_context = gpytorch.settings.max_cholesky_size(self.settings.max_cholesky_size)
        else:
            evaluate_context = nullcontext()

        with evaluate_context:

            estimated_objective_values = self._likelihood(
                *self._model(
                    *([variable_values] * self.n_objectives)
                )
            )

        self._set_mode(model_mode=previous_mode)

        return estimated_objective_values

    def gather_dicts_to_save(self) -> dict:

        model_dicts: dict[str, dict] = {}

        for model_no, model in enumerate(self._model_list):
            model_dicts[f'model_{model_no}'] = model.gather_dicts_to_save()

        return {
            'name': self.name,
            'state': {
                'model_dicts': model_dicts,
                'settings': self.settings.gather_dicts_to_save(),
                'model_optimiser': self._model_optimiser.gather_dicts_to_save(),
                'n_variables': self.n_variables,
                'n_objectives': self.n_objectives,
            }
        }

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def train_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
            noise_std_in_model_space: Optional[torch.Tensor] = None,
            noise_std_min_in_model_space: Optional[torch.Tensor] = None,
            noise_std_max_in_model_space: Optional[torch.Tensor] = None,
    ) -> None:

        self.initialise_model(
            variable_values=variable_values,
            objective_values=objective_values
        )

        if noise_std_in_model_space is not None:
            self._apply_physical_noise(
                noise_std_in_model_space=noise_std_in_model_space
            )
        elif noise_std_min_in_model_space is not None or noise_std_max_in_model_space is not None:
            self._apply_noise_bounds(
                noise_std_min_in_model_space=noise_std_min_in_model_space,
                noise_std_max_in_model_space=noise_std_max_in_model_space
            )

        self._set_mode_train()

        self._marginal_log_likelihood = gpytorch.mlls.SumMarginalLogLikelihood(
            likelihood=self._likelihood,
            model=self._model
        )

        self._initialise_optimiser()

        self._train_backwards()

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def initialise_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        for objective_number in range(self.n_objectives):

            self._model_list[objective_number].initialise_model_with_data(
                train_inputs=variable_values,
                train_targets=objective_values[:, objective_number]
            )

        self._initialise_model_likelihood_lists()

    def _initialise_model_likelihood_lists(self) -> None:

        for model in self._model_list:
            assert model.model_with_data is not None, "Single models must be trained to use this function"
            assert model.model_with_data.likelihood is not None, "Single models must be trained to use this function"

        # TODO: Might need to look into more options here
        #   - Currently seems to be assuming independent models. Maybe need to add an option for this?
        #       - Probably would need a different class (GPyTorchIndependentModels vs the opposite)
        #       - Botorch has some options for this
        self._model = botorch.models.ModelListGP(
            *[model.model_with_data for model in self._model_list]  # type: ignore  # (type is checked above)
        )
        self._likelihood = gpytorch.likelihoods.LikelihoodList(
            *[model.model_with_data.likelihood for model in self._model_list]  # type: ignore[union-attr]
        )

    def _apply_physical_noise(
            self,
            noise_std_in_model_space: torch.Tensor
    ) -> None:
        """Pin every single model's noise to the physical variance derived from noise_std.
        Must NOT be called on the train=False reload path (state_dict already encodes correct noise)."""

        assert len(noise_std_in_model_space) == self.n_objectives, (
            f"noise_std_in_model_space length ({len(noise_std_in_model_space)}) "
            f"must match n_objectives ({self.n_objectives})."
        )

        for objective_index, single_model in enumerate(self._model_list):

            if single_model.model_with_data is None:
                continue

            physical_variance = float(noise_std_in_model_space[objective_index] ** 2)
            assert physical_variance > _NOISE_CONSTRAINT_FLOOR, (
                f"Physical noise variance {physical_variance:.2e} for objective {objective_index} "
                f"is below the numerical floor {_NOISE_CONSTRAINT_FLOOR:.2e}. "
                f"Increase noise_std to a physically meaningful value."
            )
            # Reset the noise constraint to the canonical floor before setting the value.
            # load_state_dict may have restored a stale lower_bound from an older JSON
            # (e.g. v3 where lower_bound = 0.99 * physical_variance).  Without this reset,
            # the raw_noise GPyTorch stores is calibrated against the wrong lower_bound and
            # a subsequent save would re-produce a v3-style state_dict.
            single_model.set_noise_constraint(lower_bound=_NOISE_CONSTRAINT_FLOOR)
            single_model.set_noise(physical_variance)

    def _apply_noise_bounds(
            self,
            noise_std_min_in_model_space: Optional[torch.Tensor],
            noise_std_max_in_model_space: Optional[torch.Tensor],
    ) -> None:
        """Constrain the learned noise within [min_variance, max_variance] for each objective.
        Called when train_noise=True and noise_std_min or noise_std_max is set."""

        for objective_index, single_model in enumerate(self._model_list):

            if single_model.model_with_data is None:
                continue

            min_var = (
                float(noise_std_min_in_model_space[objective_index] ** 2)
                if noise_std_min_in_model_space is not None
                else None
            )
            max_var = (
                float(noise_std_max_in_model_space[objective_index] ** 2)
                if noise_std_max_in_model_space is not None
                else None
            )
            lower = max(_NOISE_CONSTRAINT_FLOOR, min_var) if min_var is not None else _NOISE_CONSTRAINT_FLOOR

            if max_var is not None:
                assert max_var > lower, (
                    f"noise_std_max variance {max_var:.2e} for objective {objective_index} must exceed "
                    f"noise_std_min variance {lower:.2e}."
                )
                change_interval_constraints(
                    lower_bound=lower,
                    upper_bound=max_var,
                    parameter_name='raw_noise',
                    module=single_model.likelihood.noise_covar
                )
            else:
                single_model.set_noise_constraint(lower_bound=lower)

    def get_gpytorch_model(self) -> botorch.models.ModelListGP:

        assert self._model is not None, "Model must be initiated to use this function"

        return self._model

    @property
    def model_has_been_trained(self) -> bool:
        if self._model is None:
            return False
        else:
            return True

    def _train_backwards(self) -> None:

        assert self._model is not None, "Model must be initialised to use this function"
        assert self._marginal_log_likelihood is not None, "Model must be initialised to use this function"

        assert self._model_optimiser.optimiser is not None, "Model optimiser must be initiated to use this function"

        loss_difference = torch.tensor(1e5)  # initial values
        loss = torch.tensor(1e20)  # TODO: Find a way to make sure this number is always big enough
        assert self.settings.loss_change_to_stop < loss_difference
        iteration = 1

        # TODO: This should be 0 when using spectral delta (I think :)) )
        #   - So that's a little awkward, but we might not rly need spectral delta?
        #   - Maybe we can do a warning somewhere or just delete the kernel (and this setting?)
        if self.settings.max_cholesky_size is not None:
            training_context = gpytorch.settings.max_cholesky_size(self.settings.max_cholesky_size)
        else:
            training_context = nullcontext()

        with training_context:

            while bool(loss_difference > self.settings.loss_change_to_stop):

                self._model_optimiser.optimiser.zero_grad()

                assert isinstance(self._model.train_inputs, list)
                train_inputs: list = self._model.train_inputs

                output = self._model(*train_inputs)

                previous_loss = loss
                loss = -self._marginal_log_likelihood(  # type: ignore  # gpytorch seems to be missing type-hints
                    output,
                    self._model.train_targets
                )

                loss.backward()
                loss_difference = torch.abs(previous_loss - loss)

                self._model_optimiser.optimiser.step()

                if self.settings.verbose and (iteration % 10 == 0):
                    print(
                        f"Training model... Iteration {iteration} (of a maximum {self.settings.max_iter})"
                        f" - MLL: {loss.item():.3f}",
                        end="\r"
                    )

                iteration += 1
                if iteration > self.settings.max_iter:
                    warnings.warn("Stopped training due to maximum iterations reached.")
                    break

        if self.settings.verbose:
            print("\n")

    def _initialise_optimiser(self) -> None:

        parameters: list[dict[str, Iterator[torch.nn.Parameter]]] = []
        for model in self._model_list:
            parameters += model.trained_parameters

        self._model_optimiser.initialise_optimiser(
            parameters=parameters
        )

    def _set_mode_evaluate(self) -> None:

        assert self._model is not None, "Model must be initialised to set its mode."
        assert self._likelihood is not None, "Model must be initialised to set its mode."

        self._model.eval()
        self._likelihood.eval()

    def _set_mode_train(self) -> None:

        assert self._model is not None, "Model must be initialised to set its mode."
        assert self._likelihood is not None, "Model must be initialised to set its mode."

        self._model.train()
        self._likelihood.train()

    def _set_mode(
            self,
            model_mode: ModelMode
    ) -> None:

        if model_mode == ModelMode.evaluating:
            self._set_mode_evaluate()

        elif model_mode == ModelMode.training:
            self._set_mode_train()

    @property
    def _mode(self) -> ModelMode:

        assert self._model is not None, "Model must be initialised to get its mode."

        if self._model.training:
            return ModelMode.training

        else:
            return ModelMode.evaluating

    @property
    def multi_objective(self) -> bool:

        if self.n_objectives > 1:
            return True
        else:
            return False
