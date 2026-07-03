from dataclasses import dataclass
from typing import TypedDict, Unpack, Mapping, Any, Optional, Self, Union, Literal

import gpytorch
import torch
from veropt.optimiser.model import GPyTorchSingleModel, change_interval_constraints
from veropt.optimiser.saver_loader_utility import SavableDataClass
from veropt.optimiser.utility import _validate_typed_dict


SingleKernelOptions = Literal[
    'matern', 'double_matern', 'rational_quadratic', 'rational_quadratic_and_matern',
    'SMK', 'spectral_delta'
]

KernelInputDict = Union[
    'MaternParametersInputDict', 'DoubleMaternParametersInputDict',
    'RQParametersInputDict', 'RQMaternParametersInputDict',
    'SMKParametersInputDict', 'SpectralDeltaParametersInputDict'
]


class MaternParametersInputDict(TypedDict, total=False):
    lengthscale_lower_bound: float
    lengthscale_upper_bound: float
    nu: float


@dataclass
class MaternParameters(SavableDataClass):
    lengthscale_lower_bound: float = 0.1
    lengthscale_upper_bound: float = 2.0
    nu: float = 2.5


class DoubleMaternParametersInputDict(TypedDict, total=False):
    lengthscale_long_lower_bound: float
    lengthscale_long_upper_bound: float
    nu_long: float
    lengthscale_short_lower_bound: float
    lengthscale_short_upper_bound: float
    nu_short: float


@dataclass
class DoubleMaternParameters(SavableDataClass):
    lengthscale_long_lower_bound: float = 0.1
    lengthscale_long_upper_bound: float = 2.0
    nu_long: float = 2.5
    lengthscale_short_lower_bound: float = 0.001
    lengthscale_short_upper_bound: float = 0.1
    nu_short: float = 2.5


class MaternKernel(GPyTorchSingleModel):

    name = 'matern'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[MaternParametersInputDict]
    ):

        self.settings = MaternParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([]),
            nu=self.settings.nu
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> 'MaternKernel':

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=MaternParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def _set_up_kernel_specific_constraints(self) -> None:

        self.change_lengthscale_constraints(
            lower_bound=self.settings.lengthscale_lower_bound,
            upper_bound=self.settings.lengthscale_upper_bound
        )

    def change_lengthscale_constraints(
            self,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initialised to call this method"

        change_interval_constraints(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            module=self.model_with_data.covar_module,
            parameter_name='raw_lengthscale'
        )

    def get_lengthscale(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return self.model_with_data.covar_module.lengthscale

    def get_settings(self) -> SavableDataClass:
        return self.settings


class DoubleMaternKernel(GPyTorchSingleModel):

    name = 'double_matern'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[DoubleMaternParametersInputDict]
    ):

        self.settings = DoubleMaternParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()

        kernel_long = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([]),
            nu=self.settings.nu_long
        )

        kernel_short = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([]),
            nu=self.settings.nu_short
        )

        kernel = kernel_long + kernel_short

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> 'DoubleMaternKernel':

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=DoubleMaternParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def _set_up_kernel_specific_constraints(self) -> None:

        self.change_lengthscale_constraints(
            kernel_number=0,
            lower_bound=self.settings.lengthscale_long_lower_bound,
            upper_bound=self.settings.lengthscale_long_upper_bound
        )

        self.change_lengthscale_constraints(
            kernel_number=1,
            lower_bound=self.settings.lengthscale_short_lower_bound,
            upper_bound=self.settings.lengthscale_short_upper_bound
        )

    def change_lengthscale_constraints(
            self,
            kernel_number: int,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initialised to use this method"

        change_interval_constraints(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            parameter_name='raw_lengthscale',
            module=self.model_with_data.covar_module.kernels[kernel_number]
        )

    def get_lengthscale(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        raise NotImplementedError()

    def get_settings(self) -> SavableDataClass:
        return self.settings


class RQParametersInputDict(TypedDict, total=False):
    lengthscale_lower_bound: float
    lengthscale_upper_bound: float
    alpha_lower_bound: Optional[float]
    alpha_upper_bound: Optional[float]


@dataclass
class RQParameters(SavableDataClass):
    lengthscale_lower_bound: float = 0.1
    lengthscale_upper_bound: float = 2.0
    alpha_lower_bound: Optional[float] = None
    alpha_upper_bound: Optional[float] = None


class RationalQuadraticKernel(GPyTorchSingleModel):

    name = 'rational_quadratic'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[RQParametersInputDict]
    ):

        self.settings = RQParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.RQKernel(
            ard_num_dims=n_variables
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=RQParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def _set_up_kernel_specific_constraints(self) -> None:

        self.change_lengthscale_constraints(
            lower_bound=self.settings.lengthscale_lower_bound,
            upper_bound=self.settings.lengthscale_upper_bound
        )

        if (self.settings.alpha_lower_bound is not None) and (self.settings.alpha_upper_bound is not None):
            self.change_alpha_constraints(
                lower_bound=self.settings.alpha_lower_bound,
                upper_bound=self.settings.alpha_upper_bound
            )

        elif (self.settings.alpha_lower_bound is not None) or (self.settings.alpha_upper_bound is not None):
            raise NotImplementedError("Currently only support setting both or none of alpha's bounds.")

    def change_alpha_constraints(
            self,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initialised to call this method"

        change_interval_constraints(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            module=self.model_with_data.covar_module,
            parameter_name='alpha'
        )

    def change_lengthscale_constraints(
            self,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initialised to call this method"

        change_interval_constraints(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            module=self.model_with_data.covar_module,
            parameter_name='raw_lengthscale'
        )

    def get_lengthscale(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return self.model_with_data.covar_module.lengthscale

    def get_alpha(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return self.model_with_data.covar_module.alpha

    def get_settings(self) -> SavableDataClass:
        return self.settings


class RQMaternParametersInputDict(TypedDict, total=False):
    rq_lengthscale_lower_bound: float
    rq_lengthscale_upper_bound: float
    alpha_lower_bound: Optional[float]
    alpha_upper_bound: Optional[float]
    matern_lengthscale_lower_bound: float
    matern_lengthscale_upper_bound: float


@dataclass
class RQMaternParameters(SavableDataClass):
    rq_lengthscale_lower_bound: float = 0.1
    rq_lengthscale_upper_bound: float = 2.0
    alpha_lower_bound: Optional[float] = None
    alpha_upper_bound: Optional[float] = None
    matern_lengthscale_lower_bound: float = 0.1
    matern_lengthscale_upper_bound: float = 2.0


# TODO: Probably need to implement combining kernel classes
#   - So we can avoid needing an extra class for every combination >:)
class RationalQuadraticMaternKernel(GPyTorchSingleModel):

    name = 'rational_quadratic_and_matern'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[RQMaternParametersInputDict]
    ):

        self.settings = RQMaternParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        rq_kernel = gpytorch.kernels.RQKernel(
            ard_num_dims=n_variables
        )

        matern_kernel = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            nu=0.5
        )

        kernel = gpytorch.kernels.ScaleKernel(rq_kernel) + gpytorch.kernels.ScaleKernel(matern_kernel)

        kernel.kernels[0].outputscale = 0.85
        kernel.kernels[1].outputscale = 0.15

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=RQMaternParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def _set_up_kernel_specific_constraints(self) -> None:

        assert self.model_with_data is not None, "Model must be initialised to call this method"

        change_interval_constraints(
            lower_bound=self.settings.rq_lengthscale_lower_bound,
            upper_bound=self.settings.rq_lengthscale_upper_bound,
            module=self.model_with_data.covar_module.kernels[0].base_kernel,
            parameter_name='raw_lengthscale'
        )

        if (self.settings.alpha_lower_bound is not None) and (self.settings.alpha_upper_bound is not None):
            change_interval_constraints(
                lower_bound=self.settings.alpha_lower_bound,
                upper_bound=self.settings.alpha_upper_bound,
                module=self.model_with_data.covar_module.kernels[0].base_kernel,
                parameter_name='raw_alpha'
            )

        elif (self.settings.alpha_lower_bound is not None) or (self.settings.alpha_upper_bound is not None):
            raise NotImplementedError("Currently only support setting both or none of alpha's bounds.")

        change_interval_constraints(
            lower_bound=self.settings.matern_lengthscale_lower_bound,
            upper_bound=self.settings.matern_lengthscale_upper_bound,
            module=self.model_with_data.covar_module.kernels[1].base_kernel,
            parameter_name='raw_lengthscale'
        )

    def get_lengthscale(self) -> dict[str, torch.Tensor]:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return {
            'rational_quadratic': self.model_with_data.covar_module.kernels[0].base_kernel.lengthscale,
            'matern': self.model_with_data.covar_module.kernels[1].base_kernel.lengthscale
        }

    def get_alpha(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return self.model_with_data.covar_module.kernels[0].base_kernel.alpha

    def get_settings(self) -> SavableDataClass:
        return self.settings

    def _set_up_trained_parameters(self) -> None:

        # Note: Overwriting this method to exclude the outputscales in the ScaleKernel's

        parameter_group_list = []

        assert self.model_with_data is not None, "Model must be initialised to use this function."

        if self.train_noise is True:
            raise NotImplementedError("Not implemented for this kernel")

        parameter_group_list.append(
            {'params': self.model_with_data.mean_module.parameters()}
        )

        parameter_group_list.append(
            {'params': self.model_with_data.covar_module.kernels[0].base_kernel.parameters()}
        )

        parameter_group_list.append(
            {'params': self.model_with_data.covar_module.kernels[1].base_kernel.parameters()}
        )

        self.trained_parameters = parameter_group_list


class SMKParametersInputDict(TypedDict, total=False):
    n_mixtures: int


@dataclass
class SMKParameters(SavableDataClass):
    n_mixtures: int = 4


class SpectralMixtureKernel(GPyTorchSingleModel):
    name = 'SMK'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[SMKParametersInputDict]
    ):

        self.settings = SMKParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.SpectralMixtureKernel(
            ard_num_dims=n_variables,
            num_mixtures=self.settings.n_mixtures,
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=SMKParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings

    def _set_up_kernel_specific_constraints(self) -> None:
        pass  # No kernel-specific constraints for SpectralMixtureKernel

    def initialise_model_with_data(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
    ) -> None:

        super().initialise_model_with_data(
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

        assert self.model_with_data is not None, "Model must be initialised after calling super"

        self.model_with_data.covar_module.initialize_from_data(
            train_x=train_inputs,
            train_y=train_targets,
        )


class SpectralDeltaParametersInputDict(TypedDict, total=False):
    n_deltas: int


@dataclass
class SpectralDeltaParameters(SavableDataClass):
    n_deltas: Optional[int] = None


class SpectralDeltaKernel(GPyTorchSingleModel):

    name = 'spectral_delta'

    def __init__(
            self,
            n_variables: int,
            train_noise: bool = False,
            **settings: Unpack[SpectralDeltaParametersInputDict]
    ):

        self.settings = SpectralDeltaParameters(
            **settings
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        if self.settings.n_deltas is not None:
            optional_settings = {'num_deltas': self.settings.n_deltas}
        else:
            optional_settings = {}
        kernel = gpytorch.kernels.SpectralDeltaKernel(
            num_dims=n_variables,
            **optional_settings
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables,
            train_noise=train_noise
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any],
            train_noise: bool = False
    ) -> Self:

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=SpectralDeltaParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            train_noise=train_noise,
            **settings
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings

    def _set_up_kernel_specific_constraints(self) -> None:
        pass  # No kernel-specific constraints for SpectralDeltaKernel

    def initialise_model_with_data(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
    ) -> None:

        super().initialise_model_with_data(
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

        assert self.model_with_data is not None, "Model must be initialised after calling super"

        self.model_with_data.covar_module.initialize_from_data(
            train_x=train_inputs,
            train_y=train_targets,
        )
