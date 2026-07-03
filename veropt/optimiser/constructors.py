from typing import Literal, Optional, TypedDict, Union, Unpack, get_args

import torch

from veropt.optimiser.acquisition import BotorchAcquisitionFunction, UpperConfidenceBoundOptionsInputDict
from veropt.optimiser.acquisition_optimiser import (
    AcquisitionOptimiser, DualAnnealingSettingsInputDict,
    ProximityPunishSettingsInputDict, ProximityPunishmentSequentialOptimiser
)
from veropt.optimiser.kernels import KernelInputDict, SingleKernelOptions
from veropt.optimiser.model import (
    AdamInputDict, GPyTorchFullModel, GPyTorchSingleModel,
    GPyTorchTrainingParametersInputDict, TorchModelOptimiser
)
from veropt.optimiser.normalisation import Normaliser, NormaliserChoice, get_normaliser_class
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettingsInputDict
from veropt.optimiser.prediction import BotorchPredictor
from veropt.optimiser.saver_loader_utility import get_all_subclasses
from veropt.optimiser.utility import _load_defaults, _validate_typed_dict

KernelOptimiserOptions = Literal['adam']

AcquisitionOptions = Literal['qlogehvi', 'qlogneHVI', 'ucb']
AcquisitionOptimiserOptions = Literal['dual_annealing']

AcquisitionSettings = UpperConfidenceBoundOptionsInputDict  # expand with more options when adding acq_funcs
AcquisitionOptimiserSettings = DualAnnealingSettingsInputDict  # expand when adding more options

ModelOptimiserSettings = AdamInputDict


class ProblemInformationRequired(TypedDict):
    n_variables: int
    n_objectives: int
    n_evaluations_per_step: int
    bounds: list[list[float]]


class ProblemInformation(ProblemInformationRequired, total=False):
    is_noisy: bool
    train_noise: bool


class GPytorchModelChoice(TypedDict, total=False):
    kernels: Union[SingleKernelOptions, list[SingleKernelOptions], list[GPyTorchSingleModel], None]
    kernel_settings: Optional['KernelInputDict']
    kernel_optimiser: Optional[KernelOptimiserOptions]
    kernel_optimiser_settings: Optional[ModelOptimiserSettings]
    training_settings: Optional[GPyTorchTrainingParametersInputDict]


class AcquisitionChoice(TypedDict, total=False):
    function: Optional[AcquisitionOptions]
    parameters: Optional[AcquisitionSettings]


class AcquisitionOptimiserChoice(TypedDict, total=False):
    optimiser: Optional[AcquisitionOptimiserOptions]
    optimiser_settings: Optional[AcquisitionOptimiserSettings]
    allow_proximity_punishment: bool
    proximity_punish_settings: Optional[ProximityPunishSettingsInputDict]


# TODO: Consider making a function that can give valid arguments to the user?
#   - Like something that prints out an overview of options
#   - Should probably live in some documentation somewhere actually...?


def bayesian_optimiser(
        n_initial_points: int,
        n_bayesian_points: int,
        n_evaluations_per_step: int,
        objective: Union[CallableObjective, InterfaceObjective],
        model: Union[GPyTorchFullModel, 'GPytorchModelChoice', None] = None,
        acquisition_function: Union[BotorchAcquisitionFunction, 'AcquisitionChoice', None] = None,
        acquisition_optimiser: Union[AcquisitionOptimiser, 'AcquisitionOptimiserChoice', None] = None,
        normaliser: Union[type[Normaliser], NormaliserChoice, None] = None,
        **kwargs: Unpack[OptimiserSettingsInputDict]
) -> BayesianOptimiser:

    problem_information: ProblemInformation = {
        'n_variables': objective.n_variables,
        'n_objectives': objective.n_objectives,
        'n_evaluations_per_step': n_evaluations_per_step,
        'bounds': objective.bounds.tolist(),
        'is_noisy': objective.noise_std is not None,
        'train_noise': objective.train_noise,
    }

    built_predictor = botorch_predictor(
        problem_information=problem_information,
        model=model,
        acquisition_function=acquisition_function,
        acquisition_optimiser=acquisition_optimiser,
    )

    if type(normaliser) is type:

        if issubclass(normaliser, Normaliser):
            normaliser_class = normaliser

        else:
            raise ValueError("Normaliser_class must be a subclass of Normaliser")

    else:

        normaliser_class = get_normaliser_class(
            normaliser_choice=normaliser  # type: ignore[arg-type]  # checked above with 'issubclass'
        )

    return BayesianOptimiser.from_the_beginning(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=objective,
        predictor=built_predictor,
        normaliser_class=normaliser_class,
        **kwargs
    )


def botorch_predictor(
        problem_information: ProblemInformation,
        model: Optional[Union[GPyTorchFullModel, 'GPytorchModelChoice']] = None,
        acquisition_function: Union[BotorchAcquisitionFunction, 'AcquisitionChoice', None] = None,
        acquisition_optimiser: Union[AcquisitionOptimiser, 'AcquisitionOptimiserChoice', None] = None
) -> BotorchPredictor:

    if isinstance(model, GPyTorchFullModel):

        built_model = model

    else:

        if model is not None:
            _validate_typed_dict(
                typed_dict=model,
                expected_typed_dict_class=GPytorchModelChoice,
                object_name='gpytorch_model'
            )

        built_model = gpytorch_model(
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
            train_noise=problem_information.get('train_noise', False),
            **model or {},
        )
    if isinstance(acquisition_function, BotorchAcquisitionFunction):

        built_acquisition_function = acquisition_function

    else:
        built_acquisition_function = botorch_acquisition_function(
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
            is_noisy=problem_information.get('is_noisy', False),
            **acquisition_function or {}
        )

    if isinstance(acquisition_optimiser, AcquisitionOptimiser):
        built_acquisition_optimiser = acquisition_optimiser

    else:
        built_acquisition_optimiser = acquisition_optimiser_with_proximity_punishment(
            bounds=problem_information['bounds'],
            n_evaluations_per_step=problem_information['n_evaluations_per_step'],
            **acquisition_optimiser or {}
        )

    return BotorchPredictor(
        model=built_model,
        acquisition_function=built_acquisition_function,
        acquisition_optimiser=built_acquisition_optimiser,
    )


def gpytorch_model(
        n_variables: int,
        n_objectives: int,
        kernels: Union[SingleKernelOptions, list[SingleKernelOptions], list[GPyTorchSingleModel], None] = None,
        kernel_settings: Union['KernelInputDict', list['KernelInputDict'], None] = None,
        train_noise: bool = False,
        kernel_optimiser: Optional[KernelOptimiserOptions] = None,
        kernel_optimiser_settings: Optional[ModelOptimiserSettings] = None,
        training_settings: Optional[GPyTorchTrainingParametersInputDict] = None,
) -> GPyTorchFullModel:

    single_model_list = gpytorch_single_model_list(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels=kernels,
        kernel_settings=kernel_settings,
        train_noise=train_noise
    )

    model_optimiser = torch_model_optimiser(
        kernel_optimiser=kernel_optimiser,
        settings=kernel_optimiser_settings,
    )

    return GPyTorchFullModel.from_the_beginning(
        n_variables=n_variables,
        n_objectives=n_objectives,
        single_model_list=single_model_list,
        model_optimiser=model_optimiser,
        **(training_settings or {})
    )


def gpytorch_single_model_list(
        n_variables: int,
        n_objectives: int,
        kernels: Union[SingleKernelOptions, list[SingleKernelOptions], list[GPyTorchSingleModel], None] = None,
        kernel_settings: Union['KernelInputDict', list['KernelInputDict'], None] = None,
        train_noise: bool = False
) -> list[GPyTorchSingleModel]:

    wrong_kernel_input_message = (
        "'kernels' must be either None, a list of GPyTorchSingleModel, a valid kernel option or "
        "a list of valid kernel choices"
    )

    if isinstance(kernels, list):

        assert len(kernels) == n_objectives, (
            f"Please specify a kernel choice for each objective. "
            f"Received {n_objectives} objectives but {len(kernels)} kernels."
        )

        if isinstance(kernels[0], str):

            for kernel in kernels:
                assert type(kernel) is str, wrong_kernel_input_message

            if kernel_settings is not None:
                assert type(kernel_settings) is list, (
                    "'kernel_settings' must be a list of dicts if 'kernels' is a list of strings."
                )
                assert len(kernel_settings) == n_objectives

            else:
                kernel_settings = [{}] * n_objectives

            single_model_list = []
            for kernel_no, kernel in enumerate(kernels):  # type: ignore[assignment]  # checked above, it's 'list[str]'
                single_model_list.append(gpytorch_single_model(
                    n_variables=n_variables,
                    kernel=kernel,  # type: ignore[arg-type]  # checked above, kernel is 'str'
                    settings=kernel_settings[kernel_no],
                    train_noise=train_noise
                ))

        elif isinstance(kernels[0], GPyTorchSingleModel):

            assert kernel_settings is None, "Cannot accept kernel settings for an already created model list."

            for kernel in kernels:  # type: ignore[assignment]  # list[GPyTorchSingleModel] checked above
                assert isinstance(kernel, GPyTorchSingleModel), wrong_kernel_input_message

            single_model_list = kernels  # type: ignore[assignment]  # (type is checked above, mypy can't follow it)

        else:
            raise ValueError(wrong_kernel_input_message)

    elif isinstance(kernels, str):

        if kernel_settings is not None:
            assert isinstance(kernel_settings, dict), (
                "'kernel_settings' must be None or a single dict if 'kernels' is a single string."
            )

        single_model_list = []
        for objective_no in range(n_objectives):
            single_model_list.append(gpytorch_single_model(
                n_variables=n_variables,
                kernel=kernels,
                settings=kernel_settings,
                train_noise=train_noise
            ))

    elif kernels is None:

        assert kernel_settings is None, "Cannot accept kernel settings without a specified kernel."

        single_model_list = []
        for objective_no in range(n_objectives):
            single_model_list.append(gpytorch_single_model(
                n_variables=n_variables,
                train_noise=train_noise
            ))

    else:
        raise ValueError(wrong_kernel_input_message)

    return single_model_list


def gpytorch_single_model(
        n_variables: int,
        kernel: Optional[SingleKernelOptions] = None,
        settings: Optional[KernelInputDict] = None,
        train_noise: bool = False
) -> GPyTorchSingleModel:

    settings = settings or {}

    if kernel is None:
        defaults = _load_defaults()

        return gpytorch_single_model(
            n_variables=n_variables,
            kernel=defaults['model']['kernel'],
            settings=settings,
            train_noise=train_noise
        )

    subclasses = get_all_subclasses(
        cls=GPyTorchSingleModel
    )

    for subclass in subclasses:

        if kernel == subclass.name:

            return subclass.from_n_variables_and_settings(
                n_variables=n_variables,
                settings=settings,
                train_noise=train_noise
            )

    # Shouldn't reach this point if kernel is recognised
    raise ValueError(
        f"Kernel '{kernel}' not recognised. Implemented kernels are: {get_args(SingleKernelOptions)}"
    )


def torch_model_optimiser(
        kernel_optimiser: Optional[KernelOptimiserOptions] = None,
        settings: Optional[ModelOptimiserSettings] = None
) -> TorchModelOptimiser:

    settings = settings or {}

    if kernel_optimiser is None:

        defaults = _load_defaults()

        return torch_model_optimiser(
            kernel_optimiser=defaults['model']['optimiser']
        )

    subclasses = get_all_subclasses(
        cls=TorchModelOptimiser
    )

    for subclass in subclasses:

        if kernel_optimiser == subclass.name:

            return subclass.from_settings(
                settings=settings
            )

    raise NotImplementedError(
        f"Kernel optimiser '{kernel_optimiser}' not recognised. "
        f"Implemented kernels are: {get_args(KernelOptimiserOptions)}"
    )


def botorch_acquisition_function(
        n_variables: int,
        n_objectives: int,
        is_noisy: bool = False,
        function: Optional[AcquisitionOptions] = None,
        parameters: Optional[AcquisitionSettings] = None
) -> BotorchAcquisitionFunction:

    if function is None:

        defaults = _load_defaults()

        if n_objectives > 1:

            default_key = 'noisy_multi_objective' if is_noisy else 'multi_objective'
            return botorch_acquisition_function(
                n_variables=n_variables,
                n_objectives=n_objectives,
                is_noisy=is_noisy,
                function=defaults['acquisition'][default_key]
            )

        elif n_objectives == 1:

            default_key = 'noisy_single_objective' if is_noisy else 'single_objective'
            return botorch_acquisition_function(
                n_variables=n_variables,
                n_objectives=n_objectives,
                is_noisy=is_noisy,
                function=defaults['acquisition'][default_key]
            )

        else:
            raise ValueError("'n_objectives' must be a positive integer above 0.")

    subclasses = get_all_subclasses(
        cls=BotorchAcquisitionFunction
    )

    for subclass in subclasses:

        if function == subclass.name:

            return subclass.from_n_variables_n_objectives_and_settings(
                n_variables=n_variables,
                n_objectives=n_objectives,
                settings=parameters or {}
            )

    raise ValueError(f"acquisition_choice must be None or {get_args(AcquisitionOptions)}")


def acquisition_optimiser_with_proximity_punishment(
        bounds: list[list[float]],
        n_evaluations_per_step: int,
        optimiser: Optional[AcquisitionOptimiserOptions] = None,
        optimiser_settings: Optional[AcquisitionOptimiserSettings] = None,
        allow_proximity_punishment: bool = True,
        proximity_punish_settings: Optional[ProximityPunishSettingsInputDict] = None
) -> AcquisitionOptimiser:

    if allow_proximity_punishment is False:
        assert proximity_punish_settings is None, (
            "Can't receive settings for proximity punishment if it's disabled"
        )

    if optimiser is None:

        assert optimiser_settings is None, (
            "Can't accept settings for acquisition function optimiser without a specified optimiser."
            f"Options are {get_args(AcquisitionOptimiserOptions)}."
        )

        # Could support this being False in the future
        assert allow_proximity_punishment is True, (
            "Must allow proximity punishment if using default optimiser."
        )

        defaults = _load_defaults()

        return acquisition_optimiser_with_proximity_punishment(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step,
            optimiser=defaults['acquisition_optimiser'],
            proximity_punish_settings=proximity_punish_settings
        )

    if optimiser == 'proximity_punish':
        raise ValueError(
            f"Can't choose 'proximity_punish' here. Options are {get_args(AcquisitionOptimiserOptions)}."
        )

    subclasses = get_all_subclasses(
        cls=AcquisitionOptimiser
    )

    for subclass_no, subclass in enumerate(subclasses):
        # Probably not necessary but more correct
        if subclass == ProximityPunishmentSequentialOptimiser:
            del subclasses[subclass_no]

    for subclass in subclasses:

        if optimiser == subclass.name:

            if n_evaluations_per_step <= subclass.maximum_evaluations_per_step:

                return subclass.from_bounds_n_evaluations_per_step_and_settings(
                    bounds=torch.tensor(bounds),
                    n_evaluations_per_step=n_evaluations_per_step,
                    settings=optimiser_settings or {}
                )

            else:

                if allow_proximity_punishment is True:

                    single_step_optimiser = subclass.from_bounds_n_evaluations_per_step_and_settings(
                        bounds=torch.tensor(bounds),
                        n_evaluations_per_step=1,
                        settings=optimiser_settings or {}
                    )

                    _validate_typed_dict(
                        typed_dict=proximity_punish_settings or {},
                        expected_typed_dict_class=ProximityPunishSettingsInputDict,
                        object_name='proximity_punish'
                    )

                    return ProximityPunishmentSequentialOptimiser(
                        bounds=torch.tensor(bounds),
                        n_evaluations_per_step=n_evaluations_per_step,
                        single_step_optimiser=single_step_optimiser,
                        **proximity_punish_settings or {}
                    )

                else:
                    raise ValueError(
                        f"Acquisition Optimiser '{subclass.name}' can only find "
                        f"{subclass.maximum_evaluations_per_step} point(s) per step. Either allow using proximity "
                        f"punish or choose a different acquisition function optimiser."
                    )

    raise NotImplementedError(
        f"Acquisition optimiser '{optimiser}' not recognised. Options are {get_args(AcquisitionOptimiserOptions)}."
    )
