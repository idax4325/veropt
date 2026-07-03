from typing import Union, Optional

from veropt.interfaces.batch_manager import DirectBatchManager, SubmitBatchManager
from veropt.interfaces.experiment import Experiment
from veropt.interfaces.experiment_utility import ExperimentConfig
from veropt.interfaces.result_processing import ResultProcessor
from veropt.interfaces.simulation import SimulationRunner


def experiment(
        simulation_runner: SimulationRunner,
        result_processor: ResultProcessor,
        experiment_config: Union[str, ExperimentConfig],
        optimiser_config: Union[str, dict],
        batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None,
        continue_if_possible: bool = True,
        allow_automatic_json_updates: Optional[bool] = None
) -> Experiment:

    if continue_if_possible:
        experiment_ = Experiment.continue_if_possible(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=optimiser_config,
            batch_manager_class=batch_manager_class,
            allow_automatic_json_updates=allow_automatic_json_updates
        )

    else:
        experiment_ = Experiment.from_the_beginning(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=optimiser_config,
            batch_manager_class=batch_manager_class
        )

    return experiment_


def experiment_with_new_version(
        simulation_runner: SimulationRunner,
        result_processor: ResultProcessor,
        old_experiment_config: Union[str, ExperimentConfig],
        new_experiment_config: Union[str, ExperimentConfig],
        optimiser_config: Union[str, dict],
        batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None,
        allow_automatic_json_updates: Optional[bool] = None
) -> Experiment:

    experiment_ = Experiment.continue_with_new_version(
        simulation_runner=simulation_runner,
        result_processor=result_processor,
        old_experiment_config=old_experiment_config,
        new_experiment_config=new_experiment_config,
        optimiser_config=optimiser_config,
        batch_manager_class=batch_manager_class,
        allow_automatic_json_updates=allow_automatic_json_updates
    )

    return experiment_
