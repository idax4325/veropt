from typing import Any, Optional

from veropt.interfaces.constructors import experiment
from veropt.interfaces.result_processing import ResultProcessor
from veropt.interfaces.simulation import SimulationResult, SimulationRunner


class UserDefinedSimulationRunner(SimulationRunner):
    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: Optional[str],
            output_filename: str
    ) -> SimulationResult:

        # Write your own code here!
        # - You need to get your simulations ready to run and either submit them to slurm or run them directly
        # - If you set experiment_mode to 'local', veropt will expect you to run the simulations locally
        # - If you set experiment_mode to 'local_slurm', veropt will expect you to submit your simulations to slurm
        #   and automatically wait for them to finish

        raise NotImplementedError()


class UserDefinedResultProcessor(ResultProcessor):

    def calculate_objectives(
            self,
            result: SimulationResult
    ) -> dict[str, float]:

        # Write your own code here!
        #   - You need to define how your objectives are calculated from each simulation result
        #   - The 'SimulationResult' that is passed to this method contains the folder path for the current point
        #   - You will need to return a dictionary with the objective names and values for the current point

        raise NotImplementedError()

    def open_output_file(
            self,
            result: SimulationResult
    ) -> Any:

        # Write your own code here!
        #   - You need to define how your results are opened and return the resulting python object (e.g. an xr.Dataset)

        raise NotImplementedError()


# For an example of a SimulationRunner, see 'SlurmVerosRunner' under veropt/interfaces/slurm_simulation
simulation_runner = UserDefinedSimulationRunner()

# Write your own json configuration files but check out ours to see some examples.
# The experiment config controls problem settings (bounds, objective noise, etc.).
# To set a known constant observation noise per objective, add "noise_std" to the
# experiment config JSON, for example:
#
#   {
#       "experiment_name": "my_experiment",
#       "parameter_names": ["param1", "param2"],
#       "parameter_bounds": {"param1": [0.0, 1.0], "param2": [-1.0, 1.0]},
#       "path_to_experiment": "path/to/experiment",
#       "experiment_mode": "local_slurm",
#       "output_filename": "output.nc",
#       "noise_std": {"my_objective": 0.05}
#   }
#
# When noise_std is set, veropt automatically uses the noisy acquisition function
# (qLogNoisyEHVI for multi-objective) and shows uncertainty ellipses on Pareto plots.
optimiser_config = "veropt/examples/example_optimiser_settings.json"
experiment_config = "veropt/interfaces/configs/veros_experiment_config.json"

# For an example of a ResultProcessor, see 'TestVerosResultProcessor' under veropt/interfaces/result_processing
result_processor = UserDefinedResultProcessor(
    objective_names=[
        'your_first_objective',
        'your_second_objective',
    ]
)

user_experiment = experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

user_experiment.run_experiment()
