from abc import ABC, abstractmethod
import json
import subprocess
import os
from typing import Optional, Literal

from veropt.interfaces.simulation import SimulationResult, Simulation, SimulationRunner
from veropt.interfaces.veros_utility import edit_veros_run_script
from veropt.interfaces.utility import Config


class VirtualEnvironmentManager(ABC):
    @abstractmethod
    def make_activation_command(self) -> str:
        ...

    def activate_virtual_environment(self) -> None:
        source = self.make_activation_command()
        dump = 'python -c "import os, json;print(json.dumps(dict(os.environ)))"'
        pipe = subprocess.Popen(
            args=['/bin/bash', '-c', '%s && %s' % (source, dump)],
            stdout=subprocess.PIPE)

        assert pipe.stdout is not None
        # os.environ stubs don't accept dict[str, Any] from json
        os.environ = json.loads(s=pipe.stdout.read())  # type: ignore[assignment]

    def run_in_virtual_environment(
            self,
            command_arguments: list[str],
            directory: str
    ) -> subprocess.CompletedProcess:
        self.activate_virtual_environment()
        os.chdir(directory)  # TODO: Consider if this is necessary and/or stable?
        env_copy = os.environ.copy()

        return subprocess.run(
            args=command_arguments,
            cwd=directory,
            env=env_copy,
            capture_output=True,
            text=True
        )


class Conda(VirtualEnvironmentManager):
    def __init__(
            self,
            path_to_conda: str,
            env_name: str
    ):
        self.path_to_conda = path_to_conda
        self.env_name = env_name

    def make_activation_command(self) -> str:
        return f"source {self.path_to_conda}/bin/activate {self.env_name}"


class Venv(VirtualEnvironmentManager):
    def __init__(
            self,
            path_to_env: str
    ):
        self.path_to_env = path_to_env

    def make_activation_command(self) -> str:
        return f"source {self.path_to_env}/bin/activate"


def virtual_environment_manager(
        manager: Literal["conda", "venv"],
        path_to_conda: Optional[str] = None,
        env_name: Optional[str] = None,
        path_to_env: Optional[str] = None,
) -> VirtualEnvironmentManager:

    if manager == "conda":
        assert path_to_conda is not None, \
            "Conda picked as virtual env manager, but path to conda is missing."
        assert env_name is not None, \
            "Conda picked as virtual env manager, but env name is missing."

        return Conda(
            path_to_conda=path_to_conda,
            env_name=env_name
        )

    elif manager == "venv":
        assert path_to_env is not None, \
            "Venv picked as virtual env manager, but path to env is missing."

        return Venv(path_to_env=path_to_env)


class LocalSimulation(Simulation):
    """Run a simulation in a specified environment as a subprocess."""
    def __init__(
            self,
            simulation_id: str,
            run_script_directory: str,
            run_command_arguments: list[str],
            env_manager: VirtualEnvironmentManager,
            output_filename: str
    ):

        self.id = simulation_id
        self.run_script_directory = run_script_directory
        self.run_command_arguments = run_command_arguments
        self.env_manager = env_manager
        self.output_filename = output_filename

    def run(
            self,
            parameters: dict[str, float]
    ) -> SimulationResult:

        result = self.env_manager.run_in_virtual_environment(
            command_arguments=self.run_command_arguments,
            directory=self.run_script_directory
        )

        stdout_file = os.path.join(self.run_script_directory, f"{self.id}.out")
        stderr_file = os.path.join(self.run_script_directory, f"{self.id}.err")

        with open(stdout_file, "w") as f:
            f.write(result.stdout)

        with open(stderr_file, "w") as f:
            f.write(result.stderr)

        return SimulationResult(
            simulation_id=self.id,
            parameters=parameters,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            return_code=result.returncode,
            output_directory=self.run_script_directory,
            output_filename=self.output_filename
        )


class MockSimulationConfig(Config):
    stdout_file: str = "test_stdout.txt"
    stderr_file: str = "test_stderr.txt"
    return_code: int = 0
    output_filename: str = "test_output.nc"
    output_directory: str = ""


class MockSimulationRunner(SimulationRunner):
    """A mock simulation runner for testing purposes."""
    def __init__(
            self,
            config: MockSimulationConfig
    ) -> None:
        self.config = config

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str = "",
            run_script_filename: Optional[str] = None,
            output_filename: str = ""
    ) -> SimulationResult:

        print(f"Running test simulation with parameters: {parameters} and config: {self.config.model_dump()}")

        return SimulationResult(
            simulation_id=simulation_id,
            parameters=parameters,
            stdout_file=self.config.stdout_file,
            stderr_file=self.config.stderr_file,
            output_directory=self.config.output_directory,
            output_filename=self.config.output_filename,
            return_code=self.config.return_code
        )


class LocalVerosConfig(Config):
    env_manager: Literal["conda", "venv"]
    env_name: Optional[str]
    path_to_conda: Optional[str]
    path_to_env: Optional[str]
    path_to_veros_executable: str
    backend: Literal["jax", "numpy"]
    device: Literal["cpu", "gpu"]
    float_type: Literal["float32", "float64"]
    keep_old_params: bool = False


class LocalVerosRunner(SimulationRunner):
    """Set up and run a Veros simulation in a local environment."""
    def __init__(
            self,
            config: LocalVerosConfig
    ) -> None:
        self.config = config

    def _make_command(
            self,
            run_script: str
    ) -> str:
        gpu_string = f"--backend {self.config.backend} --device {self.config.device}"
        command = f"{self.config.path_to_veros_executable} run {gpu_string}" \
                  f" --float-type {self.config.float_type} {run_script}"
        return command

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: Optional[str],
            output_filename: str
    ) -> SimulationResult:

        assert run_script_filename is not None, (
            "LocalVerosRunner requires a run_script_filename."
        )

        run_script = os.path.join(run_script_directory, f"{run_script_filename}.py")

        edit_veros_run_script(
            run_script=run_script,
            parameters=parameters
        ) if not self.config.keep_old_params else None

        command_arguments = self._make_command(run_script=run_script).split(" ")

        env_manager = virtual_environment_manager(
            manager=self.config.env_manager,
            path_to_conda=self.config.path_to_conda,
            env_name=self.config.env_name,
            path_to_env=self.config.path_to_env
        )

        simulation = LocalSimulation(
            simulation_id=simulation_id,
            run_script_directory=run_script_directory,
            run_command_arguments=command_arguments,
            env_manager=env_manager,
            output_filename=output_filename
        )

        result = simulation.run(parameters=parameters)

        return result
