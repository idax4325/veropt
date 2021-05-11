import sys
from shutil import copyfile
import os
import stat
import numpy as np
import subprocess
import time


class MakeShellFiles:
    def write_shell_files(self):
        """
        Fill in in subclass!
        """
        pass

    @staticmethod
    def make_shell_file(file_path_name, string):
        with open(file_path_name, 'w+') as file:
            file.write(string)

        st = os.stat(file_path_name)
        os.chmod(file_path_name, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH )


class MakeShellFilesOptMODITemplate(MakeShellFiles):

    def __init__(self, optimiser_name, partition_name, veros_file_name, execution_path, using_singularity=True,
                 image_path=None, conda_env=None):
        self.optimiser_name = optimiser_name

        if type(partition_name) == str:
            self.partition_name_contr = partition_name
            self.partition_name_veros = partition_name

        elif type(partition_name) == list:
            self.partition_name_contr = partition_name[0]
            self.partition_name_veros = partition_name[1]

        self.veros_file_name = veros_file_name
        self.execution_path = execution_path
        self.using_singularity = using_singularity
        self.image_path = image_path
        self.conda_env = conda_env

        self.sim_job_name = None

    def write_shell_files(self):

        contr_job_name = "opt_controller"

        sim_id = np.random.randint(9999)
        sim_job_name = f"{sim_id}_ocean"
        self.sim_job_name = sim_job_name

        if self.using_singularity:

            self.make_shell_file(
                self.execution_path + "/controller.sh",
                self.make_string_singularity(contr_job_name, self.partition_name_contr, "controller_2.sh"))

            self.make_shell_file(
                self.execution_path + "/controller_2.sh",
                self.make_controller_string())

            self.make_shell_file(
                self.execution_path + "/veros.sh",
                self.make_string_singularity(sim_job_name, self.partition_name_veros, "veros_2.sh"))

            self.make_shell_file(
                self.execution_path + "/veros_2.sh",
                self.make_veros_string())

        else:
            self.make_shell_file(
                self.execution_path + "/controller.sh",
                self.make_controller_string(contr_job_name, self.partition_name_contr))

            self.make_shell_file(self.execution_path + "/veros.sh",
                                 self.make_veros_string(sim_job_name, self.partition_name_veros))

    @staticmethod
    def make_sbatch_options(job_name, partition_name):
        sh_string = f"""#SBATCH --job-name={job_name}
#SBATCH --partition={partition_name}\n"""
        return sh_string

    def make_string_singularity(self, job_name, partition_name, second_sh_file):
        sh_string = f"""#!/bin/bash
{self.make_sbatch_options(job_name, partition_name)}

srun singularity exec {self.image_path} ./{second_sh_file} $1"""

        return sh_string

    def make_controller_string(self, job_name=None, partition_name=None):
        if job_name and partition_name:
            sbatch_options = self.make_sbatch_options(job_name, partition_name)
        else:
            sbatch_options = ""

        if self.conda_env:
            conda_activate = f"source $CONDA_DIR/etc/profile.d/conda.sh \n" \
                             f"conda activate {self.conda_env} \n"
        else:
            conda_activate = ""

        sh_string = f"""#!/bin/bash
{sbatch_options}
{conda_activate}
python slurm_controller.py --optimiser {self.optimiser_name} --job_name {self.sim_job_name} --node_name $1"""

        return sh_string

    def make_veros_string(self, job_name=None, partition_name=None):
        if job_name and partition_name:
            sbatch_options = self.make_sbatch_options(job_name, partition_name)
        else:
            sbatch_options = ""

        if self.conda_env:
            conda_activate = f"source $CONDA_DIR/etc/profile.d/conda.sh \n" \
                             f"conda activate {self.conda_env} \n"
        else:
            conda_activate = ""

        sh_string = f"""#!/bin/bash
{sbatch_options}
## AUTOMATICALLY GENERATED! ##
# If your veros simulations require a parallel set-up, please change this file! #

{conda_activate}
python {self.veros_file_name} --optimiser {self.optimiser_name} --identifier $1"""

        return sh_string


class MakeShellFilesExpAegirTemplate(MakeShellFiles):
    def __init__(self, experiment_name, partition_name, execution_path, constraint="v1", n_cores=None):
        self.experiment_name = experiment_name
        self.partition_name = partition_name
        self.execution_path = execution_path

        if constraint not in ["v1", "v2", "v3"]:
            raise Exception("Constraint must be 'v1', 'v2' or 'v3'!")

        self.constraint = constraint

        if n_cores is None:

            if self.constraint == "v1":
                self.n_cores = 16
            elif self.constraint == "v2":
                self.n_cores = 32
            elif self.constraint == "v3":
                self.n_cores = 48
        else:
            self.n_cores = n_cores

    def write_shell_files(self):
        self.make_shell_file(self.execution_path + "/exp_slurm.sh", self.make_basic_sh())

    def make_basic_sh(self):

        sh_string = f"""#!/bin/bash -l
#SBATCH -p {self.partition_name}
#SBATCH -A ocean
#SBATCH --job-name=exp_veropt
#SBATCH --time=23:59:59
#SBATCH --constraint={self.constraint}
#SBATCH --nodes=1
#SBATCH --ntasks={self.n_cores}
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --exclusive

ml load anaconda/python3

conda init bash
source $HOME/.bashrc
conda activate pytorch

mpiexec -n {self.n_cores} python3 run_full_exp_mpi.py --experiment {self.experiment_name}"""

        return sh_string


def set_up(optimiser_name, partition_name, veros_file_name, make_new_slurm_controller=True, make_shell_files=True,
           shell_file_class=None, using_singularity=False, image_path=None, conda_env=None):
    execution_path = sys.path[0]
    package_path = sys.modules['veropt'].__path__[0]

    if make_new_slurm_controller:
        copyfile(package_path + "/slurm_support/slurm_controller.py", execution_path + "/slurm_controller.py")

    if make_shell_files:

        if shell_file_class is None:
            shell_file_class = MakeShellFilesOptMODITemplate(optimiser_name, partition_name, veros_file_name,
                                                             execution_path, using_singularity, image_path, conda_env)

        shell_file_class.write_shell_files()


def start_opt_run(node_name, sh_file_name=None):

    if sh_file_name is None:
        sh_file_name = "controller.sh"

    sbatch = subprocess.Popen(
        f"sbatch -w {node_name} {sh_file_name} {node_name}",
        shell=True, stdout=subprocess.PIPE)

    time.sleep(1)

    for i in range(50):
        line = sbatch.stdout.readline().decode("utf-8")
        if len(line) > 1:
            print(line)


def set_up_experiment(experiment_name, partition_name, constraint="v1", make_new_mpi_run_file=True,
                      make_shell_files=True, shell_file_class=None, n_cores=None):
    execution_path = sys.path[0]
    package_path = sys.modules['veropt'].__path__[0]

    if make_new_mpi_run_file:
        copyfile(package_path + "/slurm_support/run_full_exp_mpi.py", execution_path + "/run_full_exp_mpi.py")

    if make_shell_files:

        if shell_file_class is None:
            shell_file_class = MakeShellFilesExpAegirTemplate(experiment_name, partition_name, execution_path,
                                                              constraint=constraint, n_cores=n_cores)

            shell_file_class.write_shell_files()


def start_exp_run(node_name, sh_file_name=None):

    if sh_file_name is None:
        sh_file_name = "exp_slurm.sh"

    sbatch = subprocess.Popen(
        f"sbatch -w {node_name} {sh_file_name}", shell=True, stdout=subprocess.PIPE
    )

    time.sleep(1)

    for i in range(50):
        line = sbatch.stdout.readline().decode("utf-8")
        if len(line) > 1:
            print(line)

