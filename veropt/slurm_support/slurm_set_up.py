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


class MakeShellFilesTemplateZero(MakeShellFiles):

    def __init__(self, optimiser_name, partition_name, veros_file_name, execution_path, using_singularity=False,
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


def set_up(optimiser_name, partition_name, veros_file_name, make_new_slurm_controller=True, make_shell_files=True,
           shell_file_class=None, using_singularity=False,
           image_path=None, conda_env=None):
    package_path = sys.path[0]  # Could also use .__file__
    execution_path = sys.path[1]

    if make_new_slurm_controller:
        copyfile(package_path + "/slurm_controller.py", execution_path + "/slurm_controller.py")

    if make_shell_files:

        if shell_file_class is None:
            shell_file_class = MakeShellFilesTemplateZero(optimiser_name, partition_name, veros_file_name, execution_path,
                                                          using_singularity, image_path, conda_env)

        shell_file_class.write_shell_files()


def start_opt_run(node_name):
    sbatch = subprocess.Popen(
        f"sbatch -w {node_name} controller.sh {node_name}",
        shell=True, stdout=subprocess.PIPE)

    time.sleep(1)

    for i in range(50):
        line = sbatch.stdout.readline().decode("utf-8")
        if len(line) > 1:
            print(line)



