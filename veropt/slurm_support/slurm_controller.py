import subprocess
import time
from veropt import load_optimiser
import dill
import click
import re


def check_jobs_running(job_name):
    job_name_id = job_name[0:4]
    squeue = subprocess.Popen("squeue", stdout=subprocess.PIPE)

    stdout = squeue.stdout

    time.sleep(2)  # seconds

    for raw_line in stdout:

        line = raw_line.decode("utf-8")

        if job_name_id in line:
            jobs_running = True
            break
        else:
            jobs_running = False

    return jobs_running


def cancel_jobs(job_name):
    job_name_id = str(job_name)[0:4]

    squeue = subprocess.Popen("squeue", stdout=subprocess.PIPE)

    stdout = squeue.stdout

    lines = []
    ids = []
    for raw_line in stdout:

        line = raw_line.decode("utf-8")

        if job_name_id in line:
            lines.append(line)
            ids.append(int(re.search(r'\d+', line).group()))

    for id in ids:
        scancel = subprocess.Popen(f"scancel {id}", shell=True)


@click.command()
@click.option('--optimiser')
@click.option('--job_name')
@click.option('--node_name')
def run(*args, **kwargs):
    optimiser_path = kwargs["optimiser"]
    job_name = kwargs["job_name"]
    node_name = kwargs["node_name"]

    optimiser = load_optimiser(optimiser_path)

    finished = False

    while finished is False:

        jobs_running = check_jobs_running(job_name)

        if jobs_running is False:

            if optimiser.current_step < optimiser.n_steps:

                suggested_step_filename = optimiser.run_opt_step()
                optimiser.save_optimiser()

                # chosen_modi_node = "001"
                # all_modi_nodes = ["000", "001", "002", "003", "004", "005", "006", "007"]
                shell_script = "veros.sh"

                spread_out = False

                with open(suggested_step_filename, 'rb') as file:
                    suggested_steps_pd = dill.load(file)

                no = 0
                for identifier, row in suggested_steps_pd.iterrows():

                    # if spread_out:
                    #     modi_node = all_modi_nodes[no % 8]
                    # else:
                    #     modi_node = chosen_modi_node

                    # sbatch = subprocess.Popen(f"sbatch -w {node_name} {shell_script} "
                    #                           f"{int(identifier)} {optimiser_path}", shell=True)
                    sbatch = subprocess.Popen(f"sbatch -w {node_name} {shell_script} "
                                              f"{int(identifier)}", shell=True)

                    no += 1

                time.sleep(2 * 60)  # 2 (* 60 seconds) minutes

                jobs_running = check_jobs_running(job_name)

                if jobs_running is False:
                    raise Exception("Couldn't start jobs. Check slurm .out files.")

            else:
                optimiser.load_new_data()
                optimiser.save_optimiser()
                finished = True

        else:

            time.sleep(10 * 60)  # 10 (* 60 seconds) minutes


if __name__ == '__main__':
    run()
