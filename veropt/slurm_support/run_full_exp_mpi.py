from mpi4py import MPI
import dill
MPI.pickle.__init__(dill.dumps, dill.loads)
from veropt import load_experiment
import click
from copy import deepcopy
import time
import torch
import numpy as np


def run_rep(bayes_optimiser, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    bayes_optimiser.run_all_opt_steps()


@click.command()
@click.option('--experiment')
def run(**kwargs):

    experiment_path = kwargs["experiment"]
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    master_rank = 0
    
    if rank == master_rank:
        master = True
    else:
        master = False

    if master:
        print(f"Running experiment with mpi, {size} processes")
    
    if master:
        experiment = load_experiment(file_name=experiment_path)
        remaining_runs = deepcopy(experiment.n_runs)
        finished = experiment.finished
    else:
        finished = None

    finished = comm.bcast(finished, root=master_rank)

    while finished is False:

        if master:
            # round_start_time = time.time()
            n_round_runs = min([size, remaining_runs])

            bayes_opts = []
            seeds = []
            config_inds = []
            rep_inds = []
            for proc_no in range(size):
                if proc_no < remaining_runs:
                    bayes_opts.append(deepcopy(experiment.bayes_opt_configs[experiment.current_config_no]))
                    seeds.append(int(torch.randint(1, int(2 ** 32 - 1), (1,))))
                    config_inds.append(deepcopy(experiment.current_config_no))
                    rep_inds.append(deepcopy(experiment.current_rep))

                    if experiment.current_rep < experiment.repetitions - 1:
                        experiment.current_rep += 1

                    elif experiment.current_config_no < experiment.n_configs - 1:
                        experiment.current_rep = 0
                        experiment.current_config_no += 1

                    else:
                        experiment.finished = True
                        finished = True
                else:
                    bayes_opts.append(None)
                    seeds.append(None)
                    config_inds.append(None)
                    rep_inds.append(None)
        else:
            bayes_opts = None
            seeds = None
            config_inds = None
            rep_inds = None

        finished = comm.bcast(finished, root=master_rank)

        bayes_opts = comm.scatter(bayes_opts, root=master_rank)
        seeds = comm.scatter(seeds, root=master_rank)

        if bayes_opts is not None:
            run_rep(bayes_opts, seeds)

        comm.barrier()

        bayes_opts = comm.gather(bayes_opts, root=master_rank)

        if master:
            for run_no, bayes_opt in enumerate(bayes_opts):
                if bayes_opt is not None:
                    config_ind = config_inds[run_no]
                    rep_ind = rep_inds[run_no]
                    experiment.bayes_opts[config_ind][rep_ind] = deepcopy(bayes_opt)
                    experiment.vals[config_ind][rep_ind] = bayes_opt.obj_func_vals_real_units()
                    experiment.best_vals[config_ind][rep_ind] = bayes_opt.best_val(in_real_units=True)

        if master:
            remaining_runs -= n_round_runs

            experiment.print_status()

            experiment.save_experiment()


run()
