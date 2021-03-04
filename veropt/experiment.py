from typing import List, Dict
from veropt.acq_funcs import *
from veropt import BayesOptimiser
import matplotlib.pyplot as plt
import dill
import datetime
import torch
from pathos.helpers import mp as pathos_multiprocess
import os
import time
import subprocess


class BayesExperiment:
    def __init__(self, bayes_opt_configs: List[BayesOptimiser], parameters: Dict, repetitions=5, file_name=None,
                 do_random_reps=True):
        self.bayes_opt_configs = bayes_opt_configs
        self.n_configs = len(bayes_opt_configs)
        self.bayes_opts = [[0]*repetitions] * len(bayes_opt_configs)
        self.repetitions = repetitions
        self.n_runs = self.n_configs * self.repetitions
        self.parameters = parameters
        self.do_random_reps = do_random_reps

        if file_name is None:
            key_string = ""
            for key in self.parameters.keys():
                key_string += key + "_"
            self.file_name = "Experiment_" + key_string + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pkl"
        else:
            self.file_name = file_name

        self.random_vals = None
        self.random_best_vals = None

        self.n_points_per_run = self.bayes_opt_configs[0].n_points
        self.best_vals = torch.zeros([self.n_configs, self.repetitions])
        self.vals = torch.zeros([self.n_configs, self.repetitions, self.n_points_per_run])

        self.obj_funcs = [0] * self.n_configs
        for config_no, config in enumerate(self.bayes_opt_configs):
            if config.obj_func.obj_names:
                self.obj_funcs[config_no] = config.obj_func.obj_names
            else:
                self.obj_funcs[config_no] = config.obj_func.__class__.__name__

        self.current_config_no = 0
        self.current_rep = 0
        self.finished = False

    def run_random_rep(self):

        optimiser = deepcopy(self.bayes_opt_configs[0])

        self.random_vals = torch.zeros([self.repetitions, self.n_points_per_run])
        self.random_best_vals = torch.zeros([self.repetitions])

        for rep in range(self.repetitions):
            init_steps = (optimiser.bounds[1] - optimiser.bounds[0]) \
                         * torch.rand(optimiser.n_points, optimiser.n_params) + optimiser.bounds[0]
            init_steps = init_steps.unsqueeze(0)
            init_vals = optimiser.obj_func.run(init_steps)
            self.random_vals[rep] = init_vals
            self.random_best_vals[rep] = torch.max(init_vals)

    def run_rep(self, save=True):

        if not self.finished:

            bayes_opt_config = self.bayes_opt_configs[self.current_config_no]

            bayes_optimiser = deepcopy(bayes_opt_config)
            bayes_optimiser.run_all_opt_steps()
            self.bayes_opts[self.current_config_no][self.current_rep] = bayes_optimiser
            self.vals[self.current_config_no][self.current_rep] = bayes_optimiser.obj_func_vals_real_units().flatten()
            self.best_vals[self.current_config_no, self.current_rep] = bayes_optimiser.best_val(in_real_units=True)
            self.print_status()

            if self.current_rep < self.repetitions - 1:
                self.current_rep += 1

            elif self.current_config_no < self.n_configs - 1:
                self.current_rep = 0
                self.current_config_no += 1

            else:
                self.finished = True

            if save:
                self.save_experiment()

    def run_full_experiment(self, save=True):

        if self.do_random_reps:
            self.run_random_rep()

        while not self.finished:
            self.run_rep(save)

    @staticmethod
    def run_rep_parallel(bayes_optimiser: BayesOptimiser, queue: pathos_multiprocess.Queue, config_ind, rep_ind, seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        bayes_optimiser.run_all_opt_steps()
        # message = (config_ind, rep_ind, bayes_optimiser.best_val(in_real_units=True))
        vals = bayes_optimiser.obj_func_vals_real_units()
        best_vals = bayes_optimiser.best_val(in_real_units=True)
        message = (config_ind, rep_ind, vals, best_vals)
        # message = (config_ind, rep_ind, best_vals)
        queue.put(message)

    def run_full_exp_parallel(self, save=True):

        if self.do_random_reps:
            self.run_random_rep()

        if "SLURM_JOB_ID" in os.environ:
            n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            print(f"Doing {self.n_runs} opt-runs on slurm with {n_cpus} cpus.")
            self.run_full_exp_parallel_cluster(save)

        else:
            n_cpus = os.cpu_count()
            print(f"Doing {self.n_runs} opt-runs with {n_cpus} cpus.")
            self.run_full_exp_parallel_smp(save)

    def run_full_exp_parallel_cluster(self, save=True):

        # Note: 'save' is True no matter what this^ receives

        n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        node_name = os.environ["SLURM_JOB_NODELIST"]
        shell_script_name = "mpi_exp_1.sh"

        self.save_experiment()

        sbatch = subprocess.Popen(f"sbatch -w {node_name} --cpus-per-task={n_cpus} "
                                  f"{shell_script_name} {n_cpus} {self.file_name}", shell=True)

    def run_full_exp_parallel_smp(self, save=True):

        n_cpus = os.cpu_count()

        remaining_runs = deepcopy(self.n_runs)

        while self.finished is False:

            round_start_time = time.time()
            n_round_runs = np.min([n_cpus, remaining_runs])
            processes = [0] * n_round_runs
            # queue = pathos_multiprocess.Queue()
            queues = [0] * n_round_runs
            for queue_no in range(len(queues)):
                queues[queue_no] = pathos_multiprocess.Queue()

            for proc_no in range(n_round_runs):
                bayes_optimiser = deepcopy(self.bayes_opt_configs[self.current_config_no])
                seed = int(torch.randint(1, int(2**32 - 1), (1,)))
                processes[proc_no] = pathos_multiprocess.Process(target=self.run_rep_parallel,
                                                                 args=(bayes_optimiser, queues[proc_no], self.current_config_no,
                                                                       self.current_rep, seed))
                processes[proc_no].start()

                if self.current_rep < self.repetitions - 1:
                    self.current_rep += 1

                elif self.current_config_no < self.n_configs - 1:
                    self.current_rep = 0
                    self.current_config_no += 1

                else:
                    self.finished = True

            jobs_running = True
            procs_status = [5] * len(processes)
            last_waiting_n = 10e10
            while jobs_running:
                for proc_no, process in enumerate(processes):
                    process.join(timeout=1)
                    if process.is_alive():
                        procs_status[proc_no] = 1
                    else:
                        procs_status[proc_no] = 0

                for queue in queues:
                    while not queue.empty():
                        message = queue.get()
                        config_ind, rep_ind, vals, best_vals = message
                        # config_ind, rep_ind, best_vals = message

                        # self.bayes_opts[config_ind][rep_ind] = b_opt
                        self.vals[config_ind][rep_ind] = deepcopy(vals)
                        self.best_vals[config_ind][rep_ind] = deepcopy(best_vals)

                waiting_n = np.sum(np.count_nonzero(procs_status))
                if last_waiting_n != waiting_n:
                    current_time = time.time()
                    elapsed_time = (current_time - round_start_time) / 60.0
                    print(f"Waited for {elapsed_time} minutes in this round, "
                          f"for {waiting_n} processes out of {len(processes)}", flush=True)
                    last_waiting_n = deepcopy(waiting_n)

                if np.sum(procs_status) < 1:
                    jobs_running = False

            remaining_runs -= n_round_runs

            self.print_status()

            if save:
                self.save_experiment()

    def print_status(self):
        print(f"Finished repetition {self.current_rep+1} of {self.repetitions} "
              f"in optimiser config {self.current_config_no+1} of {self.n_configs}.", flush=True)

    def plot_mean_std(self):

        n_finished_points = deepcopy(self.current_config_no)
        if self.current_rep < self.repetitions - 1:
            n_finished_points -= 1
        points_left_to_plot = deepcopy(n_finished_points)

        for parameter in self.parameters:
            plotted_points = n_finished_points - points_left_to_plot
            plt.figure()

            if self.random_vals is not None:
                random_loc = self.parameters[parameter][0] - (self.parameters[parameter][1] - self.parameters[parameter][0])
                plt.plot(random_loc, self.random_best_vals.unsqueeze(0), '.r', alpha=0.2)
                plt.errorbar(random_loc, self.random_best_vals.mean(), self.random_best_vals.std(),
                             capsize=5, marker='*', color='red')
                plt.annotate(" Random \n  runs", (random_loc, self.random_best_vals.mean()))

            points_in_this_paramater = np.min([len(self.parameters[parameter]), points_left_to_plot])
            plt.errorbar(self.parameters[parameter][0:points_in_this_paramater],
                         self.best_vals[plotted_points:plotted_points + points_in_this_paramater].mean(axis=1),
                         yerr=self.best_vals[plotted_points:plotted_points + points_in_this_paramater].std(axis=1),
                         marker='*', linestyle='', capsize=5)
            plt.plot(self.parameters[parameter][0:points_in_this_paramater],
                     self.best_vals[plotted_points:plotted_points + points_in_this_paramater],
                     marker='.', color='black', linestyle='', alpha=0.2)

            plt.xlabel(parameter)
            plt.ylabel("Objective Function")

            plotted_points += points_in_this_paramater
            points_left_to_plot -= points_in_this_paramater

            if not points_left_to_plot > 0:
                break

    def plot_iteration(self, logscale=False):

        cu_maxs = self.vals.cummax(dim=2)[0].mean(dim=1)
        stds = self.vals.cummax(dim=2)[0].std(dim=1)

        n_finished_points = deepcopy(self.current_config_no)
        if self.current_rep < self.repetitions - 1:
            n_finished_points -= 1
        points_left_to_plot = deepcopy(n_finished_points)

        points_per_plot = 5

        for parameter in self.parameters:
            plotted_points = n_finished_points - points_left_to_plot

            points_in_this_paramater = np.min([len(self.parameters[parameter]), points_left_to_plot])
            points_left_in_this_parameter = deepcopy(points_in_this_paramater)

            while points_left_in_this_parameter > 0:

                points_to_plot = np.amin([points_per_plot, points_left_in_this_parameter])

                plt.figure()

                if self.random_vals is not None:
                    random_mean = self.random_vals.cummax(dim=1)[0].mean(dim=0)
                    random_std = self.random_vals.cummax(dim=1)[0].std(dim=0)
                    plt.plot(torch.arange(1, len(random_mean) + 1), random_mean, label='Random opt', color='black')
                    plt.fill_between(torch.arange(1, len(random_mean) + 1), random_mean - random_std,
                                     random_mean + random_std, alpha=0.1, color='black')

                cu_maxs_this_parameter = cu_maxs[plotted_points:plotted_points + points_to_plot]
                stds_this_parameter = stds[plotted_points:plotted_points + points_to_plot]

                if logscale:
                    plt.yscale('symlog')

                for run_no, cu_maxs_run in enumerate(cu_maxs_this_parameter):
                    start_point = (points_in_this_paramater - points_left_in_this_parameter)
                    plt.plot(torch.arange(1, len(cu_maxs_run) + 1), cu_maxs_run,
                             label=f'{parameter}: {self.parameters[parameter][run_no + start_point]:.2f}')
                    stds_run = stds_this_parameter[run_no]
                    plt.fill_between(torch.arange(1, len(stds_run) + 1), cu_maxs_run - stds_run, cu_maxs_run + stds_run,
                                     alpha=0.1)

                plt.xlabel("Point")
                plt.ylabel("Objective Function")
                plt.legend()

                # plotted_points += points_in_this_paramater
                # points_left_to_plot -= points_in_this_paramater

                plotted_points += points_to_plot
                points_left_to_plot -= points_to_plot
                points_left_in_this_parameter -= points_to_plot

            if not points_left_to_plot > 0:
                break

    @staticmethod
    def close_plots():
        plt.close("all")

    def save_experiment(self):
        with open(self.file_name, 'wb') as file:
            dill.dump(self, file)
        print(f"Experiment saved as {self.file_name}")
        print("\n")
        print("\n")


def load_experiment(file_name):
    """

    :rtype: BayesExperiment
    """
    with open(file_name, 'rb') as file:
        experiment = dill.load(file)
    return experiment