from typing import List, Dict, Union
from veropt.acq_funcs import *
from veropt import BayesOptimiser
import matplotlib.pyplot as plt
import dill
import datetime
import torch
try:
    import pathos
    from pathos.helpers import mp as pathos_multiprocess
except (ImportError, NameError, ModuleNotFoundError):
    pass
import os
import time
import subprocess
import sys
from sklearn import preprocessing


class BayesExperiment:
    def __init__(self, bayes_opt_configs: List[BayesOptimiser], parameters: Union[List, Dict], n_objs, repetitions=5,
                 obj_weights=None, file_name=None, do_random_reps=True, normaliser=None):

        self.bayes_opt_configs = bayes_opt_configs
        self.n_configs = len(bayes_opt_configs)

        self.bayes_opts = [[0]*repetitions] * len(bayes_opt_configs)
        self.repetitions = repetitions
        self.n_runs = self.n_configs * self.repetitions

        self.parameters = parameters
        self.do_random_reps = do_random_reps

        self.n_objs = n_objs
        self.obj_weights = obj_weights

        if self.obj_weights is None:
            self.obj_weights = torch.ones(self.n_objs) / self.n_objs

        if file_name is None:
            key_string = ""
            key_list = self.parameters.keys() if type(self.parameters) == dict else self.parameters
            for key in key_list:
                key_string += key + "_"
                if len(key_string) > 30:
                    break
            self.file_name = "Experiment_" + key_string + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pkl"
        else:
            self.file_name = file_name

        self.random_vals = None
        self.random_best_vals = None

        self.normed_rvals = None
        self.normed_rvals_wsums = None
        self.normed_rvals_wsums_best_vals = None

        self.n_points_per_run = self.bayes_opt_configs[0].n_points

        self.best_vals = torch.zeros([self.n_configs, self.repetitions, self.n_objs])
        self.vals = torch.zeros([self.n_configs, self.repetitions, self.n_points_per_run, self.n_objs])

        self.normed_vals = torch.zeros([self.n_configs, self.repetitions, self.n_points_per_run, self.n_objs])
        self.normed_best_vals = torch.zeros([self.n_configs, self.repetitions, self.n_objs])

        self.normed_wsums = torch.zeros([self.n_configs, self.repetitions, self.n_points_per_run])
        self.normed_wsum_best_vals = torch.zeros([self.n_configs, self.repetitions])

        self.obj_funcs = [0] * self.n_configs
        for config_no, config in enumerate(self.bayes_opt_configs):
            if config.obj_func.obj_names:
                self.obj_funcs[config_no] = config.obj_func.obj_names
            else:
                self.obj_funcs[config_no] = config.obj_func.__class__.__name__

        self.current_config_no = 0
        self.current_rep = 0
        self.finished = False

        if normaliser is None:
            normaliser = preprocessing.StandardScaler()

        self.normaliser = normaliser

        self.did_normalisation = False

    def run_random_rep(self):

        optimiser = deepcopy(self.bayes_opt_configs[0])

        self.random_vals = torch.zeros([self.repetitions, self.n_points_per_run, self.n_objs])
        self.random_best_vals = torch.zeros([self.repetitions, self.n_objs])

        for rep in range(self.repetitions):
            random_points = (optimiser.bounds[1] - optimiser.bounds[0]) \
                         * torch.rand(optimiser.n_points, optimiser.n_params) + optimiser.bounds[0]
            random_points = random_points.unsqueeze(0)
            random_vals = optimiser.obj_func.run(random_points)
            random_vals = random_vals.reshape(self.n_points_per_run, self.n_objs)
            self.random_vals[rep] = random_vals
            if self.n_objs == 1:
                self.random_best_vals[rep] = torch.max(random_vals, dim=0)[0]

    def run_rep(self, save=True):

        if not self.finished:

            bayes_opt_config = self.bayes_opt_configs[self.current_config_no]

            bayes_optimiser = deepcopy(bayes_opt_config)
            bayes_optimiser.run_all_opt_steps()
            self.bayes_opts[self.current_config_no][self.current_rep] = bayes_optimiser
            self.vals[self.current_config_no][self.current_rep] = bayes_optimiser.obj_func_vals_real_units()
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
    def run_rep_parallel(bayes_optimiser: BayesOptimiser, queue, config_ind, rep_ind, seed):

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
            # n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            # print(f"Doing {self.n_runs} opt-runs on slurm with {n_cpus} cpus.")
            self.run_full_exp_parallel_cluster()

        else:
            n_cpus = os.cpu_count()
            print(f"Doing {self.n_runs} opt-runs with {n_cpus} cpus.")
            self.run_full_exp_parallel_smp(save)

    @staticmethod
    def run_full_exp_parallel_cluster():

        print("Deprecated feature! Please use slurm_support/slurm_set_up.py to automatically run experiments with mpi."
              "Example: \n"
              "from veropt.slurm_support import slurm_set_up"
              "slurm_set_up.set_up_experiment(\"Experiment_VehicleSafety_2021_03_24_13_45_18.pkl\", \"aegir\")"
              "slurm_set_up.start_exp_run(\"node000\")")

        # n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        # node_name = os.environ["SLURM_JOB_NODELIST"]
        # shell_script_name = "mpi_exp_1.sh"
        #
        # self.save_experiment()
        #
        # sbatch = subprocess.Popen(f"sbatch -w {node_name} --cpus-per-task={n_cpus} "
        #                           f"{shell_script_name} {n_cpus} {self.file_name}", shell=True)

    def run_full_exp_parallel_smp(self, save=True):

        if 'pathos' in sys.modules:

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

        else:
            print("Could not run experiment in parallel because pathos is not imported"
                  "This is probably because it isn't installed.")

    def print_status(self):
        print(f"Finished repetition {self.current_rep+1} of {self.repetitions} "
              f"in optimiser config {self.current_config_no+1} of {self.n_configs}.", flush=True)

    def update_normed_vals(self):

        n_finished_configs = deepcopy(self.current_config_no) + 1

        if n_finished_configs > 1 and self.current_rep < self.repetitions - 1:
            n_finished_configs -= 1

        if n_finished_configs > 1:
            n_finished_reps = deepcopy(self.repetitions)
            flattened_vals = torch.flatten(deepcopy(self.vals[0:n_finished_configs]), start_dim=0, end_dim=2)

        else:
            n_finished_reps = self.current_rep + 1
            flattened_vals = torch.flatten(deepcopy(self.vals[0:n_finished_configs, 0:n_finished_reps]),
                                           start_dim=0, end_dim=2)

        self.normaliser.fit(flattened_vals)
        normed_vals_flat = self.normaliser.transform(flattened_vals)

        self.normed_vals = normed_vals_flat.reshape(n_finished_configs, n_finished_reps, self.n_points_per_run, self.n_objs)

        self.normed_vals = torch.tensor(self.normed_vals)

        self.obj_weights = self.obj_weights.type(dtype=torch.float64)

        weighted_sums = self.normed_vals @ self.obj_weights

        maxs, max_inds = weighted_sums.max(dim=2)
        max_inds = max_inds

        I, J = torch.arange(n_finished_configs).unsqueeze(1), torch.arange(n_finished_reps)
        self.best_vals = self.vals[I, J, max_inds]
        self.normed_best_vals = self.normed_vals[I, J, max_inds]

        self.normed_wsums = weighted_sums
        self.normed_wsum_best_vals = weighted_sums[I, J, max_inds]

        if self.random_vals is not None:

            normed_random_vals = self.normaliser.transform(self.random_vals[0:n_finished_reps].flatten(start_dim=0, end_dim=1))
            normed_random_vals = torch.tensor(normed_random_vals).reshape(
                n_finished_reps, self.n_points_per_run, self.n_objs)
            normed_random_vals_wsums = normed_random_vals @ self.obj_weights

            rv_maxs, rv_max_inds = normed_random_vals_wsums.max(dim=1)

            normed_random_vals_wsums_best_vals = normed_random_vals_wsums[np.arange(n_finished_reps), rv_max_inds]

            self.random_best_vals = self.random_vals[np.arange(n_finished_reps), rv_max_inds]

            self.normed_rvals = normed_random_vals
            self.normed_rvals_wsums = normed_random_vals_wsums
            self.normed_rvals_wsums_best_vals = normed_random_vals_wsums_best_vals

        self.did_normalisation = True

    def plot_mean_std(self):

        if (self.did_normalisation is False) and (self.n_objs > 1):
            self.update_normed_vals()

        def make_plot(p_is_dict):
            if p_is_dict or par_no == 0:
                plt.figure()

                if self.random_vals is not None:
                    if p_is_dict:
                        random_loc = \
                            self.parameters[parameter][0] - (self.parameters[parameter][1] - self.parameters[parameter][0])
                    else:
                        random_loc = -1.0

                    if self.n_objs > 1:
                        random_best_vals = self.normed_rvals_wsums_best_vals
                    else:
                        random_best_vals = self.random_best_vals.squeeze(1)

                    plt.plot(random_loc, random_best_vals.unsqueeze(0), '.r', alpha=0.2)
                    plt.errorbar(random_loc, random_best_vals.mean(), random_best_vals.std(),
                                 capsize=5, marker='*', color='red', label="Random Search" if not p_is_dict else "")
                    if p_is_dict:
                        plt.annotate(" Random \n  runs", (random_loc, self.random_best_vals.mean()))

            if p_is_dict:
                x_arr = self.parameters[parameter][0:configs_in_this_paramater]
            else:
                x_arr = range_arr[par_no]

            if self.n_objs > 1:
                best_vals = self.normed_wsum_best_vals
            else:
                best_vals = self.best_vals.squeeze(2)

            plt.errorbar(x_arr, best_vals[plotted_configs:plotted_configs + configs_in_this_paramater].mean(axis=1),
                         yerr=best_vals[plotted_configs:plotted_configs + configs_in_this_paramater].std(axis=1),
                         marker='*', linestyle='', capsize=5, label=parameter if not p_is_dict else "")
            plt.plot(x_arr, best_vals[plotted_configs:plotted_configs + configs_in_this_paramater],
                     marker='.', color='black', linestyle='', alpha=0.2)

            if not p_is_dict:
                plt.legend()

            if p_is_dict:
                plt.xlabel(parameter)
            else:
                plt.xlabel("Configurations")
                ax = plt.gca()
                ax.get_xaxis().set_ticks([])
                xlim_min = -1.5 if self.random_vals is not None else -0.5
                xlim_max = len(self.parameters) - 0.5
                plt.xlim([xlim_min, xlim_max])

            plt.ylabel("Objective Function")

        n_finished_configs = deepcopy(self.current_config_no) + 1

        if n_finished_configs > 1 and self.current_rep < self.repetitions - 1:
            n_finished_configs -= 1
        configs_left_to_plot = deepcopy(n_finished_configs)

        parameters_is_dict = type(self.parameters) == dict

        if not parameters_is_dict:
            range_arr = np.arange(0, len(self.parameters))

        for par_no, parameter in enumerate(self.parameters):
            plotted_configs = n_finished_configs - configs_left_to_plot

            len_par = len(self.parameters[parameter]) if parameters_is_dict else 1

            configs_in_this_paramater = np.min([len_par, configs_left_to_plot])

            make_plot(p_is_dict=parameters_is_dict)

            plotted_configs += configs_in_this_paramater
            configs_left_to_plot -= configs_in_this_paramater

            if not configs_left_to_plot > 0:
                break

        plt.show()

    def plot_iteration(self, max_configs_per_plot=5, plot_objs_separately=False, plot_mean_unc=True, logscale=False):

        if (self.did_normalisation is False) and (self.n_objs > 1):
            self.update_normed_vals()

        if self.n_objs > 1:
            plot_vals = self.normed_wsums
            if plot_objs_separately:
                plot_vals = self.normed_vals
                obj_cmap = plt.get_cmap("tab10")
                conf_line_styles = ['--', '-', '-.', ':'] * 10
                # hatch_styles = ['/', '', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] * 10
                hatch_styles = ['']*20
        else:
            plot_vals = self.vals.squeeze(3)
            plot_objs_separately = False

        n_finished_configs = deepcopy(self.current_config_no) + 1
        if n_finished_configs > 1 and self.current_rep < self.repetitions - 1:
            n_finished_configs -= 1
        configs_left_to_plot = deepcopy(n_finished_configs)

        if n_finished_configs > 1:
            n_finished_reps = deepcopy(self.repetitions)

        else:
            n_finished_reps = self.current_rep + 1

        parameters_is_dict = type(self.parameters) == dict

        if parameters_is_dict:
            configs_cmap = plt.get_cmap('viridis')

        cu_maxs = plot_vals.cummax(dim=2)[0].mean(dim=1)
        if plot_mean_unc:
            stds = plot_vals.cummax(dim=2)[0].std(dim=1) / np.sqrt(n_finished_reps)
        else:
            stds = plot_vals.cummax(dim=2)[0].std(dim=1)

        for par_no, parameter in enumerate(self.parameters):
            plotted_configs = n_finished_configs - configs_left_to_plot

            len_par = len(self.parameters[parameter]) if parameters_is_dict else 1

            configs_in_this_paramater = np.min([len_par, configs_left_to_plot])
            configs_left_in_this_parameter = deepcopy(configs_in_this_paramater)

            while configs_left_in_this_parameter > 0:

                configs_to_plot = np.amin([max_configs_per_plot, configs_left_in_this_parameter])

                # TODO: Fix this
                # if not parameters_is_dict and (((par_no+1) % max_configs_per_plot) == 0 or par_no == 0):
                if True:

                    plt.figure()
                    if logscale:
                        plt.yscale('symlog')

                    if plot_objs_separately:
                        linestyle_count = 0

                    if self.random_vals is not None:

                        if self.n_objs > 1:
                            random_vals = self.normed_rvals_wsums
                            if plot_objs_separately:
                                random_vals = self.normed_rvals
                        else:
                            random_vals = self.random_vals.squeeze(2)

                        random_mean = random_vals.cummax(dim=1)[0].mean(dim=0)
                        if plot_mean_unc:
                            random_std = random_vals.cummax(dim=1)[0].std(dim=0) / np.sqrt(n_finished_reps)
                        else:
                            random_std = random_vals.cummax(dim=1)[0].std(dim=0)

                        if not plot_objs_separately:
                            plt.plot(torch.arange(1, len(random_mean) + 1), random_mean, label='Random Search',
                                     color='black')
                            plt.fill_between(torch.arange(1, len(random_mean) + 1), random_mean - random_std,
                                             random_mean + random_std, alpha=0.1, color='black')
                        else:
                            for obj_no in range(self.n_objs):
                                plt.plot(torch.arange(1, len(random_mean) + 1), random_mean[:, obj_no],
                                         label=f'Random Search, obj {obj_no+1}', color=obj_cmap(obj_no),
                                         linestyle=conf_line_styles[linestyle_count])

                                r_mean_low = random_mean[:, obj_no] - random_std[:, obj_no]
                                r_mean_high = random_mean[:, obj_no] + random_std[:, obj_no]
                                plt.fill_between(torch.arange(1, len(random_mean) + 1), r_mean_low,
                                                 r_mean_high, alpha=0.1, color=obj_cmap(obj_no),
                                                 hatch=hatch_styles[linestyle_count], ec='black')

                            linestyle_count += 1

                cu_maxs_this_parameter = cu_maxs[plotted_configs:plotted_configs + configs_to_plot]
                stds_this_parameter = stds[plotted_configs:plotted_configs + configs_to_plot]

                for run_no, cu_maxs_run in enumerate(cu_maxs_this_parameter):
                    start_point = (configs_in_this_paramater - configs_left_in_this_parameter)

                    label = f'{parameter}: {self.parameters[parameter][run_no + start_point]:.2f}' \
                        if parameters_is_dict else parameter
                    stds_run = stds_this_parameter[run_no]
                    if not plot_objs_separately:
                        if parameters_is_dict:
                            configs_cmap_vals = np.flip(np.linspace(0, 250, num=configs_in_this_paramater, dtype=np.int_))
                            color = configs_cmap(configs_cmap_vals[run_no])
                        else:
                            color = None
                        plt.plot(torch.arange(1, len(cu_maxs_run) + 1), cu_maxs_run, label=label,
                                 color=color)
                        plt.fill_between(torch.arange(1, len(stds_run) + 1), cu_maxs_run - stds_run,
                                         cu_maxs_run + stds_run, alpha=0.1,
                                         color=color)
                    else:
                        for obj_no in range(self.n_objs):
                            plt.plot(torch.arange(1, len(cu_maxs_run) + 1), cu_maxs_run[:, obj_no],
                                     label=label + f", obj {obj_no + 1}", color=obj_cmap(obj_no),
                                     linestyle=conf_line_styles[linestyle_count])

                            low = cu_maxs_run[:, obj_no] - stds_run[:, obj_no]
                            high = cu_maxs_run[:, obj_no] + stds_run[:, obj_no]
                            plt.fill_between(torch.arange(1, len(stds_run) + 1), low, high, color=obj_cmap(obj_no),
                                             alpha=0.1, hatch=hatch_styles[linestyle_count], ec='black')

                        linestyle_count += 1

                plt.xlabel("Evaluation No")
                plt.ylabel("Objective Function")
                plt.legend()

                plotted_configs += configs_to_plot
                configs_left_to_plot -= configs_to_plot
                configs_left_in_this_parameter -= configs_to_plot

            if not configs_left_to_plot > 0:
                break

        plt.show()

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