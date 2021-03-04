from veropt.kernels import *
from sklearn import preprocessing
from veropt.acq_funcs import *
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import warnings
import dill
import datetime
import torch
import botorch


class PriorClass:

    def __init__(self, prior_list):
        self.prior_list = prior_list

    def prior_cdf(self, x):

        prior_vals = np.zeros(x.shape)

        for par_no, prior in enumerate(self.prior_list):
            if x.dim() == 3:
                prior_vals[0, :, par_no] = prior.cdf(x[0, :, par_no])
            else:
                prior_vals[:, par_no] = prior[par_no].cdf(x[:, par_no])

        return torch.tensor(prior_vals)


def prior_dists(init_vals, bounds, stds):
    priors = []
    for par_no, bound in enumerate(bounds.T):

        mean = init_vals[0, par_no]
        std = stds[0, par_no]

        a, b = (bound[0] - mean) / std, (bound[1] - mean) / std

        priors.append(truncnorm(a, b, loc=mean, scale=std))

    return priors


class ObjFunction:
    def __init__(self, function, bounds, n_params, n_objs, init_vals=None, stds=None, saver=None, loader=None,
                 var_names=None, obj_names=None):
        self.function = function  # Can be None
        self.bounds = bounds
        self.n_params = n_params
        self.n_objs = n_objs

        self.init_vals = init_vals
        self.stds = stds

        self.saver = saver
        self.loader = loader

        self.var_names = var_names

        if isinstance(obj_names, str):
            self.obj_names = [obj_names]
        else:
            self.obj_names = obj_names

    def run(self, point):
        if self.function is not None:
            return self.function(point)
        else:
            warnings.warn("An attempt was made to evaluate the objective function but no function has been set for it.")


class BayesOptimiser:

    def __init__(self, n_init_points, n_bayes_points, obj_func: ObjFunction, acq_func: AcqFunction = None, model=None,
                 obj_weights=None, using_priors=False, normalise=True, points_before_fitting=15, test_mode=False,
                 n_evals_per_step=1, file_name=None, verbose=True):

        self.n_init_points = n_init_points
        self.n_bayes_points = n_bayes_points
        self.normalise = normalise

        self.n_evals_per_step = n_evals_per_step

        if n_init_points % n_evals_per_step > 0:
            self.n_init_points = n_init_points - (n_init_points % n_evals_per_step)
            warnings.warn(f"The amount of init points is not divisable by the amount of points evaluated each step."
                          f"'n_init_points' has been changed from {n_init_points} to {self.n_init_points}")

        if n_bayes_points % n_evals_per_step > 0:
            self.n_bayes_points = n_bayes_points - (n_bayes_points % n_evals_per_step)
            warnings.warn(f"The amount of bayes points is not divisable by the amount of points evaluated each step."
                          f"'n_bayes_points' has been changed from {n_bayes_points} to {self.n_bayes_points}")

        self.n_points = self.n_init_points + self.n_bayes_points
        self.n_steps = self.n_points // self.n_evals_per_step

        self.obj_func = obj_func
        self.n_objs = obj_func.n_objs
        if self.n_objs > 1:
            self.multi_obj = True
        else:
            self.multi_obj = False
        self.n_params = obj_func.n_params

        if obj_weights is None:
            self.obj_weights = torch.ones(self.n_objs) / self.n_objs
        else:
            self.obj_weights = obj_weights

        if torch.is_tensor(obj_func.bounds):
            self.bounds = obj_func.bounds.reshape(2, self.n_params)
        else:
            self.bounds = torch.tensor(obj_func.bounds).reshape(2, self.n_params)

        self.obj_func_coords = torch.zeros([0, self.n_params])
        self.obj_func_vals = torch.zeros([0])

        if acq_func is None:
            self.acq_func = PredefinedAcqFunction(obj_func.bounds, self.n_objs, n_evals_per_step=n_evals_per_step)
        else:
            self.acq_func = acq_func
            if self.n_evals_per_step != self.acq_func.n_evals_per_step:
                raise Exception("Mismatch between requested evaluations per step in BayesOptimiser and "
                                "Acquisition Function.")

        if model is None:
            self.model = BayesOptModel(self.n_params, self.n_objs)
        else:
            self.model = model

        self.using_priors = using_priors

        if self.using_priors:
            self.stds = self.obj_func.stds
            self.init_vals = self.obj_func.init_vals

        self.made_init_steps = False

        # self.current_point = 0
        self.current_step = 0
        # self.current_point = deepcopy(n_evals_per_step)
        # self.current_step = 1
        self.points_evaluated = 0
        self.opt_mode = "init"
        self.data_fitted = False
        self.need_new_suggestions = True

        self.test_mode = test_mode
        self.verbose = verbose

        if points_before_fitting < self.n_init_points:
            self.points_before_fitting = points_before_fitting
        else:
            self.points_before_fitting = self.n_init_points

        self.suggested_steps = None
        self.suggested_steps_acq_val = None
        self.suggested_steps_filename = None

        if normalise:
            self.normaliser_x = preprocessing.StandardScaler()
            self.normaliser_y = preprocessing.StandardScaler()

        if file_name is None:
            self.file_name = "Optimiser_" + self.obj_func.__class__.__name__ + "_" + \
                            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pkl"

        else:
            self.file_name = file_name

    def suggest_opt_steps(self):

        self.check_opt_mode()

        if self.opt_mode == "init":

            if self.made_init_steps is False:
                self.find_init_steps()

            suggested_steps = self.init_steps[:, self.points_evaluated: self.points_evaluated + self.n_evals_per_step]

        else:  # elif self.mode == "bayes"

            if self.need_new_suggestions:
                self.suggest_bayes_steps()

            suggested_steps = deepcopy(self.suggested_steps)

        return suggested_steps

    def find_init_steps(self):
        if self.using_priors is False:
            self.init_steps = self.init_steps_random()

        else:  # elif self.using_priors:
            self.priors = prior_dists(self.init_vals, self.bounds, self.stds)
            self.prior_class = PriorClass(self.priors)
            self.init_steps = self.init_steps_priors()
            self.model.set_priors(self.prior_class)

        self.made_init_steps = True

    def suggest_bayes_steps(self):
        self.refresh_acq_func()

        if self.verbose:
            print("Finding maximum of the acquisition function...")

        suggested_steps = self.acq_func.suggest_point()

        if self.verbose:
            print("Done!")
            print("\n")
            print("\n")

        self.suggested_steps = suggested_steps.unsqueeze(0)
        self.suggested_steps_acq_val = self.acq_func.function(
            self.suggested_steps.reshape(self.n_evals_per_step, 1, self.n_params))

        self.need_new_suggestions = False

    def save_suggested_steps(self):
        if self.obj_func.saver:
            if not self.need_new_suggestions or self.opt_mode == 'init':
                suggested_steps = self.suggest_opt_steps()
                if self.normalise and self.data_fitted:
                    suggested_steps = self.normaliser_x.inverse_transform(suggested_steps.squeeze(0))
                    suggested_steps = torch.tensor(suggested_steps).unsqueeze(0)
                filenames = self.obj_func.saver(suggested_steps, self.current_step + 1)
                self.suggested_steps_filename = filenames

                # print(f"Saved suggested point(s) in {filenames}")
                # print("\n")
                # print("\n")

                return filenames
            else:
                print("No valid points to save!")
                print("\n")
                print("\n")
        else:
            warnings.warn("The objective function doesn't have a method to save suggested points!")

    def load_new_data(self):
        if self.obj_func.loader:
            new_x, new_y = self.obj_func.loader()
            if new_y is not None:
                self.add_new_points(new_x, new_y)
            # print(f"Loaded some points or something")
            # print("\n")
            # print("\n")
        else:
            warnings.warn("The objective function doesn't have a method to load new points!")

    def refresh_acq_func(self):
        acq_func_args = {}

        if self.acq_func.acqfunc_name == 'EI':
            acq_func_args['best_f'] = self.obj_func_vals.max()
        elif self.acq_func.acqfunc_name == 'EHVI' or self.acq_func.acqfunc_name == 'qEHVI':
            # TODO: Consider a better ref point
            acq_func_args['ref_point'] = [-1.0] * self.n_objs
            acq_func_args['partitioning'] = botorch.utils.multi_objective.box_decomposition.NondominatedPartitioning(
                self.n_objs, self.obj_func_vals
            )

        self.acq_func.refresh(self.model.model, **acq_func_args)

    def reshape_batch(self, new_x, new_y):
        if not torch.is_tensor(new_x):
            new_x = torch.tensor(new_x)

        if not torch.is_tensor(new_y):
            new_y = torch.tensor(new_y)

        amount_evals = new_x.numel() // self.n_params

        if new_x.dim() < 3:
            new_x = new_x.reshape(1, amount_evals, self.n_params)

        if new_y.dim() < 3:
            new_y = new_y.reshape(1, amount_evals, self.n_objs)

        return new_x, new_y

    def normalise_batch(self, new_x, new_y):
        new_x = torch.tensor(self.normaliser_x.transform(new_x.squeeze(0))).unsqueeze(0)
        new_y = torch.tensor(self.normaliser_y.transform(new_y.squeeze(0))).unsqueeze(0)
        return new_x, new_y

    def add_new_points(self, new_x, new_y):
        """
        Adds new point(s), updates/initialises the model and prints status
        :param new_x: New objective function coordinates to add
        :param new_y: New objective function values to add
        """

        new_x, new_y = self.reshape_batch(new_x, new_y)

        if self.normalise and self.data_fitted:
            new_x, new_y = self.normalise_batch(new_x, new_y)

        # if (self.current_step == 1) or (self.current_step == 2 and (self.obj_func.function is None)):
        if self.points_evaluated == 0:

            self.obj_func_coords = new_x
            self.obj_func_vals = new_y

        else:
            self.obj_func_coords = torch.cat([self.obj_func_coords, new_x], dim=1)
            self.obj_func_vals = torch.cat([self.obj_func_vals, new_y], dim=1)

        amount_evals = new_x.numel() // self.n_params

        self.points_evaluated += amount_evals

        self.current_step += 1
        # self.current_point += self.n_evals_per_step

        if amount_evals < self.n_evals_per_step:
            warnings.warn(f"Imported {amount_evals} points but expected {self.n_evals_per_step}.")

        if self.data_fitted:
            self.update_model()

        elif self.data_fitted is False and self.points_evaluated >= self.points_before_fitting:
            self.init_model()

        if self.verbose:
            self.print_status()

    def init_model(self):
        if self.normalise:
            self.fit_normaliser()
            self.init_normaliser()
        self.refit_model()
        self.refresh_acq_func()
        self.data_fitted = True

    def refit_model(self):
        self.model.refit_model(self.obj_func_coords, self.obj_func_vals)
        self.need_new_suggestions = True
        self.refresh_acq_func()

    # TODO: Automatically refit model every n steps?
    def update_model(self):
        self.model.update_model(self.obj_func_coords, self.obj_func_vals)
        self.refresh_acq_func()
        self.need_new_suggestions = True

    def evaluate_points(self, suggested_steps):
        if self.obj_func.function:

            if self.normalise and self.data_fitted:
                new_x = torch.tensor(self.normaliser_x.inverse_transform(suggested_steps))

            else:
                new_x = suggested_steps

            new_y = self.obj_func.run(new_x)

            self.need_new_suggestions = True

            return new_x, new_y
        else:
            warnings.warn("An attempt was made to evaluate the objective function but no function has been set for it.")

    # TODO: Change some of the names of the methods so it's more transparent where model is being updated?
    #  Basically adding _update_model to add_new_points and load_new_data
    def run_opt_step(self):

        if self.obj_func.function:

            suggested_steps = self.suggest_opt_steps()

            new_x, new_y = self.evaluate_points(suggested_steps)

            self.add_new_points(new_x, new_y)

        else:

            self.load_new_data()

            self.suggest_opt_steps()

            filenames = self.save_suggested_steps()

            return filenames

    def run_all_opt_steps(self):
        for step in range(self.current_step, self.n_steps):
            self.run_opt_step()

    def print_status(self):

        if self.n_evals_per_step == 1:

            if self.n_objs > 1:
                best_val_string = self.format_list(self.best_val().detach().tolist())
            else:
                best_val_string = str(self.best_val())

            print(f"Optimisation running in {self.opt_mode} mode"
                  f" at step {self.current_step} out of {self.n_steps}"
                  f" | Best value: {best_val_string}")

            print_string = f"Newest obj. func. value: {self.obj_func_vals[0, -1]:.2f}"
            print_string += " | Newest point: " + self.format_list(self.obj_func_coords[0, -1].detach().tolist())

            print(print_string)

            print("\n")

        else:

            if self.n_objs > 1:
                best_val_string = self.format_list(self.best_val().detach().tolist())
            else:
                best_val_string = f"{float(self.best_val()):.2f}"

            print(f"Optimisation running in {self.opt_mode} mode"
                  f" at step {self.current_step} out of {self.n_steps}"
                  f" | Best value: {best_val_string}")

            print_string = "Newest obj. func. values: " + self.format_list(
                self.obj_func_vals[0, -self.n_evals_per_step:].detach().tolist())
            print(print_string)

            print_string = "Newest points: "
            print_string += self.format_list(self.obj_func_coords[0, -self.n_evals_per_step:].detach().tolist())

            print(print_string)

            print("\n")

    def check_opt_mode(self):
        if self.points_evaluated < self.n_init_points:
            self.opt_mode = "init"
        else:
            self.opt_mode = "bayes"

    def fit_normaliser(self):
        # .fit might not actually do anything when it's StandardScaler
        self.normaliser_x.fit(self.obj_func_coords.squeeze(0))
        self.normaliser_y.fit(self.obj_func_vals.squeeze(0))

        self.obj_func_coords = self.normaliser_x.transform(self.obj_func_coords.squeeze(0))
        self.obj_func_coords = torch.tensor(self.obj_func_coords).unsqueeze(0)

        self.obj_func_vals = self.normaliser_y.transform(self.obj_func_vals.squeeze(0))
        self.obj_func_vals = torch.tensor(self.obj_func_vals).unsqueeze(0)

    def init_normaliser(self):

        if self.n_init_points > 0:
            self.init_steps = self.normaliser_x.transform(self.init_steps.squeeze(0))
            self.init_steps = torch.tensor(self.init_steps).unsqueeze(0)

        self.bounds = self.normaliser_x.transform(self.bounds)
        self.bounds = torch.tensor(self.bounds)

        if self.using_priors:
            self.init_vals = self.normaliser_x.transform(self.init_vals)
            self.init_vals = torch.tensor(self.init_vals)

            x1 = self.normaliser_x.transform(self.init_vals - self.stds)
            x2 = self.normaliser_x.transform(self.init_vals + self.stds)
            self.stds = (x2 - x1) / 2
            self.stds = torch.tensor(self.stds)

            self.priors = prior_dists(self.init_vals, self.bounds, self.stds)
            self.prior_class = PriorClass(self.priors)
            self.model.set_priors(self.prior_class)

        self.acq_func.change_bounds(self.bounds)

        if self.verbose:
            if self.n_objs > 1:
                best_val_string = self.format_list(self.best_val().detach().tolist())
            else:
                best_val_string = f"{float(self.best_val()):.2f}"

            print(f"Normalisation completed! 'Best value' changed to {best_val_string}.")
            print("\n")

    def obj_function_normalised(self, coords):
        # Implemented directly from old code, might be clumsy
        val = self.obj_func.function(torch.tensor(self.normaliser_x.inverse_transform(coords)))
        norm_val = self.normaliser_y.transform(val)
        return norm_val

    def calculate_prediction(self, var_ind, in_real_units=False, plot_samples=10):
        if self.suggested_steps is None or self.need_new_suggestions:
            max_ind = (self.obj_func_vals * self.obj_weights).sum(2).argmax()
            eval_point = deepcopy(self.obj_func_coords[0, max_ind])
            # print("Plotting for the point with highest known value.")
            title = "at the point with the highest known value"
        else:
            high_ind = self.suggested_steps_acq_val.argmax()
            eval_point = deepcopy(self.suggested_steps[0, high_ind:high_ind + 1]).squeeze(0)
            # print("Plotting for the suggested next step.")
            title = "at the suggested next step with highest acq val"

        n = 200

        # Note that this is de-normalised below by redefining so don't change this without changing that too.
        var_arr = np.linspace(self.bounds.T[var_ind][0], self.bounds.T[var_ind][1], num=n)

        if self.test_mode:
            fun_arr = np.zeros([n, self.n_objs])
        else:
            fun_arr = None

        coords_arr = np.zeros([n, len(self.bounds.T)])
        acq_fun_vals = np.zeros(n)
        current_point = deepcopy(eval_point)
        for var_val_no, var_val in enumerate(var_arr):
            if len(current_point) < len(self.bounds.T):
                current_point = current_point[0]
            current_point[var_ind] = var_val
            coords_arr[var_val_no] = current_point

            if self.test_mode:
                if not in_real_units:
                    fun_arr[var_val_no] = self.obj_function_normalised(current_point.unsqueeze(0))
                else:
                    fun_arr[var_val_no] = self.obj_func.run(torch.tensor(
                        self.normaliser_x.inverse_transform(current_point.unsqueeze(0))))

            acq_fun_vals[var_val_no] = self.acq_func.function(
                torch.tensor(coords_arr[var_val_no]).unsqueeze(0)).detach().numpy()

        samples = [[]] * self.n_objs
        samples = torch.zeros([self.n_objs, plot_samples, n])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_eval = self.model.eval(torch.tensor(coords_arr))
            # model_mean = self.model.likelihood(self.model.model(torch.tensor(coords_arr)))
            model_lower_std = [0.0] * self.n_objs
            model_upper_std = [0.0] * self.n_objs
            for obj_no in range(self.n_objs):
                model_lower_std[obj_no], model_upper_std[obj_no] = model_eval[obj_no].confidence_region()
                for sample_no in range(plot_samples):
                    # samples[obj_no].append(model_eval[obj_no].sample())
                    samples[obj_no][sample_no] = model_eval[obj_no].sample().squeeze(0)

        model_mean = [0] * self.n_objs

        for obj_no in range(self.n_objs):
            model_mean[obj_no] = model_eval[obj_no].loc.numpy().flatten()
            model_lower_std[obj_no] = model_lower_std[obj_no].flatten()
            model_upper_std[obj_no] = model_upper_std[obj_no].flatten()

        var_arr = var_arr.flatten()
        # if fun_arr is not None:
        #     fun_arr = fun_arr.flatten()

        if in_real_units and self.normalise:

            def fix_dims(np_arr_list):
                return np.expand_dims(np.stack(np_arr_list, axis=-1), axis=0)

            def fix_dims_2(unnormed_np_arr):
                normed_np_list = np.split(unnormed_np_arr, self.n_objs, axis=2)
                for np_arr_no in range(len(normed_np_list)):
                    normed_np_list[np_arr_no] = normed_np_list[np_arr_no].flatten()

                return normed_np_list

            model_mean = fix_dims_2(self.normaliser_y.inverse_transform(fix_dims(model_mean)))
            model_lower_std = fix_dims_2(self.normaliser_y.inverse_transform(fix_dims(model_lower_std)))
            model_upper_std = fix_dims_2(self.normaliser_y.inverse_transform(fix_dims(model_upper_std)))
            var_arr = np.linspace(self.bounds_real_units().T[var_ind][0], self.bounds_real_units().T[var_ind][1], num=n)
            # for obj_no in range(self.n_objs):
            #     for sample_no in range(plot_samples):
            #         samples[obj_no][sample_no] = self.normaliser_y.inverse_transform(samples[obj_no][sample_no])

            for sample_no in range(plot_samples):
                samples[:, sample_no, :] = torch.tensor(
                    self.normaliser_y.inverse_transform(samples[:, sample_no, :].unsqueeze(0).transpose(1, 2)))\
                    .transpose(1, 2).squeeze(0)

        return title, var_arr, model_mean, model_lower_std, model_upper_std, acq_fun_vals, fun_arr, eval_point, samples

    def plot_prediction(self, obj_ind, var_ind, in_real_units=False, plot_acq_func=True, logscale=False,
                        plot_samples=10):

        if self.data_fitted is False:

            print("The model hasn't been fitted yet!")

        else:

            self.model.set_eval()

            # TODO: Currently inefficient because this calculates preds for each objective but only one is used.
            #  Could optimise by saving calced vals.
            title, var_arr, model_mean_list, model_lower_std_list, model_upper_std_list, acq_fun_vals, fun_arr, \
                eval_point, samples = self.calculate_prediction(var_ind, in_real_units, plot_samples)

            # for obj_no in range(self.n_objs):

            model_mean = model_mean_list[obj_ind]
            model_lower_std = model_lower_std_list[obj_ind]
            model_upper_std = model_upper_std_list[obj_ind]

            plt.figure()

            if plot_acq_func:
                plt.subplot(211)

            if logscale:
                plt.yscale('symlog')

            obj_name = self.obj_func.obj_names[obj_ind] if self.obj_func.obj_names is not None \
                else f"Objective {obj_ind + 1}"

            if plot_acq_func:
                plt.title(obj_name + " " + title, pad=75.0)
            else:
                plt.title(obj_name + " " + title, pad=60.0)

            if self.n_params > 1:

                if not self.need_new_suggestions:
                    sugg_and_eval_coords = torch.cat([self.obj_func_coords, self.suggested_steps], dim=1)
                else:
                    sugg_and_eval_coords = self.obj_func_coords

                distances = []
                index_wo_var_ind = torch.arange(self.n_params) != var_ind
                for point_no in range(sugg_and_eval_coords.shape[1]):
                    distances.append(np.linalg.norm(
                        eval_point[index_wo_var_ind] - sugg_and_eval_coords[0, point_no, index_wo_var_ind]))

                # alpha_values = torch.tensor(deepcopy(distances))
                alpha_min = 0.2
                alpha_max = 0.7

                distances = torch.tensor(distances)

                norm_distances = ((distances - distances.min()) / distances.max()) / \
                                 ((distances - distances.min()) / distances.max()).max()

                norm_distances = 1 - norm_distances

                alpha_values = (alpha_max - alpha_min) * norm_distances + alpha_min

                alpha_values[alpha_values == alpha_max] = 1.0

                alpha_values_sugsteps = alpha_values[self.points_evaluated:]

            else:
                alpha_values = torch.ones([self.points_evaluated + self.n_evals_per_step])
                alpha_values_sugsteps = alpha_values[self.points_evaluated:]

            marker_style = {'marker': '*',
                            'markeredgewidth': 0.5,
                            'markersize': 8,
                            'linestyle': ''}

            if not in_real_units:
                obj_func_coords = self.obj_func_coords
                obj_func_vals = self.obj_func_vals
            else:
                obj_func_coords = self.obj_func_coords_real_units()
                obj_func_vals = self.obj_func_vals_real_units()

            # Only init points
            if self.points_evaluated < self.n_init_points:
                for point_no in range(self.points_evaluated):
                    plt.plot(obj_func_coords[0, point_no, var_ind],
                             obj_func_vals[0, point_no, obj_ind],
                             'b', label="Initial points" if point_no == 0 else "",
                             alpha=float(alpha_values[point_no]), **marker_style)

            # Init points and Bayes points
            else:
                # Init points
                for point_no in range(self.points_evaluated):
                    if point_no < self.n_init_points:
                        plt.plot(obj_func_coords[0, point_no, var_ind],
                                 obj_func_vals[0, point_no, obj_ind],
                                 'b', label="Initial points" if point_no == 0 else "",
                                 alpha=float(alpha_values[point_no]), **marker_style)

                    # Bayes points
                    else:
                        plt.plot(obj_func_coords[0, point_no, var_ind],
                                 obj_func_vals[0, point_no, obj_ind],
                                 'k', label="Bayes points" if point_no == self.n_init_points else "",
                                 alpha=float(alpha_values[point_no]), **marker_style)

            if self.need_new_suggestions is False:

                for point_no in range(self.suggested_steps.shape[1]):

                    if not in_real_units:
                        suggested_steps = self.suggested_steps
                    else:
                        suggested_steps = torch.tensor(self.normaliser_x.inverse_transform(self.suggested_steps))

                    suggested_step_np = suggested_steps[:, point_no].squeeze(0).detach().numpy()

                    if not self.test_mode:
                        # expec_val = self.model.eval(suggested_steps[:, point_no])[obj_no]
                        # lower, upper = expec_val.confidence_region()
                        expec_val_list = self.model.eval(self.suggested_steps[:, point_no])
                        lower = [0.0] * self.n_objs
                        upper = [0.0] * self.n_objs
                        for obj_no in range(self.n_objs):
                            lower[obj_no], upper[obj_no] = expec_val_list[obj_no].confidence_region()

                        if self.multi_obj:
                            lower = torch.cat(lower, dim=1).detach().numpy()
                            upper = torch.cat(upper, dim=1).detach().numpy()
                            expec_val = torch.cat([val.loc for val in expec_val_list], dim=1).detach().numpy()

                        else:
                            lower = lower[0].unsqueeze(0).detach().numpy()
                            upper = upper[0].unsqueeze(0).detach().numpy()
                            expec_val = expec_val_list[0].loc.unsqueeze(0).detach().numpy()

                        if in_real_units:
                            # TODO: Have to .loc.detach() before we transform (so add an else and move it up)
                            #  Aaaand we need to wait with picking [obj_no]
                            expec_val = self.normaliser_y.inverse_transform(expec_val)
                            lower = self.normaliser_y.inverse_transform(lower)
                            upper = self.normaliser_y.inverse_transform(upper)

                        expec_val = expec_val[:, obj_ind]
                        lower = lower[:, obj_ind]
                        upper = upper[:, obj_ind]

                        # lower = (expec_val.loc - lower).detach().numpy()
                        # upper = (upper - expec_val.loc).detach().numpy()
                        lower = expec_val - lower
                        upper = upper - expec_val

                        plt.errorbar(suggested_step_np[var_ind], expec_val, yerr=np.array([lower, upper]).reshape(2, 1),
                                     color='firebrick', capsize=5, linewidth=1.0,
                                     label="Suggested point" if point_no == 0 else "",
                                     alpha=float(alpha_values_sugsteps[point_no]), **marker_style)
                    else:
                        if in_real_units:
                            actual_val = self.obj_func.run(
                                suggested_steps[0, point_no:point_no+1])[obj_ind].reshape(1, )
                        else:
                            actual_val = self.obj_function_normalised(
                                suggested_steps[0, point_no:point_no+1])[0, obj_ind].reshape(1, )

                        plt.plot(suggested_step_np[var_ind].reshape(1,), actual_val, 'r*',
                                 label=f"Suggested point (real value)" if point_no == 0 else "",
                                 alpha=float(alpha_values_sugsteps[point_no]), **marker_style)

            plt.plot(var_arr, model_mean, label="Predicted function value")
            plt.fill_between(var_arr, model_lower_std, model_upper_std, color='green', alpha=.1,
                             label="Uncertainty")

            if self.test_mode:
                plt.plot(var_arr, fun_arr[:, obj_ind], label="True function value", alpha=0.7)

            if plot_samples > 0:
                for sample_no in range(plot_samples):
                    plt.plot(var_arr, samples[obj_ind][sample_no].squeeze(0), color='grey', alpha=0.2)

            if self.obj_func.obj_names:
                ylabel = self.obj_func.obj_names[obj_ind]
            else:
                ylabel = "Objective Function"

            if not in_real_units:
                ylabel += " (normalised)"

            plt.ylabel(ylabel)

            # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
            if plot_acq_func:
                plt.legend(bbox_to_anchor=(0., 1.2, 1., .102), loc='lower left', ncol=2, mode="expand",
                           borderaxespad=0.)
            else:
                plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower left', ncol=2, mode="expand",
                           borderaxespad=0.)

            if in_real_units and not logscale:
                plt.ticklabel_format(scilimits=[-3, 4])

            if plot_acq_func:

                plt.subplot(212)

                if logscale:
                    plt.yscale('symlog')

                plt.plot(var_arr, acq_fun_vals, color='#686868')
                plt.ylabel('Acq. Func.')

            if self.obj_func.var_names:
                xlabel = self.obj_func.var_names[var_ind]
            else:
                xlabel = "Variable number " + str(var_ind+1)

            if not in_real_units:
                xlabel += " (normalised)"

            if in_real_units and not logscale:
                plt.ticklabel_format(scilimits=[-3, 4])

            plt.xlabel(xlabel)

            plt.tight_layout()

            # if self.using_priors:
            #     plt.figure()
            #     plt.plot(np.linspace(self.bounds.T[var_ind, 0], self.bounds.T[var_ind, 1]),
            #              self.priors[var_ind].pdf(np.linspace(self.bounds.T[var_ind, 0], self.bounds.T[var_ind, 1])))
            #     # TODO: Axis labels and maybe a different set-up for plotting these. Make a self.plot_priors()?

            self.model.set_train()

            plt.show()

    # TODO: Update to support MO
    def calculate_prediction_2d(self, var_0_ind, var_1_ind, in_real_units=False):
        if self.suggested_steps is None or self.need_new_suggestions:
            eval_point = deepcopy(self.obj_func_coords[0, self.obj_func_vals.argmax()])
            # print("Plotting for the point with highest known value.")
            title = "Predictions at the point with the highest known value"
        else:
            high_ind = self.suggested_steps_acq_val.argmax()
            eval_point = deepcopy(self.suggested_steps[:, high_ind:high_ind + 1]).squeeze(0)
            # print("Plotting for the suggested next step.")
            title = "Predictions at the suggested next step with highest acq val"

        n = 200

        # Note that this is de-normalised below by redefining so don't change this without changing that too.
        var_0_arr = np.linspace(self.bounds.T[var_0_ind][0], self.bounds.T[var_0_ind][1], num=n)
        var_1_arr = np.linspace(self.bounds.T[var_1_ind][0], self.bounds.T[var_1_ind][1], num=n)

        coords_arr = np.zeros([n,  n, len(self.bounds.T)])
        current_point = deepcopy(eval_point)
        for var_0_val_no, var_0_val in enumerate(var_0_arr):
            for var_1_val_no, var_1_val in enumerate(var_1_arr):
                if len(current_point) < len(self.bounds.T):
                    current_point = current_point[0]
                current_point[var_0_ind] = var_0_val
                current_point[var_1_ind] = var_1_val
                coords_arr[var_0_val_no, var_1_val_no] = current_point

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model_mean = self.model.likelihood(self.model.model(torch.tensor(coords_arr)))
            model_lower_std, model_upper_std = model_mean.confidence_region()

        model_mean = model_mean.loc.numpy()
        model_lower_std = model_lower_std
        model_upper_std = model_upper_std

        var_0_arr = var_0_arr.flatten()
        var_1_arr = var_1_arr.flatten()

        if in_real_units and self.normalise:
            model_mean = self.normaliser_y.inverse_transform(model_mean)
            model_lower_std = self.normaliser_y.inverse_transform(model_lower_std)
            model_upper_std = self.normaliser_y.inverse_transform(model_upper_std)
            var_0_arr = np.linspace(self.bounds_real_units().T[var_0_ind][0], self.bounds_real_units().T[var_0_ind][1], num=n)
            var_1_arr = np.linspace(self.bounds_real_units().T[var_1_ind][0], self.bounds_real_units().T[var_1_ind][1], num=n)

        var_0_mat, var_1_mat = np.meshgrid(var_0_arr, var_1_arr)

        return title, var_0_mat, var_1_mat, model_mean, model_lower_std, model_upper_std

    # TODO: Update to support MO
    def plot_prediction_2d_real_units(self, var_0_ind, var_1_ind, plot_suggestions=False):

        self.model.set_eval()

        title, var_0_mat, var_1_mat, model_mean, model_lower_std, model_upper_std = \
            self.calculate_prediction_2d(var_0_ind, var_1_ind, in_real_units=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.ticklabel_format(scilimits=[-3, 3])
        plt.xlabel(self.obj_func.var_names[var_0_ind])
        plt.ylabel(self.obj_func.var_names[var_1_ind])
        ax.set_zlabel('Objective')
        ax.scatter(self.obj_func_coords_real_units()[0, :, var_0_ind], self.obj_func_coords_real_units()[0, :, var_1_ind],
                   self.obj_func_vals_real_units().squeeze(0), c=self.obj_func_vals_real_units().squeeze(0),
                   cmap='seismic', marker='.', s=50, alpha=1)
        ax.plot_surface(var_0_mat, var_1_mat, model_mean.T, alpha=0.5)
        ax.plot_surface(var_0_mat, var_1_mat, model_lower_std.T, alpha=0.1)
        ax.plot_surface(var_0_mat, var_1_mat, model_upper_std.T, alpha=0.1)
        ax.set_title(title)

        if plot_suggestions:
            if self.need_new_suggestions is False:
                z_min, z_max = ax.get_zlim()
                suggested_steps_real_units = self.normaliser_x.inverse_transform(self.suggested_steps)
                for ss_ind in range(len(suggested_steps_real_units[0])):
                    ax.plot([suggested_steps_real_units[0, ss_ind, var_0_ind]]*2, [suggested_steps_real_units[0, ss_ind, var_1_ind]]*2,
                            [z_min, z_max], 'k-', linewidth=3, alpha=0.3, color='red')

        self.model.set_train()

    # TODO: Implement true obj value for multi_obj
    def plot_progress(self):

        plt.figure()

        if not self.multi_obj:

            if self.points_evaluated < self.n_init_points:

                plt.plot(self.obj_func_vals[0, :self.points_evaluated], '.', label="Init points")

            else:
                plt.plot(range(self.n_init_points), self.obj_func_vals[0, :self.n_init_points], '*b',
                         label="Init points")
                plt.plot(range(self.n_init_points, self.points_evaluated),
                         self.obj_func_vals[0, self.n_init_points: self.points_evaluated], '*', color='black',
                         label="Bayes points")

        else:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for obj_no in range(self.n_objs):
                if self.obj_func.obj_names:
                    objective_name = self.obj_func.obj_names[obj_no]
                else:
                    objective_name = f"Objective {obj_no+1}"

                if self.points_evaluated <= self.n_init_points:

                    plt.plot(self.obj_func_vals[0, :self.points_evaluated, obj_no], marker='.', color=colours[obj_no],
                             linestyle='', label=f"Init points, {objective_name}", alpha=0.6)
                    if obj_no == 0:
                        plt.plot((self.obj_func_vals * self.obj_weights).sum(2)[0, :self.points_evaluated],
                                 marker='h', color='black', linestyle='', label=f"Init points, summed",
                                 markersize=4)

                else:
                    plt.plot(range(self.n_init_points), self.obj_func_vals[0, :self.n_init_points, obj_no], marker='.',
                             color=colours[obj_no], linestyle='',
                             label=f"Init points, {objective_name }", alpha=0.6)
                    plt.plot(range(self.n_init_points, self.points_evaluated),
                             self.obj_func_vals[0, self.n_init_points: self.points_evaluated, obj_no], marker='P',
                             markersize=4, color=colours[obj_no], linestyle='',
                             label=f"Bayes points, {objective_name }", alpha=0.6)

                    if obj_no == 0:
                        plt.plot(range(self.n_init_points), (self.obj_func_vals * self.obj_weights).sum(2)
                                 [0, :self.n_init_points], marker='h', color='black', linestyle='',
                                 label=f"Init points, summed", markersize=4)
                        plt.plot(range(self.n_init_points, self.points_evaluated),
                                 (self.obj_func_vals * self.obj_weights).sum(2)[0,
                                 self.n_init_points: self.points_evaluated], marker='p', markersize=4,
                                 color='black', linestyle='', label=f"Bayes points, summed")

        if self.test_mode:
            if 'true_vals' in self.obj_func.__dict__:
                if self.normalise and self.data_fitted:
                    true_optimum = self.normaliser_y.transform(self.obj_func.function(self.obj_func.true_vals).unsqueeze(0))
                else:
                    true_optimum = self.obj_func.run(self.obj_func.true_vals)
                plt.plot([0, self.points_evaluated], [float(true_optimum)] * 2, 'k',
                         label="True best objective", linewidth=2)

        plt.legend()
        plt.xlabel("Point")
        plt.ylabel("Obj func value")

        plt.show()

    def plot_variable_values(self):

        if self.data_fitted:
            self.model.set_eval()

        plt.figure()
        plt.plot(range(self.points_evaluated), self.obj_func_coords[0], '*')
        plt.xlabel("Iteration")
        plt.ylabel("Variable value")
        if self.obj_func.var_names:
            plt.legend(self.obj_func.var_names)
        else:
            plt.legend(np.core.defchararray.add(np.array(["Variable number "] * self.n_params),
                                                np.arange(self.n_params).astype(str)))

        plt.show()

        if self.data_fitted:
            self.model.set_train()

    @staticmethod
    def close_plots():
        plt.close("all")

    # def set_kernel_train(self):
    #     self.model.model.train()
    #     self.model.likelihood.train()

    # def set_kernel_eval(self):
    #     self.model.model.eval()
    #     self.model.likelihood.eval()

    def init_steps_random(self):
        init_steps = (self.bounds[1] - self.bounds[0]) * torch.rand(self.n_init_points, self.n_params) + \
                     self.bounds[0]
        init_steps = init_steps.unsqueeze(0)
        return init_steps

    def init_steps_priors(self):
        priors_not_normalised = prior_dists(self.init_vals, self.bounds, self.stds)
        init_steps = torch.zeros(self.n_init_points, self.n_params)
        for par_no, prior in enumerate(priors_not_normalised):
            init_steps[:, par_no] = torch.tensor(prior.rvs(self.n_init_points))
        return init_steps.unsqueeze(0)

    def best_coords(self, in_real_units=False):
        if not in_real_units:
            return self.obj_func_coords[:, self.obj_func_vals.argmax()]
        else:
            best_coords = self.obj_func_coords[:, self.obj_func_vals.argmax()]
            best_coords = self.normaliser_x.inverse_transform(best_coords)
            best_coords = torch.tensor(best_coords)
            return best_coords

    def best_val(self, in_real_units=False, weighted_best=False, max_for_single_obj_ind=None):
        if max_for_single_obj_ind is None:
            best_vals_ind = (self.obj_func_vals * self.obj_weights).sum(2).argmax()
            best_vals = self.obj_func_vals[:, best_vals_ind]

            if in_real_units:
                best_vals = self.normaliser_y.inverse_transform(best_vals)
                best_vals = torch.tensor(best_vals)

            if not weighted_best:
                return best_vals

            else:
                best_val = (best_vals * self.obj_weights).sum(1)
                return best_val
        else:
            if in_real_units:
                obj_func_vals = torch.tensor(self.normaliser_y.inverse_transform(self.obj_func_vals))
            else:
                obj_func_vals = self.obj_func_vals

            best_val = obj_func_vals[0, :, max_for_single_obj_ind].max()

            return best_val

    def obj_func_coords_real_units(self):
        coords_not_normalised = self.normaliser_x.inverse_transform(self.obj_func_coords)
        coords_not_normalised = torch.tensor(coords_not_normalised)
        return coords_not_normalised

    def obj_func_vals_real_units(self):
        vals_not_normalised = self.normaliser_y.inverse_transform(self.obj_func_vals)
        vals_not_normalised = torch.tensor(vals_not_normalised)
        return vals_not_normalised

    def bounds_real_units(self):
        bounds_not_normalised = self.normaliser_x.inverse_transform(self.bounds)
        bounds_not_normalised = torch.tensor(bounds_not_normalised)
        return bounds_not_normalised

    def set_acq_func_params(self, par_name, value):
        self.acq_func.params[par_name] = value

        if self.data_fitted:
            self.refresh_acq_func()
            self.need_new_suggestions = True

    def save_optimiser(self, new_name=None):

        if new_name:
            name = new_name
        else:
            name = self.file_name

        with open(name, 'wb') as file:
            dill.dump(self, file)
        print(f"Optimiser saved as {name}")
        print("\n")
        print("\n")

    def extend_run_n_rounds(self, n_extra_rounds):

        self.n_bayes_points += n_extra_rounds * self.n_evals_per_step

        self.n_points = self.n_init_points + self.n_bayes_points
        self.n_steps = self.n_points // self.n_evals_per_step

    def load_data_from_saved_optimiser(self, file_name):
        with open(file_name, 'rb') as file:
            optimiser = dill.load(file)
        self.obj_func_coords = optimiser.obj_func_coords
        self.obj_func_vals = optimiser.obj_func_vals
        self.normaliser_x = optimiser.normaliser_x
        self.normaliser_y = optimiser.normaliser_y

        self.n_init_points = optimiser.n_init_points
        self.n_bayes_points += optimiser.n_bayes_points
        self.n_points = self.n_init_points + self.n_bayes_points
        self.n_steps = self.n_points // self.n_evals_per_step
        self.points_evaluated = optimiser.points_evaluated
        # self.current_point = optimiser.current_point
        self.current_step = optimiser.current_step

        self.init_steps = torch.tensor(optimiser.normaliser_x.inverse_transform(optimiser.init_steps))
        self.init_normaliser()
        self.refit_model()
        self.refresh_acq_func()
        self.data_fitted = True
        self.check_opt_mode()

    def set_new_acq_func(self, acq_func):
        self.acq_func = acq_func
        if self.normalise and self.data_fitted:
            self.acq_func.change_bounds(self.bounds)
        self.refresh_acq_func()
        self.need_new_suggestions = True

    def set_new_model(self, model):
        self.model = model
        self.refit_model()

    @staticmethod
    def format_list(unformatted_list):
        formatted_list = "["
        if isinstance(unformatted_list[0], list):
            for it, list_item in enumerate(unformatted_list):
                for number_ind, number in enumerate(list_item):
                    if number_ind < len(list_item) - 1:
                        formatted_list += f"{number:.2f}, "
                    elif it < len(unformatted_list) - 1:
                        formatted_list += f"{number:.2f}], ["
                    else:
                        formatted_list += f"{number:.2f}]"
        else:
            for it, list_item in enumerate(unformatted_list):
                if it < len(unformatted_list) - 1:
                    formatted_list += f"{list_item:.2f}, "
                else:
                    formatted_list += f"{list_item:.2f}]"

        return formatted_list


def load_optimiser(file_name):
    """

    :rtype: BayesOptimiser
    """
    with open(file_name, 'rb') as file:
        optimiser = dill.load(file)
    return optimiser




