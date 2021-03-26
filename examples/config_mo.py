from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.acq_funcs import *
from veropt.kernels import *
from veropt.gui import veropt_gui

n_init_points = 16
n_bayes_points = 64

# n_evals_per_step = 8
n_evals_per_step = 4


# obj_func = PredefinedTestFunction("BraninCurrin")
obj_func = PredefinedTestFunction("VehicleSafety")

n_objs = obj_func.n_objs


# alpha = 1.0
# omega = 1.0
#
# acq_func = PredefinedAcqFunction(obj_func.bounds, n_objs, n_evals_per_step, acqfunc_name='EHVI', seq_dist_punish=True,
#                                  alpha=alpha, omega=omega)
#
# model_list = n_objs * [MaternModelBO]  # Matern is the default
#
# kernel = BayesOptModel(obj_func.n_params, obj_func.n_objs, model_class_list=model_list, init_train_its=1000,
#                        using_priors=False)
#
# optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel, test_mode=False,
#                            using_priors=False, n_evals_per_step=n_evals_per_step)


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

# veropt_gui.run(optimiser)

print(optimiser.acq_func.acqfunc_name)

optimiser.run_all_opt_steps()

# po_coords, po_vals = optimiser.pareto_optimal_points()
