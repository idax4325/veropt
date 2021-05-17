from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.gui import veropt_gui

n_init_points = 24
n_bayes_points = 64

n_evals_per_step = 4

# obj_func = PredefinedTestFunction("Hartmann")
# obj_func = PredefinedFitTestFunction("sine_sum")
obj_func = PredefinedFitTestFunction("sine_1param")
# obj_func = PredefinedFitTestFunction("sine_2params_offset")
# obj_func = PredefinedFitTestFunction("sine_3params")


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

optimiser.run_all_opt_steps()

# veropt_gui.run(optimiser)
