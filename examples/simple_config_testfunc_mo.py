from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.gui import veropt_gui

n_init_points = 16
n_bayes_points = 64

# n_evals_per_step = 8
n_evals_per_step = 4


obj_func = PredefinedTestFunction("BraninCurrin")
# obj_func = PredefinedTestFunction("VehicleSafety")

n_objs = obj_func.n_objs

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)


veropt_gui.run(optimiser)
