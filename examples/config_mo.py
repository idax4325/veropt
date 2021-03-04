from BayesOpt_main import BayesOptimiser
from test_functions import *
from acq_funcs import *
from kernels import *
import simple_gui
import botorch

n_init_points = 16
n_bayes_points = 32*2
# beta = 3.0
# gamma = 0.01
alpha = 1.0
omega = 1.0

# n_evals_per_step = 8
n_evals_per_step = 4


# obj_func = PredefinedTestFunction("BraninCurrin")
obj_func = PredefinedTestFunction("VehicleSafety")

n_objs = obj_func.n_objs

# acq_func = PredefinedAcqFunction(obj_func.bounds, acqfunc_name="UCB_Var_Dist", beta=beta, gamma=gamma,
#                                  n_evals_per_step=n_evals_per_step)

# acq_optimiser = PredefinedAcqOptimiser(obj_func.bounds, n_objs=n_objs, n_evals_per_step=n_evals_per_step,
#                                        optimiser_name='botorch')
# acq_optimiser = PredefinedAcqOptimiser(obj_func.bounds, n_objs=n_objs, n_evals_per_step=n_evals_per_step,
#                                        optimiser_name='dual_annealing', serial_opt=True)

# acq_func = AcqFunction(botorch.acquisition.multi_objective.qExpectedHypervolumeImprovement, acq_optimiser,
#                        obj_func.bounds, n_objs, n_evals_per_step=n_evals_per_step, acqfunc_name="qEHVI")

acq_func = PredefinedAcqFunction(obj_func.bounds, n_objs, n_evals_per_step, acqfunc_name='EHVI', seq_dist_punish=True,
                                 alpha=alpha, omega=omega)

# acq_func = AcqFunction(botorch.acquisition.multi_objective.ExpectedHypervolumeImprovement, acq_optimiser,
#                        obj_func.bounds, n_objs, n_evals_per_step=n_evals_per_step, acqfunc_name="EHVI")

model_list = n_objs * [MaternModelBO]  # Matern is the default

kernel = BayesOptModel(obj_func.n_params, obj_func.n_objs, model_class_list=model_list, init_train_its=1000,
                       using_priors=False)

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel, test_mode=False,
                           using_priors=False, n_evals_per_step=n_evals_per_step)

# for i in range(4):
#     optimiser.run_opt_step()
#
# optimiser.plot_prediction(0, 0, in_real_units=True)

# optimiser.run_all_opt_steps()

simple_gui.run(optimiser)
