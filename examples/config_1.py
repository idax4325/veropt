from BayesOpt_main import BayesOptimiser
from test_functions import *
from acq_funcs import *
from kernels import *
import simple_gui

# n_init_points = 8
# n_bayes_points = 32*2
n_init_points = 8*8
n_bayes_points = 32*2
beta = 3.0
gamma = 0.01

n_evals_per_step = 4

n_objs = 1

obj_func = PredefinedTestFunction("Hartmann")
# obj_func = PredefinedFitTestFunction("sine_sum")
# obj_func = PredefinedFitTestFunction("sine_1param")  # , noise_init_conds=10.0
# obj_func = PredefinedFitTestFunction("sine_2params_offset")
# obj_func = PredefinedFitTestFunction("sine_3params")

# acq_func = PredefinedAcqFunction(obj_func.bounds, acqfunc_name="UCB_Var", beta=beta, gamma=gamma,
#                                  n_evals_per_step=n_evals_per_step)
acq_func = PredefinedAcqFunction(obj_func.bounds, n_objs, n_evals_per_step, acqfunc_name="UCB_Var", beta=beta,
                                 gamma=gamma, seq_dist_punish=True)

# kernel = BayesOptModel(obj_func.n_params, model_class=MaternPriorModelBO, using_priors=True)
kernel = BayesOptModel(obj_func.n_params, n_objs, model_class_list=[MaternModelBO], init_train_its=1000,
                       using_priors=False)


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel, test_mode=False,
                           using_priors=False, n_evals_per_step=n_evals_per_step)


# for i in range(5):
#     optimiser.run_opt_step()

# optimiser.run_all_opt_steps()

simple_gui.run(optimiser)
