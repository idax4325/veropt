from veropt import load_optimiser
from veropt import veropt_gui
import matplotlib.pyplot as plt
# from acq_funcs import PredefinedAcqFunction
# from kernels import BayesOptModel, RBFModelBO

optimiser = load_optimiser("Optimiser_UCB_Var_example.pkl")

# simple_gui.run(optimiser)
optimiser.plot_prediction(0, 0)

# plt.title("")

# ax = plt.gca()
fig = plt.gcf()
axes = fig.axes
axes[0].set_title("")
axes[0].set_ylabel("Obj. Func.")
axes[1].set_xlabel("Parameter Value")
axes[0].get_children()[-2].get_texts()[0].set_text("Evaluated Points")
axes[0].get_children()[1].set_label("Evaluated Points")
fig.set_size_inches(10, 5)
plt.tight_layout()

##

# beta = 3.0
# gamma = 0.01
#
# n_evals_per_step = 4
#
# n_objs = 1
#
# acq_func = PredefinedAcqFunction(optimiser.obj_func.bounds, n_objs, n_evals_per_step, acqfunc_name="UCB_Var", beta=beta,
#                                  gamma=gamma, seq_dist_punish=True)
#
# optimiser.set_new_acq_func(acq_func)


# kernel = BayesOptModel(optimiser.obj_func.n_params, n_objs, model_class_list=[RBFModelBO], init_train_its=1000,
#                        using_priors=False)
#
# optimiser.set_new_model(kernel)

##


