from veropt import BayesOptimiser
from veropt.acq_funcs import *
from veropt.obj_funcs.ocean_sims import OceanObjFunction
from veropt.kernels import *
from veropt.slurm_support import slurm_set_up


class OceanObjSimOne(OceanObjFunction):
    def __init__(self, target_mean_psi_sb, measure_year=200, file_path=None):
        bounds = [500, 1500]
        n_params = 1
        n_objs = 1
        var_names = ["kappa_gm_iso"]
        obj_names = ["mean_psi_sb"]

        self.target_mean_psi_sb = target_mean_psi_sb
        self.measure_year = measure_year
        self.file_path = file_path

        param_dic = {
            "target_mean_psi_sb": target_mean_psi_sb,
            "measure_year": measure_year
        }
        filetype = "averages"

        def calc_y(averages, param_dic):
            mean_psi_sb = float(averages["psi"][param_dic["measure_year"] - 1, 0].mean())
            y = - (mean_psi_sb - param_dic["target_mean_psi_sb"])**2
            return y, mean_psi_sb

        calc_y_method = (calc_y, filetype, param_dic)

        super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
                         var_names=var_names, obj_names=obj_names, file_path=file_path)


n_init_points = 8
n_bayes_points = 24

n_evals_per_step = 8

target_psi = 158.84593135989644 * 10**6
measure_year = 200

obj_func = OceanObjSimOne(target_psi, measure_year)

# bounds = obj_func.bounds
# n_objs = obj_func.n_objs
# n_params = obj_func.n_params
#
# beta = 3.0
# gamma = 0.01
#
# acq_func = PredefinedAcqFunction(bounds, n_objs=n_objs, n_evals_per_step=n_evals_per_step, acqfunc_name="UCB_Var",
#                                  beta=beta, gamma=gamma)
#
# kernel = BayesOptModel(n_params, n_objs=n_objs, model_class_list=[MaternModelBO], init_train_its=1000)
#
# optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel,
#                            n_evals_per_step=n_evals_per_step)


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

optimiser.save_optimiser()


slurm_set_up.set_up(
    optimiser.file_name, ["modi_long", "modi_short"], "acc.py", make_new_slurm_controller=False,
    using_singularity=True, image_path="~/modi_images/hpc-ocean-notebook_latest.sif", conda_env="python3")

slurm_set_up.start_opt_run("modi002")




