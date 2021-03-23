from veropt import BayesOptimiser
# from acq_funcs import *
from veropt.obj_funcs.ocean_sims import OceanObjFunction
from veropt.kernels import *
from veropt.slurm_support import slurm_set_up


class OceanObjSimThree(OceanObjFunction):
    def __init__(self, target_min_vsf_depth_20N, measure_year=100, file_path=None):
        bounds_lower = [500, 500, 2e-6]
        bounds_upper = [1500, 1500, 2e-4]
        bounds = [bounds_lower, bounds_upper]
        n_params = 3
        n_objs = 1
        var_names = ["kappa_iso", "kappa_gm", "kappa_min"]
        obj_names = ["min_vsf_depth_20N"]

        self.measure_year = measure_year
        self.target_min_vsf_depth_20N = target_min_vsf_depth_20N
        self.file_path = file_path

        param_dic = {
            "measure_year": measure_year,
            "target_min_vsf_depth_20N": target_min_vsf_depth_20N
        }
        filetype = "overturning"

        def calc_y(overturning, param_dic):
            min_vsf_depth_20N = float(overturning["vsf_depth"][param_dic["measure_year"] - 1].min("zw")[25])
            y = - (min_vsf_depth_20N - param_dic["target_min_vsf_depth_20N"])**2
            return y, min_vsf_depth_20N

        calc_y_method = (calc_y, filetype, param_dic)

        super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
                         var_names=var_names, obj_names=obj_names, file_path=file_path)


n_init_points = 16
n_bayes_points = 48

n_evals_per_step = 8

measure_year = 100
target_min_vsf_depth_20N = -15 * 10**6

obj_func = OceanObjSimThree(target_min_vsf_depth_20N, measure_year=measure_year)

# n_params = obj_func.n_params
# n_objs = obj_func.n_objs
# bounds = obj_func.bounds

# beta = 3.0
# gamma = 0.01
#
# acq_func = PredefinedAcqFunction(bounds=bounds, n_objs=n_objs, acqfunc_name="UCB_Var", beta=beta, gamma=gamma,
#                                  n_evals_per_step=n_evals_per_step)
#
# kernel = BayesOptModel(n_params=n_params, n_objs=n_objs, model_class_list=[MaternModelBO], init_train_its=1000,
#                        using_priors=False)
#
#
# optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func=acq_func, model=kernel,
#                            n_evals_per_step=n_evals_per_step)

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

optimiser.save_optimiser()

slurm_set_up.set_up(
    optimiser.file_name, ["modi_long", "modi_short"], "global_four_degree.py", make_new_slurm_controller=True,
    using_singularity=True, image_path="~/modi_images/hpc-ocean-notebook_latest.sif", conda_env="python3")


slurm_set_up.start_opt_run("modi001")


