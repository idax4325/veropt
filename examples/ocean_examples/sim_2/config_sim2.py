from veropt import BayesOptimiser
from veropt.obj_funcs.ocean_sims import OceanObjFunction


class OceanObjSimTwo(OceanObjFunction):
    def __init__(self, target_min_vsf_depth_equator, measure_year=100, file_path=None):
        bounds_lower = [500, 2e-6]
        bounds_upper = [1500, 2e-4]
        bounds = [bounds_lower, bounds_upper]

        n_params = 2
        var_names = ["kappa_gm_iso", "kappa_min"]

        n_objs = 1
        obj_names = ["min_vsf_depth_equator"]

        self.measure_year = measure_year
        self.target_vsf_depth_min_equator = target_min_vsf_depth_equator

        param_dic = {
            "measure_year": measure_year,
            "target_min_vsf_depth_equator": target_min_vsf_depth_equator
        }
        filetype = "overturning"

        self.file_path = file_path

        def calc_y(overturning, param_dic):
            min_vsf_depth_eq = float(overturning["vsf_depth"].min("zw")[param_dic["measure_year"] - 1][20])
            y = - (min_vsf_depth_eq - param_dic["target_min_vsf_depth_equator"])**2
            return y, min_vsf_depth_eq

        calc_y_method = (calc_y, filetype, param_dic)

        super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
                         var_names=var_names, obj_names=obj_names, file_path=file_path)


n_init_points = 8
n_bayes_points = 40

n_evals_per_step = 8

measure_year = 100
target_vsf_depth_min_equator = -3200856.86124386

obj_func = OceanObjSimTwo(target_vsf_depth_min_equator, measure_year=measure_year)

optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, using_priors=False,
                           n_evals_per_step=n_evals_per_step)


optimiser.save_optimiser()


