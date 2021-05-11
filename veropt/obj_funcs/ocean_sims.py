import sys
# from subprocess import Popen, PIPE
from veropt import ObjFunction, load_optimiser
import os
import xarray as xr
import re
import torch
import datetime
import dill
import pandas as pd
import numpy as np


class OceanObjFunction(ObjFunction):
    def __init__(self, bounds, n_params, n_objs, calc_y_method, var_names, obj_names=None, file_path=None):

        self.saver_class = SaverOceanSim(var_names)
        saver = self.saver_class.save_vals_pandas

        calc_y_func, filetype, param_dic = calc_y_method

        self.loader_class = LoaderOceanSim(calc_y_func, filetype, param_dic, file_path)
        loader = self.loader_class.load_all_xy_to_optimiser

        super().__init__(function=None, bounds=bounds, n_params=n_params, n_objs=n_objs, saver=saver, loader=loader,
                         var_names=var_names, obj_names=obj_names)


# class OceanObjSimOne(OceanObjFunction):
#     def __init__(self, target_mean_psi_sb, measure_year=200, file_path=None):
#         bounds = [500, 1500]
#         n_params = 1
#         n_objs = 1
#         var_names = ["kappa_gm_iso"]
#         obj_names = ["mean_psi_sb"]
#
#         self.target_mean_psi_sb = target_mean_psi_sb
#         self.measure_year = measure_year
#         self.file_path = file_path
#
#         param_dic = {
#             "target_mean_psi_sb": target_mean_psi_sb,
#             "measure_year": measure_year
#         }
#         filetype = "averages"
#
#         def calc_y(averages, param_dic):
#             mean_psi_sb = float(averages["psi"][param_dic["measure_year"] - 1, 0].mean())
#             y = - (mean_psi_sb - param_dic["target_mean_psi_sb"]) ** 2
#             return y, mean_psi_sb
#
#         calc_y_method = (calc_y, filetype, param_dic)
#
#         super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
#                          var_names=var_names, obj_names=obj_names, file_path=file_path)


# class OceanObjSimTwo(OceanObjFunction):
#     def __init__(self, target_min_vsf_depth_equator, measure_year=100, file_path=None):
#         bounds_lower = [500, 2e-6]
#         bounds_upper = [1500, 2e-4]
#         bounds = [bounds_lower, bounds_upper]
#         n_params = 2
#         n_objs = 1
#         var_names = ["kappa_gm", "kappa_min"]
#         obj_names = ["min_vsf_depth_equator"]
#
#         self.measure_year = measure_year
#         self.target_vsf_depth_min_equator = target_min_vsf_depth_equator
#
#         param_dic = {
#             "measure_year": measure_year,
#             "target_min_vsf_depth_equator": target_min_vsf_depth_equator
#         }
#         filetype = "overturning"
#
#         self.file_path = file_path
#
#         def calc_y(overturning, param_dic):
#             min_vsf_depth_eq = float(overturning["vsf_depth"].min("zw")[param_dic["measure_year"] - 1][20])
#             y = - (min_vsf_depth_eq - param_dic["target_min_vsf_depth_equator"])**2
#             return y, min_vsf_depth_eq
#
#         calc_y_method = (calc_y, filetype, param_dic)
#
#         super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
#                          var_names=var_names, obj_names=obj_names, file_path=file_path)


# class OceanObjSimThree(OceanObjFunction):
#     def __init__(self, target_min_vsf_depth_20N, measure_year=100, file_path=None):
#         bounds_lower = [500, 500, 2e-6]
#         bounds_upper = [1500, 1500, 2e-4]
#         bounds = [bounds_lower, bounds_upper]
#         n_params = 3
#         n_objs = 1
#         var_names = ["kappa_iso", "kappa_gm", "kappa_min"]
#         obj_names = ["min_vsf_depth_20N"]
#
#         self.measure_year = measure_year
#         self.target_min_vsf_depth_20N = target_min_vsf_depth_20N
#         self.file_path = file_path
#
#         param_dic = {
#             "measure_year": measure_year,
#             "target_min_vsf_depth_20N": target_min_vsf_depth_20N
#         }
#         filetype = "overturning"
#
#         def calc_y(overturning, param_dic):
#             min_vsf_depth_20N = float(overturning["vsf_depth"][param_dic["measure_year"] - 1].min("zw")[25])
#             y = - (min_vsf_depth_20N - param_dic["target_min_vsf_depth_20N"])**2
#             return y, min_vsf_depth_20N
#
#         def calc_y_log(overturning, param_dic):
#             min_vsf_depth_20N = float(overturning["vsf_depth"][param_dic["measure_year"] - 1].min("zw")[25])
#             y = - np.log((min_vsf_depth_20N - param_dic["target_min_vsf_depth_20N"])**2)
#             return y, min_vsf_depth_20N
#
#         def calc_y_logx(overturning, param_dic):
#             min_vsf_depth_20N = float(overturning["vsf_depth"][param_dic["measure_year"] - 1].min("zw")[25])
#
#             target_min_vsf_depth_20N = param_dic["target_min_vsf_depth_20N"]
#
#             y = - (np.log(min_vsf_depth_20N / target_min_vsf_depth_20N))**2
#
#             if np.isnan(y):
#                 y = -10**100
#
#             return y, min_vsf_depth_20N
#
#         calc_y_method = (calc_y, filetype, param_dic)
#
#         super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
#                          var_names=var_names, obj_names=obj_names, file_path=file_path)


# class OceanObjSimThreeTest(OceanObjFunction):
#     def __init__(self, target_min_vsf_depth_20N, measure_year=100, file_path=None):
#         bounds_lower = [500]
#         bounds_upper = [1500]
#         bounds = [bounds_lower, bounds_upper]
#         n_params = 1
#         n_objs = 1
#         var_names = ["kappa_iso"]
#         obj_names = ["min_vsf_depth_20N"]
#
#         self.measure_year = measure_year
#         self.target_min_vsf_depth_20N = target_min_vsf_depth_20N
#
#         param_dic = {
#             "measure_year": measure_year,
#             "target_min_vsf_depth_20N": target_min_vsf_depth_20N
#         }
#         filetype = "overturning"
#
#         self.file_path = file_path
#
#         def calc_y(overturning, param_dic):
#             min_vsf_depth_20N = float(overturning["vsf_depth"][param_dic["measure_year"]].min("zw")[25])
#             y = - (min_vsf_depth_20N - param_dic["target_min_vsf_depth_20N"]) ** 2
#             return y
#
#         calc_y_method = (calc_y, filetype, param_dic)
#
#         super().__init__(bounds=bounds, n_params=n_params, n_objs=n_objs, calc_y_method=calc_y_method,
#                          var_names=var_names, obj_names=obj_names, file_path=file_path)


class SaverOceanSim:
    def __init__(self, var_names):
        self.var_names = var_names

    def save_vals_pandas(self, suggested_steps, current_step, filename=None):

        if filename is None:
            filename = "suggestions_step" + str(current_step) + "_time_" + \
                       datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".pkl"

        identifiers = torch.randint(int(1e10), (len(suggested_steps.squeeze(0)),))

        suggested_steps_pd = pd.DataFrame(data=suggested_steps.squeeze(0).numpy(), index=identifiers.tolist(),
                                          columns=self.var_names)

        with open(filename, 'wb') as file:
            dill.dump(suggested_steps_pd, file)

        return filename


def find_files(filetype="averages", path=None):
    if path:
        files = os.listdir(path)
    else:
        files = os.listdir()

    id_list = []
    file_list = []
    time_start_list = []
    for file in files:
        if '.' + filetype + '.nc' in file:

            if path:
                filepath = path + "/" + file
            else:
                filepath = file  # os.getcwd() + "/" + file

            file_list.append(filepath)
            id_search = re.search('id_(.*).' + filetype, file)
            id_list.append(float(id_search.group(1)))

            ds = xr.open_dataset(filepath)
            time_start_list.append(int(ds["Time"][0] / 365))

    return file_list, id_list, time_start_list


def find_suggestion_files(path=None):
    if path:
        files = os.listdir(path)
    else:
        files = os.listdir()

    file_list = []
    for file in files:
        if 'suggestions_' in file:

            if path:
                filepath = path + "/" + file
            else:
                filepath = file

            file_list.append(filepath)

    return file_list


class LoaderOceanSim:
    def __init__(self, function, filetype, param_dic, file_path=None, save_absolute_results=True):

        self.function = function
        self.filetype = filetype

        self.param_dic = param_dic

        self.file_path = file_path
        self.save_absolute_results = save_absolute_results
        self.already_loaded_ids = []
        self.already_loaded_files = []

    @staticmethod
    def load_x_to_sim(identifier, filename):

        with open(filename, 'rb') as file:
            suggested_steps_pd = dill.load(file)

        for row_identifier, row in suggested_steps_pd.iterrows():
            if int(row_identifier) == int(identifier):
                return row

    def load_all_xy_to_optimiser(self):

        # Find all relevant files

        full_file_list, full_id_list, full_time_start_list = find_files(filetype=self.filetype, path=self.file_path)

        file_list = []
        id_list = []
        time_start_list = []

        for no, id in enumerate(full_id_list):
            if id not in self.already_loaded_ids:
                id_list.append(id)
                file_list.append(full_file_list[no])
                time_start_list.append(full_time_start_list[no])
                self.already_loaded_ids.append(id)

        if not id_list:
            return None, None

        # Open suggested steps file

        suggested_steps_file_list = find_suggestion_files(path=self.file_path)
        suggested_steps_pd_list = []

        for filename in suggested_steps_file_list:
            if filename not in self.already_loaded_files:
                with open(filename, 'rb') as file:
                    suggested_steps_pd_list.append(dill.load(file))
                    self.already_loaded_files.append(filename)

        # Load x values matching the new identifiers

        new_x = torch.zeros([len(id_list), len(suggested_steps_pd_list[0].columns)])

        for suggested_steps_pd in suggested_steps_pd_list:
            for identifier, row in suggested_steps_pd.iterrows():
                if identifier in id_list:
                    ind = id_list.index(identifier)
                    new_x[ind] = torch.tensor(row)

        # Load datasets

        datasets = []
        for filepath, id in zip(file_list, id_list):
            datasets.append(xr.open_dataset(filepath))

        # Calculate objective

        if self.save_absolute_results:
            new_results = torch.zeros(len(file_list))
        new_y = torch.zeros(len(file_list))
        for file_no in range(len(file_list)):
            try:
                if not self.save_absolute_results:
                    new_y[file_no] = self.function(datasets[file_no], self.param_dic)
                else:
                    new_y[file_no], new_results[file_no] = self.function(datasets[file_no], self.param_dic)
            except IndexError:
                print(f"Import failed for file with id {int(id_list[file_no])}")

        # TODO: Put the 'new_results' somewhere

        return new_x, new_y


def load_data_to_sim(kwargs):

    identifier = kwargs["identifier"]
    del kwargs["identifier"]

    optimiser_path = kwargs["optimiser"]
    optimiser = load_optimiser(optimiser_path)
    del kwargs["optimiser"]

    var_vals = optimiser.obj_func.loader_class.load_x_to_sim(identifier, optimiser.suggested_steps_filename)

    return identifier, var_vals, kwargs
