import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import botorch
from veropt import ObjFunction


def generate_init_conds(true_params, noise_lvl, info_types=None):
    """
    Generate initial conditions (for training algorithms) from a dict of true parameter values.

    Parameters
    ----------
    true_params : dict
        True parameter values.
    noise_lvl : float
        The level of noise, i.e. the amount the parameters are perturbed. A value of 1 corresponds to a pertubation
        of the same size as the parameter itself.
    info_types : list of str
        Kinds of information desired in the output. A list that can include any of
        'init_value', 'std', 'hard_bound' and 'dist'.
        - 'init_value' returns only the perturbed value.
        - 'std' returns the perturbed value and a standard deviation.
        - 'hard_bound' returns the perturbed value and hard boundaries.
        - 'dist' returns a probability distribution for the parameter.

    Returns
    -------
    out: dict
        Dictionary of dictionaries containing selected initial information.

    """

    if info_types is None:
        info_types = ['init_value', 'std']

    out_dict = {}

    if 'init_value' in info_types:

        init_conds = {}
        for param in true_params:
            init_conds[param] = true_params[param] + np.random.choice([1, -1]) * noise_lvl * np.abs(true_params[param]) \
                                * np.random.normal(1, 3 / 10)
        out_dict["init_value"] = init_conds

    if 'std' in info_types:

        stds = {}
        for param in true_params:
            stds[param] = 1.5 * noise_lvl * np.abs(true_params[param])

        out_dict["std"] = stds

    if 'hard_bound' in info_types:

        bounds = {}
        for param in true_params:
            bounds[param] = [init_conds[param] - 6 * noise_lvl * np.abs(true_params[param]),
                             init_conds[param] + 6 * noise_lvl * np.abs(true_params[param])]

        out_dict["bound"] = bounds

    if 'dist' in info_types:
        # TODO: Implement
        return 0

    return out_dict


class LossClass:
    def __init__(self, data, func, x_arr, negate=False):  # , torch=False
        self.data = data
        self.func = func
        self.x_arr = x_arr
        self.negate = negate
        # self.torch = torch

    def mse(self, **param):
        pred = self.func(self.x_arr, **param)
        if not self.negate:
            return mean_squared_error(self.data, pred)
        else:
            return -mean_squared_error(self.data, pred)

    def mse_row(self, row_vals):
        pred = self.func(self.x_arr, *row_vals.flatten())
        if not self.negate:
            return np.array(mean_squared_error(self.data, pred)).reshape(1, 1)
        else:
            return np.array(-mean_squared_error(self.data, pred)).reshape(1, 1)

    def mse_batch(self, batch):
        if batch.ndim == 3:
            batch = batch.squeeze(0)
        n_rows = len(batch)
        mse_arr = np.zeros(n_rows)
        for row_no in range(n_rows):
            pred = self.func(self.x_arr, *batch[row_no])
            mse_arr[row_no] = mean_squared_error(self.data, pred)

        if not self.negate:
            return mse_arr
        else:
            return -mse_arr


class FitTestFunction(ObjFunction):
    def __init__(self, fit_function, param_dic, n_objs, noise_init_conds=0.1, noise_data=0.005, obj_names=None):

        n_params = len(param_dic)
        var_names = list(param_dic.keys())

        init_conds = generate_init_conds(param_dic, noise_init_conds, info_types=['init_value', 'std', 'hard_bound'])
        init_dic, bounds_dic, stds_dic = init_conds["init_value"], init_conds["bound"], init_conds["std"]

        bounds = torch.tensor(list(bounds_dic.values())).T

        init_vals = torch.tensor(list(init_dic.values()))
        stds = torch.tensor(list(stds_dic.values()))

        self.true_vals = torch.tensor(list(param_dic.values()), dtype=torch.float64).unsqueeze(0)

        self.x_array = np.linspace(0, 5.3 * (1 / param_dic["freq"] * (2 * np.pi)), num=500)
        self.y_array = fit_function(self.x_array, **param_dic)
        self.y_array_noise = self.y_array + noise_data * np.abs(np.mean(self.y_array)) \
                             * np.random.normal(0, 1, self.y_array.size)

        self.loss_class = LossClass(self.y_array_noise, fit_function, self.x_array, negate=True)
        function = lambda x: torch.tensor(self.loss_class.mse_batch(x.detach().numpy()))

        super().__init__(function, bounds, n_params, n_objs, init_vals.unsqueeze(0), stds.unsqueeze(0),
                         var_names=var_names, obj_names=obj_names)


class PredefinedTestFunction(ObjFunction):
    def __init__(self, function_name):

        if function_name == "Hartmann":
            n_params = 6  # Can be 3, 4 or 6
            n_objs = 1
            function = botorch.test_functions.Hartmann(negate=True)
            bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
            self.true_vals = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]).unsqueeze(0)
            obj_names = function_name

        elif function_name == "Cosine8":
            n_params = 8  # Can't be changed
            n_objs = 1
            function = botorch.test_functions.Cosine8(negate=False)
            bounds = torch.tensor([[-1.0] * 8, [1.0] * 8])
            self.true_vals = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
            obj_names = function_name

        elif function_name == "Branin":
            n_params = 2  # Can only be 2
            n_objs = 1
            function = botorch.test_functions.Branin(negate=True)
            bounds = torch.tensor([(-5.0, 10.0), (0.0, 15.0)])
            self.true_vals = torch.tensor([-np.pi, 12.275]).unsqueeze(0)
            self.true_vals2 = torch.tensor([np.pi, 2.275]).unsqueeze(0)
            self.true_vals3 = torch.tensor([9.42478, 2.475]).unsqueeze(0)
            obj_names = function_name

        elif function_name == "BraninCurrin":
            n_params = 2
            n_objs = 2
            # TODO: Is the negate right? Check please
            function = botorch.test_functions.BraninCurrin(negate=True)
            bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
            obj_names = ["Branin", "Currin"]

        elif function_name == "VehicleSafety":
            n_params = 5
            n_objs = 3
            function = botorch.test_functions.VehicleSafety()
            bounds = torch.tensor([[1.0]*5, [3.0]*5])
            obj_names = [f"VeSa {obj_no+1}" for obj_no in range(n_objs)]

        super().__init__(function, bounds, n_params, n_objs, obj_names=obj_names)

    def run(self, point):
        return self.function(point.squeeze(0))


class PredefinedFitTestFunction(FitTestFunction):
    def __init__(self, function_name, noise_init_conds=0.1, noise_data=0.005):

        if function_name == "sine_1param":
            fit_function = self.sine_1param
            param_dic = {"freq": 5}

        elif function_name == "sine_2params_offset":
            fit_function = self.sine_2params_offset
            param_dic = {"freq": 5, "offset": 7}

        elif function_name == "sine_2params_amplitude":
            fit_function = self.sine_2params_amplitude
            param_dic = {"freq": 5, "amplitude": 7}

        elif function_name == "sine_3params":
            fit_function = self.sine_3params
            param_dic = {"freq": 5, "offset": 7, "amplitude": 3}

        elif function_name == "sine_addpara_4params":
            fit_function = self.sine_addpara_4params
            param_dic = {"freq": 5, "offset": 7, "amplitude": 3, "para_amp": 6}

        elif function_name == "sine_multpara_4params":
            fit_function = self.sine_multpara_4params
            param_dic = {"freq": 5, "offset": 7, "amplitude": 3, "para_amp": 6}

        elif function_name == "sine_sum":
            fit_function = self.sine_sum
            param_dic = {"freq": 3, "freq2": 5, "freq3": 2}

        elif function_name == "sine_sum2_linear":
            fit_function = self.sine_sum2_linear
            param_dic = {"freq": 6, "freq2": 1, "linamp": 0.5}

        n_dims = 1

        super().__init__(fit_function, param_dic, n_dims, noise_init_conds, noise_data, obj_names=function_name)

    @staticmethod
    def sine_1param(x, freq):
        return np.sin(x * freq)

    @staticmethod
    def sine_2params_offset(x, freq, offset):
        return np.sin(x * freq) + offset

    @staticmethod
    def sine_2params_amplitude(x, freq, amplitude):
        return np.sin(x * freq) * amplitude

    @staticmethod
    def sine_3params(x, freq, offset, amplitude):
        return np.sin(x * freq) * amplitude + offset

    @staticmethod
    def sine_addpara_4params(x, freq, offset, amplitude, para_amp):
        return np.sin(x * freq) * amplitude + offset + para_amp * x ** 2

    @staticmethod
    def sine_multpara_4params(x, freq, offset, amplitude, para_amp):
        return np.sin(x * freq) * amplitude * para_amp * x ** 2 + offset

    @staticmethod
    def sine_sum(x, freq, freq2, freq3):
        return np.sin(x * freq) + np.sin(x * freq2) + np.sin(x * freq3)

    @staticmethod
    def sine_sum2_linear(x, freq, freq2, linamp):
        return np.sin(x * freq) + np.sin(x * freq2) + x * linamp



