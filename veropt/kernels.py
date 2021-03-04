import torch
import gpytorch
import botorch
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_params):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=10, ard_num_dims=n_params))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=param_amount))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, layout, n_params):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(n_params, layout[0]))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(layout[0], layout[1]))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(layout[1], layout[2]))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(layout[2], layout[3]))


class DKModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, layout, n_params):
        super(DKModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
        #         num_mixtures=5,
        #         ard_num_dims=layout[-1],
        #         batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=layout[-1])
        self.feature_extractor = LargeFeatureExtractor(layout, n_params)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x.squeeze(0)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x.unsqueeze(0))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SMKModelBO(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_X, train_Y, n_params, num_mixtures=10):
        # squeeze output dim before passing train_Y to ExactGP. Ida: Don't know if it's necessary
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        # It says in the docs that one shouldn't use a ScaleKernel with the SMK
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=num_mixtures,
                ard_num_dims=n_params,
                batch_shape=torch.Size([1]))

        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MaternModelBO(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_X, train_Y, n_params):
        # squeeze output dim before passing train_Y to ExactGP. Don't know if it's necessary
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        # Note: Removed the ScaleKernel after I normalised the data myself
        self.covar_module = gpytorch.kernels.MaternKernel(
                ard_num_dims=n_params,
                batch_shape=torch.Size([]))
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFModelBO(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_X, train_Y, n_params):
        # squeeze output dim before passing train_Y to ExactGP. Don't know if it's necessary
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        # Note: Removed the ScaleKernel after I normalised the data myself
        self.covar_module = gpytorch.kernels.RBFKernel(
                ard_num_dims=n_params,
                batch_shape=torch.Size([]))
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MaternKernelPrior(gpytorch.kernels.Kernel):
    r"""
    Copied from GPytorch matern_kernel and modified to include a prior distribution
    """

    has_lengthscale = True

    def __init__(self, cdf, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernelPrior, self).__init__(**kwargs)
        self.nu = nu
        self.cdf = cdf

    def forward(self, x1, x2, diag=False, **params):
        if x1.requires_grad or x2.requires_grad or (self.ard_num_dims is not None and self.ard_num_dims > 1) or diag:

            # I thiiink I can do this. Maybe. Might wanna check if the lengthscale thing is ok. But is should be. Right?
            x1 = self.cdf(x1)
            x2 = self.cdf(x2)

            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return gpytorch.functions.matern_covariance.MaternCovariance().apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )


class MaternPriorModelBO(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_X, train_Y, n_params, cdf):
        # squeeze output dim before passing train_Y to ExactGP. Don't know if it's necessary
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        # Note: Removed the ScaleKernel after I normalised the data myself
        self.covar_module = MaternKernelPrior(
                cdf=cdf,
                ard_num_dims=n_params,
                batch_shape=torch.Size([]))
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKModelBO(DKModel, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_X, train_Y, feature_extractor, n_params):
        super().__init__(train_X, train_Y.squeeze(-1), gpytorch.likelihoods.GaussianLikelihood(), feature_extractor,
                         n_params)


class BayesOptModel:
    def __init__(self, n_params, n_objs, model_class_list: List[MaternModelBO] = None, init_train_its=1000,
                 train_its=200, lr=0.1, opt_params_list=None, using_priors=False, constraint_dict_list=None):

        self.n_params = n_params
        self.n_objs = n_objs
        if self.n_objs > 1:
            self.multi_obj = True
        else:
            self.multi_obj = False

        self.init_train_its = init_train_its
        self.train_its = train_its
        self.lr = lr

        self.model_list = None
        self.model_class_list = None

        self.init_model_class_list(model_class_list)

        self.model = None  # The model is initiated under init_model
        self.likelihood = None
        self.mll = None
        self.optimiser = None
        self.loss_list = []

        self.init_opt_params_list(opt_params_list)
        self.init_constraint_dict_list(constraint_dict_list)

        self.using_priors = using_priors
        self.prior_class = None

    def init_model_class_list(self, model_class_list):
        # TODO: Make these more sensible, consider printing warnings if input is wrong
        if model_class_list is None:
            self.model_class_list = [MaternModelBO] * self.n_objs

        elif not isinstance(model_class_list, list):
            self.model_class_list = [model_class_list] * self.n_objs

        elif isinstance(model_class_list, list) and len(model_class_list) < self.n_objs:
            self.model_class_list = model_class_list

            while len(self.model_class_list) < self.n_objs:
                self.model_class_list.append(MaternModelBO)

        else:
            self.model_class_list = model_class_list

    def init_opt_params_list(self, opt_params_list):
        if opt_params_list is None:
            self.opt_params_list = []

            for model_no in range(self.n_objs):
                if "Matern" in self.model_class_list[model_no].__name__ or "RBF" in self.model_class_list[model_no].__name__:
                    self.opt_params_list.append(["mean_module", "covar_module"])
                else:
                    self.opt_params_list.append(None)

        elif not isinstance(opt_params_list, list):
            self.opt_params_list = [opt_params_list] * self.n_objs

        elif isinstance(opt_params_list, list) and len(opt_params_list) < self.n_objs:
            self.opt_params_list = opt_params_list

            for model_no in range(len(self.opt_params_list), self.n_objs):
                if "Matern" in self.model_class_list[model_no].__name__ or "RBF" in self.model_class_list[model_no].__name__:
                    self.opt_params_list.append(["mean_module", "covar_module"])
                else:
                    self.opt_params_list.append(None)

        else:
            self.opt_params_list = opt_params_list

    def init_constraint_dict_list(self, constraint_dict_list):

        if constraint_dict_list is None:
            self.constraint_dict_list = []
            for model_no in range(self.n_objs):
                if "Matern" in self.model_class_list[model_no].__name__ or "RBF" in self.model_class_list[model_no].__name__:
                    self.constraint_dict_list.append({
                        "covar_module": {
                            "raw_lengthscale": [0.1, 2.0]}
                    })
                else:
                    self.constraint_dict_list.append(None)

        elif not isinstance(constraint_dict_list, list):
            self.constraint_dict_list = [constraint_dict_list] * self.n_objs

        elif isinstance(constraint_dict_list, list) and len(constraint_dict_list) < self.n_objs:
            self.constraint_dict_list = constraint_dict_list

            for model_no in range(len(self.opt_params_list), self.n_objs):
                if "Matern" in self.model_class_list[model_no].__name__ or "RBF" in self.model_class_list[model_no].__name__:
                    self.constraint_dict_list.append({
                        "covar_module": {
                            "raw_lengthscale": [0.1, 2.0]}
                    })
                else:
                    self.constraint_dict_list.append(None)

        else:
            self.constraint_dict_list = constraint_dict_list

    def eval(self, x: torch.Tensor):
        self.set_eval()
        # if self.multi_obj:
        y = self.likelihood(*self.model(*[x] * self.n_objs))
        # self.set_train()
        if not self.multi_obj:
            y = [y]
        return y
        # else:
        #     y = self.likelihood(self.model(x))
        #     self.set_train()
        #     return y

    def set_priors(self, prior_class):
        self.prior_class = prior_class

    def reset_model(self, x: torch.Tensor, y_split: Tuple):
        models = []
        if self.using_priors is False:
            for model_no in range(self.n_objs):
                models.append(self.model_class_list[model_no](x, y_split[model_no], self.n_params))
        else:
            for model_no in range(self.n_objs):
                models.append(self.model_class_list[model_no](x, y_split[model_no], self.n_params,
                                                              self.prior_class.prior_cdf))

        if self.n_objs == 1:
            self.model_list = models
            self.model = models[0]
        else:
            self.model_list = models
            self.model = botorch.models.ModelListGP(*models)

    def register_constraints(self, module: str, par_name: str, constraints: List, model_no: int):

        if self.multi_obj:

            self.model.models[model_no].__getattr__(module).register_constraint(
                par_name, gpytorch.constraints.Interval(*constraints)
            )

        else:

            self.model.__getattr__(module).register_constraint(
                par_name, gpytorch.constraints.Interval(*constraints)
            )

    def update_constraints(self):
        for model_no in range(self.n_objs):
            if self.constraint_dict_list[model_no] is not None:
                for module in self.constraint_dict_list[model_no]:
                    for var in self.constraint_dict_list[model_no][module]:
                        self.register_constraints(module, var, self.constraint_dict_list[model_no][module][var], model_no)

            # TODO: Make better. So far this is hardcoded and assumes no noise.
            if self.multi_obj:
                self.model.models[model_no].likelihood.noise_covar.register_constraint(
                    'raw_noise', gpytorch.constraints.GreaterThan(10**(-10)))
                self.model.models[model_no].likelihood.raw_noise = torch.tensor(-500.0)
            else:
                self.model.likelihood.noise_covar.register_constraint(
                    'raw_noise', gpytorch.constraints.GreaterThan(10**(-10)))
                self.model.likelihood.raw_noise = torch.tensor(-500.0)

    def view_model_hyperparameter(self, module: str, par_name: str, model_no: int):
        # self.model.__getattr__(module)
        constraint_name = par_name + "_constraint"
        if self.multi_obj:
            constraint = self.model.models[model_no].__getattr__(module).__getattr__(constraint_name)
            return constraint.transform(deepcopy(self.model.models[model_no].__getattr__(module).__getattr__(par_name)))
        else:
            constraint = self.model.__getattr__(module).__getattr__(constraint_name)
            return constraint.transform(deepcopy(self.model.__getattr__(module).__getattr__(par_name)))

    def view_constrained_hyperparameters(self):
        for model_no in range(self.n_objs):

            constraint_dict = self.constraint_dict_list[model_no]
            for module in constraint_dict:
                for par_name in constraint_dict[module]:
                    par_val = self.view_model_hyperparameter(module, par_name, model_no)
                    print(f"{par_name} has value(s):")
                    print(par_val)

    def refit_model(self, x: torch.Tensor, y: torch.Tensor):

        y_split = y.split(1, dim=2)

        self.reset_model(x, y_split)

        if self.multi_obj:
            self.likelihood = gpytorch.likelihoods.LikelihoodList(
                *[self.model.models[model_no].likelihood for model_no in range(self.n_objs)])
        else:
            self.likelihood = self.model.likelihood

        self.model.train()
        self.likelihood.train()

        if self.multi_obj:
            self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        else:
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.set_optimiser()

        self.update_constraints()

        self.train_backwards(self.init_train_its)

    def update_model(self, x: torch.Tensor, y: torch.Tensor):

        y_split = y.split(1, dim=2)

        state_dict = self.model.state_dict()

        self.reset_model(x, y_split)

        if self.multi_obj:
            self.likelihood = gpytorch.likelihoods.LikelihoodList(
                *[self.model.models[model_no].likelihood for model_no in range(self.n_objs)])
        else:
            self.likelihood = self.model.likelihood

        self.model.load_state_dict(state_dict)

        self.model.train()
        self.likelihood.train()

        if self.multi_obj:
            self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        else:
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.set_optimiser()

        self.update_constraints()

        self.train_backwards()

    def set_optimiser(self):
        opt_list = []
        for model_no in range(self.n_objs):
            if self.opt_params_list[model_no] is None:

                # for par_name, par in zip(self.model.named_parameters(), self.model.parameters()):
                #     if f'models.{model_no}' in par_name[0]:
                #         opt_list.append(par)

                if self.multi_obj:
                    opt_list.append({'params': self.model.models[model_no].parameters()})
                else:
                    opt_list.append({'params': self.model.parameters()})

            else:
                if self.multi_obj:
                    for opt_module in self.opt_params_list[model_no]:
                        opt_list.append({'params': self.model.models[model_no].__getattr__(opt_module).parameters()})
                else:
                    for opt_module in self.opt_params_list[model_no]:
                        opt_list.append({'params': self.model.__getattr__(opt_module).parameters()})

        self.optimiser = torch.optim.Adam(opt_list, lr=self.lr)

    def train_backwards(self, its: int = None):

        running_on_slurm = "SLURM_JOB_ID" in os.environ

        if its is None:
            its = self.train_its

        for train_it in range(its):
            self.optimiser.zero_grad()  # Zero gradients from previous iteration
            # output = self.model(x)
            output = self.model(*self.model.train_inputs)
            # loss = -self.mll(output, y)  # Calc loss and backprop gradients
            loss = -self.mll(output, self.model.train_targets)
            loss.backward()
            if not running_on_slurm:
                print("Training model... Iteration %d/%d - Loss: %.3f" % (train_it + 1, its, loss.item()), end="\r")
            self.loss_list.append(loss.item())
            self.optimiser.step()

        if not running_on_slurm:
            print("\n")

    def plot_loss(self):

        loss_arr = np.array(self.loss_list)

        plt.figure()
        plt.plot(loss_arr, '*')
        plt.xlabel("Backwards training iteration")
        plt.ylabel("Loss")

    def set_train(self):
        self.model.train()
        self.likelihood.train()

    def set_eval(self):
        self.model.eval()
        self.likelihood.eval()




