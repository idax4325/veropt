import botorch
import torch
import numpy as np
from copy import deepcopy
from scipy import optimize
import warnings
from typing import List


class UpperConfidenceBoundRandom(botorch.acquisition.AnalyticAcquisitionFunction):
    from typing import Optional, Union
    from torch import Tensor
    from botorch.models.model import Model
    from botorch.acquisition.objective import ScalarizedObjective
    from botorch.utils.transforms import t_batch_mode_transform

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            gamma: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        self.register_buffer("beta", beta)
        self.register_buffer("gamma", gamma)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.gamma = self.gamma.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = self.beta.expand_as(mean) * variance.sqrt()
        rand_number = (torch.randn(1).to(X) * self.gamma).expand_as(mean)
        if self.maximize:
            return mean + delta + rand_number
        else:
            return mean - delta + rand_number


class UpperConfidenceBoundRandomVar(botorch.acquisition.AnalyticAcquisitionFunction):
    from typing import Optional, Union
    from torch import Tensor
    from botorch.models.model import Model
    from botorch.acquisition.objective import ScalarizedObjective
    from botorch.utils.transforms import t_batch_mode_transform

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            gamma: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        self.register_buffer("beta", beta)
        self.register_buffer("gamma", gamma)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        self.beta = self.beta.to(X)
        self.gamma = self.gamma.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = self.beta.expand_as(mean) * variance.sqrt()
        rand_number = delta * (torch.randn(1).to(X) * self.gamma).expand_as(mean)
        if self.maximize:
            return mean + delta + rand_number
        else:
            return mean - delta + rand_number

    def perturb_opt_acq_result(self, candidate_point, bounds, n_initsearch=1000, n_randsearch=5000):

        posterior = self.model.posterior

        post_candidate = posterior(candidate_point)
        candidate_value = post_candidate.mean
        candidate_var = post_candidate.variance

        local_bounds = torch.zeros(bounds.shape)

        for parameter_ind in range(len(bounds.T)):

            par_vals = torch.linspace(bounds.T[parameter_ind][0], bounds.T[parameter_ind][1], steps=n_initsearch)

            all_pars_vals = np.repeat(deepcopy(candidate_point), n_initsearch, axis=0)
            all_pars_vals[:, parameter_ind] = par_vals

            f_0 = candidate_value - candidate_var.sqrt() * (self.beta + self.beta * self.gamma * 1.6)

            post_x = posterior(all_pars_vals)

            g_x = post_x.mean + post_x.variance.sqrt() * (self.beta + self.beta * self.gamma * 1.6)

            # TODO: Debug the bounds, maybe set the 1.6 down a bit? Idk. It's a bit troubling that beta keeps the
            #  bounds wide even when gamma is low

            lowerbound_ind = np.argmax(g_x > f_0)
            upperbound_ind = n_initsearch - 1 - torch.tensor(np.argmax(np.flip(np.array(g_x > f_0))))

            local_bounds.T[parameter_ind, 0] = par_vals[int(lowerbound_ind)]
            local_bounds.T[parameter_ind, 1] = par_vals[upperbound_ind]

        rand_search_pars = torch.rand([n_randsearch, len(bounds.T)])

        for parameter_ind in range(len(bounds.T)):
            rand_search_pars[:, parameter_ind] = (local_bounds.T[parameter_ind, 1] -
                                                  local_bounds.T[parameter_ind, 0]) * \
                                                 rand_search_pars[:, parameter_ind] + local_bounds.T[parameter_ind, 0]

        rand_search_vals = torch.zeros(n_randsearch)

        for par_vals_ind in range(len(rand_search_vals)):
            rand_search_vals[par_vals_ind] = self.forward(rand_search_pars[par_vals_ind].unsqueeze(0).unsqueeze(0))

        print("Bounds: ", bounds)
        print("Local Bounds: ", local_bounds)

        return rand_search_pars[rand_search_vals.argmax()].unsqueeze(0)


class qUpperConfidenceBoundRandomVar(botorch.acquisition.monte_carlo.MCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.
    NB: With noise! :D

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1000)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """
    from typing import Optional, Union
    from torch import Tensor
    from botorch.models.model import Model
    from botorch.acquisition.monte_carlo import MCSampler, MCAcquisitionObjective
    from botorch.utils.transforms import t_batch_mode_transform, concatenate_pending_points

    def __init__(
        self,
        model: Model,
        beta: float,
        gamma: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. Defaults to
                `SobolQMCNormalSampler(num_samples=500, collapse_batch_dims=True)`
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
        """
        super().__init__(
            model=model, sampler=sampler, objective=objective, X_pending=X_pending
        )
        import math
        self.beta_prime = math.sqrt(beta * math.pi / 2)
        self.gamma = gamma

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.sampler(posterior)
        obj = self.objective(samples)
        mean = obj.mean(dim=0)
        rand_number = self.beta_prime * (obj - mean).abs() * (torch.randn(1).to(X) * self.gamma).expand_as(mean)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs() + rand_number
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


class UpperConfidenceBoundRandomVarDist(botorch.acquisition.AnalyticAcquisitionFunction):
    from typing import Optional, Union
    from torch import Tensor
    from botorch.models.model import Model
    from botorch.acquisition.objective import ScalarizedObjective
    from botorch.utils.transforms import t_batch_mode_transform

    def __init__(
            self,
            model: Model,
            beta: Union[float, Tensor],
            gamma: Union[float, Tensor],
            alpha: Union[float, Tensor],
            omega: Union[float, Tensor],
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = True,
    ) -> None:
        super().__init__(model=model, objective=objective)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        if not torch.is_tensor(omega):
            omega = torch.tensor(omega)
        self.register_buffer("beta", beta)
        self.register_buffer("gamma", gamma)
        self.register_buffer("alpha", alpha)
        self.register_buffer("omega", omega)

    # @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor, other_points=None) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.
            other_points: List containing the other points chosen in this step

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        if other_points is None:
            other_points = []
        X = X if X.dim() > 2 else X.unsqueeze(0)
        self.beta = self.beta.to(X)
        self.gamma = self.gamma.to(X)
        self.alpha = self.alpha.to(X)
        self.omega = self.omega.to(X)
        posterior = self._get_posterior(X=X)
        batch_shape = X.shape[:-2]
        mean = posterior.mean.view(batch_shape)
        variance = posterior.variance.view(batch_shape)
        delta = self.beta.expand_as(mean) * variance.sqrt()
        rand_number = delta * (torch.randn(1).to(X) * self.gamma).expand_as(mean)

        # TODO: Check computations
        # TODO: THIS IS WRONG. ADD ALPHA
        proximity_punish = torch.tensor([0.0])
        scaling = (mean + delta) * self.omega
        for point in other_points:
            proximity_punish += scaling * torch.exp(-((torch.sum(X - point) / self.alpha)**2))
        
        if self.maximize:
            return mean + delta + rand_number - proximity_punish
        else:
            return mean - delta + rand_number + proximity_punish


class AcqFunction:
    def __init__(self, function_class, optimiser, bounds, n_objs, params=None, n_evals_per_step=1, acqfunc_name=None):

        # TODO: Implement a default optimiser

        self.function_class = function_class
        self.optimiser = optimiser
        self.bounds = bounds

        self.n_objs = n_objs
        if self.n_objs > 1:
            self.multi_obj = True
        else:
            self.multi_obj = False

        self.params = params
        self.n_evals_per_step = n_evals_per_step
        self.acqfunc_name = acqfunc_name

        self.function = None

    def refresh(self, model, **kwargs):
        if self.params is None:
            self.function = self.function_class(model=model, **kwargs)
        else:
            self.function = self.function_class(model=model, **self.params, **kwargs)

    def suggest_point(self):
        return self.optimiser.optimise(self.function)

    def change_bounds(self, new_bounds):
        self.bounds = new_bounds
        self.optimiser.bounds = new_bounds


class PredefinedAcqFunction(AcqFunction):
    def __init__(self, bounds, n_objs, n_evals_per_step, acqfunc_name="UCB_Var", optimiser_name=None,
                 seq_dist_punish=True, **kwargs):

        params = {}

        if acqfunc_name == "EI":
            if n_evals_per_step == 1 or seq_dist_punish:
                acq_func_class = botorch.acquisition.analytic.ExpectedImprovement
            else:
                acq_func_class = botorch.acquisition.monte_carlo.qExpectedImprovement

        elif acqfunc_name == "UCB":

            if "beta" in kwargs:
                beta = kwargs["beta"]
            else:
                beta = 3.0

            params["beta"] = beta

            if n_evals_per_step == 1 or seq_dist_punish:

                acq_func_class = botorch.acquisition.analytic.UpperConfidenceBound

            else:
                acq_func_class = botorch.acquisition.monte_carlo.qUpperConfidenceBound

        elif acqfunc_name == "UCB_Var":

            if "beta" in kwargs:
                beta = kwargs["beta"]
            else:
                beta = 3.0

            if "gamma" in kwargs:
                gamma = kwargs["gamma"]
            else:
                gamma = 0.01

            params["beta"] = beta
            params["gamma"] = gamma

            if n_evals_per_step == 1 or seq_dist_punish:

                acq_func_class = UpperConfidenceBoundRandomVar

            else:
                acq_func_class = qUpperConfidenceBoundRandomVar

        # TODO: Check whether there's too many objectives for EHVI? (And maybe for the q ver too)
        elif acqfunc_name == "EHVI":
            acq_func_class = botorch.acquisition.multi_objective.ExpectedHypervolumeImprovement

        elif acqfunc_name == 'qEHVI':
            acq_func_class = botorch.acquisition.multi_objective.qExpectedHypervolumeImprovement

        self.seq_dist_punish = seq_dist_punish

        if seq_dist_punish is True:
            if "alpha" in kwargs:
                alpha = kwargs["alpha"]
            else:
                alpha = 1.0

            if "omega" in kwargs:
                omega = kwargs["omega"]
            else:
                omega = 1.0

            params_seq_opt = {
                "alpha": alpha,
                "omega": omega
            }
        else:
            params_seq_opt = None

        optimiser = PredefinedAcqOptimiser(bounds, n_objs, n_evals_per_step=n_evals_per_step, optimiser_name=optimiser_name,
                                           seq_dist_punish=seq_dist_punish, params_seq_opt=params_seq_opt)

        super().__init__(acq_func_class, optimiser, bounds, n_objs, n_evals_per_step=n_evals_per_step, acqfunc_name=acqfunc_name)

        self.params = params


class OptimiseWithDistPunish:
    def __init__(self, alpha, omega):
        self.alpha = alpha
        self.omega = omega

    def add_dist_punishment(self, x, acq_func_val, other_points):
        proximity_punish = torch.tensor([0.0])
        # scaling = (mean + delta) * self.omega
        scaling = acq_func_val * self.omega
        for point in other_points:
            proximity_punish += scaling * torch.exp(-((torch.sum(x - point) / self.alpha) ** 2))

        return acq_func_val - proximity_punish


class AcqOptimiser:
    def __init__(self, bounds, function, n_objs, n_evals_per_step=1):  # , serial_opt=False
        self.bounds = bounds
        self.n_evals_per_step = n_evals_per_step
        # self.serial_opt = serial_opt

        self.n_objs = n_objs
        if self.n_objs > 1:
            self.multi_obj = True
        else:
            self.multi_obj = False

        self.function = function

    def optimise(self, acq_func):
        return self.function(acq_func)
        # if not self.serial_opt:
        #     return self.function(acq_func)
        # else:
        #     return self.optimise_serial(acq_func)

    # def optimise_serial(self, acq_func):
    #     candidates = []
    #     for candidate_no in range(self.n_evals_per_step):
    #         candidates.append(self.function(acq_func, candidates))
    #         print(f"Found point {candidate_no + 1} of {self.n_evals_per_step}.")
    #
    #     candidates = torch.stack(candidates, dim=1).squeeze(0)
    #
    #     return candidates


class PredefinedAcqOptimiser(AcqOptimiser):
    def __init__(self, bounds, n_objs, n_evals_per_step=1, optimiser_name=None, seq_dist_punish=False,
                 params_seq_opt=None):

        if optimiser_name is None:
            if n_evals_per_step > 1:
                if seq_dist_punish:
                    self.optimiser_name = 'dual_annealing'
                else:
                    self.optimiser_name = 'botorch'
            else:
                self.optimiser_name = 'dual_annealing'

        else:
            self.optimiser_name = optimiser_name

        if params_seq_opt is None and seq_dist_punish is True:
            params_seq_opt = {
                'alpha': 1.0,
                'omega': 1.0
            }

        if self.optimiser_name == 'dual_annealing':
            function = self.dual_annealing

        elif self.optimiser_name == 'botorch':
            function = self.botorch_optim

        if n_evals_per_step < 2:
            self.seq_dist_punish = False
        else:
            self.seq_dist_punish = seq_dist_punish

        if self.seq_dist_punish is True:
            self.seq_optimiser = OptimiseWithDistPunish(params_seq_opt['alpha'], params_seq_opt['omega'])

        super(PredefinedAcqOptimiser, self).__init__(bounds, function, n_objs, n_evals_per_step=n_evals_per_step)

    def optimise(self, acq_func):
        if not self.seq_dist_punish:
            return self.function(acq_func)
        else:
            return self.optimise_sequentially_w_dist_punisher(acq_func)

    def optimise_sequentially_w_dist_punisher(self, acq_func):

        def dist_punish_wrapper(x, other_points):
            acq_func_val = acq_func(x)

            new_acq_func_val = self.seq_optimiser.add_dist_punishment(x, acq_func_val, other_points)

            return new_acq_func_val

        # TODO: DEBUG THIS
        #  (Currently the first two points are always the same)

        candidates = []
        for candidate_no in range(self.n_evals_per_step):
            candidates.append(self.function(lambda x: dist_punish_wrapper(x, candidates)))
            print(f"Found point {candidate_no + 1} of {self.n_evals_per_step}.")

        candidates = torch.stack(candidates, dim=1).squeeze(0)

        return candidates

    # TODO: Change to any scipy optimiser
    def dual_annealing(self, acq_func):

        acq_opt_result = optimize.dual_annealing(
            func=lambda x: -acq_func(torch.tensor(x).unsqueeze(0)).detach().numpy(),
            bounds=self.bounds.T,
            maxiter=1000)

        candidates, acq_fun_value = [torch.tensor(acq_opt_result.x).unsqueeze(0),
                                     -torch.tensor(acq_opt_result.fun).unsqueeze(0)]
        return candidates

    def botorch_optim(self, acq_func):
        # TODO: Make these parameters changeable from the outside
        # restarts = 10
        # raw_samples = 500
        # restarts = 2
        # raw_samples = 50
        # restarts = 50
        # raw_samples = 1000

        restarts = 500
        raw_samples = 10000

        method = "L-BFGS-B"

        candidates, acq_fun_value = botorch.optim.optimize.optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=self.n_evals_per_step,
            num_restarts=restarts,
            raw_samples=raw_samples,  # used for intialization heuristic
            options={
                "method": method
            }
        )

        return candidates


